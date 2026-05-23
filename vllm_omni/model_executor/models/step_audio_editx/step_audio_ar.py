import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
import numpy as np
import torch
import torch.nn as nn
import vllm
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.output_templates import OmniOutput
from .step_audio_tokenizer import StepAudioTokenizer

logger = logging.getLogger(__name__)


@dataclass
class Step1Config:
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_attention_heads: int = 48
    num_attention_groups: int = 4
    num_hidden_layers: int = 32
    vocab_size: int = 74752
    rms_norm_eps: int = 1e-5


class Step1CausalLMConfig(LlamaConfig):
    model_type = "step1"

    def __init__(
        self,
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=48,
        num_attention_groups=4,
        num_hidden_layers=32,
        vocab_size=74752,
        rms_norm_eps=1e-5,
        torch_dtype="bfloat16",
        max_seq_len=999999,
        **kwargs,
    ):
        kwargs.update(
            {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "vocab_size": vocab_size,
                "rms_norm_eps": rms_norm_eps,
                "torch_dtype": torch_dtype,
                "max_position_embeddings": max_seq_len,
            }
        )

        super().__init__(**kwargs)

        self.num_attention_groups = num_attention_groups
        self.max_seq_len = max_seq_len


def build_alibi_cache(n_heads):
    # get slopes
    n = 2 ** math.floor(math.log2(n_heads))  # nearest 2**n to n_heads
    m0 = 2.0 ** (-8.0 / n)
    # 2^(-8/n), 2^(-8*2/n), 2^(-8*3/n), ...
    slopes = torch.pow(m0, torch.arange(1, n + 1))
    if n < n_heads:
        m1 = 2.0 ** (-4.0 / n)
        # 2^(-8/(2n)), 2^(-8*3/(2n)), 2^(-8*5/(2n)), ...
        mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        slopes = torch.cat([slopes, mm])
    return slopes.tolist()


# Group Query Attention
class Step1Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups, prefix="", dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.alibi_slope = build_alibi_cache(n_heads=self.num_heads)
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_groups,
            params_dtype=dtype,
            prefix=f"{prefix}.qkv_proj",
            bias=False,
        )

        self.o_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            params_dtype=dtype,
            prefix=f"{prefix}.o_proj",
            bias=False,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_groups,
            alibi_slopes=self.alibi_slope,
            use_alibi_sqrt=True,
            scale=self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
            attn_backend=vllm.v1.attention.backends.triton_attn.TritonAttentionBackend,
        )

    def forward(self, x: torch.Tensor):
        # logger.info(f"Step1Attention input shape: {x.shape}")
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, seq_len, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, -1, self.head_dim)
        # QKVParallelLinear output: [q_heads_local, k_heads_local, v_heads_local]
        num_local_heads = self.qkv_proj.num_heads
        num_local_kv_heads = self.qkv_proj.num_kv_heads
        split_sizes = [num_local_heads, num_local_kv_heads, num_local_kv_heads]
        q, k, v = qkv.split(split_sizes, dim=2)

        # q = q.view(batch_size, seq_len, self.num_heads * self.head_dim)
        # k = k.view(batch_size, seq_len, self.num_groups * self.head_dim)
        # v = v.view(batch_size, seq_len, self.num_groups * self.head_dim)
        q = q.view(batch_size*seq_len, self.num_heads*self.head_dim)
        k = k.view(batch_size*seq_len, self.num_groups*self.head_dim)
        v = v.view(batch_size*seq_len, self.num_groups*self.head_dim)
        # logger.info(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
        )
        # logger.info(f"attn_output shape: {attn_output.shape}")
        # attn_output = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output, _ = self.o_proj(attn_output)

        return attn_output


class Step1MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, prefix="", dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        down, _ = self.down_proj(self.act_fn(gate) * up)
        return down


class Step1Layer(nn.Module):
    def __init__(self, config: Step1Config, dtype=torch.float32, prefix=""):
        super().__init__()
        self.self_attn = Step1Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_groups=config.num_attention_groups,
            dtype=dtype,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Step1MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=dtype,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(self, x):
        def f(x):
            x = self.input_layernorm(x)
            x = self.self_attn(x)
            return x

        x = x + f(x)

        def f(x):
            x = self.post_attention_layernorm(x)
            x = self.mlp(x)
            return x

        x = x + f(x)

        return x


class Step1ForCausalLM(nn.Module):
    def __init__(self, config: Step1CausalLMConfig, dtype=torch.float32) -> None:
        super().__init__()
        if isinstance(config.torch_dtype, str):
            dtype = getattr(torch, config.torch_dtype)
        else:
            dtype = config.torch_dtype or torch.float32
        step1_config = Step1Config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_attention_groups=config.num_attention_groups,
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.config = step1_config
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, dtype=dtype)
        self.layers = nn.Sequential(
            *[Step1Layer(self.config, dtype=dtype, prefix=f"layers.{i}") for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=dtype)
        self.lm_head = ReplicatedLinear(self.config.hidden_size, self.config.vocab_size, bias=False, params_dtype=dtype)

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.layers(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_hidden_states(input_ids)


class StepAudioAR(nn.Module):
    DEBUG_MARKER = "STEP_AUDIO_AR_RUNNER_MODEL"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config
        self.model_path = vllm_config.model_config.model
        print(vllm_config)
        extra = getattr(vllm_config.model_config, "additional_kwargs", None) or {}
        self.tokenizer_path = extra.get("audio_tokenizer")
        self.config = hf_config
        self.model = Step1ForCausalLM(hf_config)
        self.logits_processor = LogitsProcessor(hf_config.vocab_size)
        self.have_multimodal_outputs = False
        self.has_preprocess = True
        self.has_postprocess = False
        self.tokenizer = StepAudioTokenizer(
                tokenizer_path=self.tokenizer_path, 
                config_path=self.model_path,
            )

    def _ensure_tokenizer_loaded(self):
        assert self.tokenizer.loaded == True, logger.info("Tokenizer not loaded")
        return self.tokenizer

    def preprocess(self, info_dict):
        pass

    def _encode_ref_audio_to_code(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        tokenizer = self._ensure_audio_tokenizer_loaded()
        codec_token = tokenizer._audio_tokenize(wav, sr=int(sr), return_dict=True)
        ref_code = codec_token
        # ref_code = getattr(codec_token, "audio_codes", None)
        if isinstance(ref_code, list):
            ref_code = ref_code[0] if ref_code else None
        if isinstance(ref_code, torch.Tensor):
            # 12Hz: likely [T, Q] or [B, T, Q]
            if ref_code.ndim == 3:
                ref_code = ref_code[0]
            return ref_code.to(device=next(self.parameters()).device, dtype=torch.long)
        raise ValueError("SpeechTokenizer.encode did not return audio_codes tensor")

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        # logger.info(f"input_ids:{input_ids}")
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            hidden_states = self.model.layers(hidden_states)
            hidden_states = self.model.norm(hidden_states)
        else:
            hidden_states = self.model.forward_hidden_states(input_ids)

        num_tokens = int(input_ids.numel())

        if hidden_states.dim() == 3:
            b, s, h = hidden_states.shape
            assert b * s == num_tokens, (
                f"Hidden states token count mismatch: b*s={b*s}, num_tokens={num_tokens}, shape={hidden_states.shape}"
            )
            hidden_states = hidden_states.reshape(b * s, h)

        assert hidden_states.dim() == 2, f"Expected 2D hidden_states, got {hidden_states.shape}"
        assert hidden_states.shape[0] == num_tokens, (
            f"Hidden states first dim mismatch: {hidden_states.shape[0]} vs num_tokens={num_tokens}"
        )

        # logger.info(
        #     f"StepAudioAR.forward input_ids_shape={tuple(input_ids.shape)} "
        #     f"hidden_states_shape={tuple(hidden_states.shape)}"
        # )
        return hidden_states
    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.logits_processor(self.model.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
        ]
        for name, loaded_weight in weights:
            if name == "lm_head.weight":
                mapped_name = "model.lm_head.weight"
                if mapped_name in params_dict:
                    param = params_dict[mapped_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    try:
                        weight_loader(param, loaded_weight)
                    except AssertionError as err:
                        raise AssertionError(f"Failed to load weight {name!r} as {name!r}") from err
                    loaded_params.add(mapped_name)
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                    # print(f"loaded weight from {name} to {mapped_name}")
                except AssertionError as err:
                    raise AssertionError(f"Failed to load weight {name!r} as {name!r}") from err
                loaded_params.add(name)
            else:
                Found = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    stacked_name = name.replace(weight_name, param_name)
                    if stacked_name not in params_dict:
                        continue
                    param = params_dict[stacked_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    # print(f"loaded weight from {stacked_name} to {name}")
                    loaded_params.add(name)
                    Found = True
                    break
                if not Found:
                    logger.debug(f"Skipping weight {name} -> {name} - not found in model")

        return loaded_params
