import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.data_entry_keys import OmniPayload
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
        q = q.view(batch_size * seq_len, self.num_heads * self.head_dim)
        k = k.view(batch_size * seq_len, self.num_groups * self.head_dim)
        v = v.view(batch_size * seq_len, self.num_groups * self.head_dim)
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
        extra = getattr(vllm_config.model_config, "additional_kwargs", None) or {}
        self.tokenizer_path = extra.get("audio_tokenizer")
        self.config = hf_config
        self.model = Step1ForCausalLM(hf_config)
        self.logits_processor = LogitsProcessor(hf_config.vocab_size)
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.tokenizer = None

    def embed_multimodal(self, **kwargs) -> torch.Tensor:
        return self._encode_ref_audio_to_code(**kwargs)

    def _ensure_audio_tokenizer_loaded(self):
        if self.tokenizer is not None:
            return

        self.tokenizer = StepAudioTokenizer(
            tokenizer_path=self.tokenizer_path,
            config_path=self.model_path,
        )

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        *,
        task_type: str,
        tokenize_prompt: Callable[[str], list[int]],
        estimate_ref_code_len: Callable[[object], int | None] | None = None,
    ) -> int:
        """Compute Stage-0 placeholder prompt length (length-only mirror of `_build_prompt_embeds()`).
        It must match the model-side `inputs_embeds` length to avoid extra padding and quality drop."""

        def _first(x: object, default: object) -> object:
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        info: dict[str, Any] = additional_information or {}
        text = _first(info.get("text"), "")
        # Official defaults: CustomVoice/VoiceDesign -> non_streaming_mode=True; Base -> False.
        non_streaming_mode = task_type in ("edit")

        if not isinstance(text, str):
            text = ""
        # ---- text conditioning portion (matches _build_prompt_embeds) ----
        assistant_text = StepAudioAR._build_assistant_text(text)
        assistant_len = len(tokenize_prompt(assistant_text))
        if assistant_len < 8:
            raise ValueError(f"Unexpected assistant prompt length: {assistant_len}")

        if task_type == "clone":
            voice_clone_prompt = _first(info.get("voice_clone_prompt"), None)

            ref_code = None
            if isinstance(voice_clone_prompt, dict):
                ref_code = _first(voice_clone_prompt.get("ref_code"), None)

            ref_code_len: int | None = None
            if isinstance(ref_code, list):
                if ref_code and isinstance(ref_code[0], list):
                    ref_code_len = len(ref_code)
                elif ref_code:
                    ref_code_len = len(ref_code)
            elif hasattr(ref_code, "shape"):
                try:
                    shape = getattr(ref_code, "shape")
                    if shape and len(shape) >= 1:
                        ref_code_len = int(shape[0])
                except Exception:
                    ref_code_len = None

            if ref_code_len is None and estimate_ref_code_len is not None:
                ref_code_len = estimate_ref_code_len(info.get("ref_audio"))
            if ref_code_len is None:
                raise ValueError(
                    "Base in-context voice cloning requires either `voice_clone_prompt.ref_code` "
                    "or a readable `ref_audio` that can be mapped to a codec frame length."
                )

            codec_lens = 1 + int(ref_code_len)  # codec_bos + ref_code
            if non_streaming_mode:
                prompt_len = 0
                # _generate_icl_prompt(non_streaming_mode=True):
                # text_embed = ref_ids + text_ids + eos.
                ref_ids = _first(info.get("ref_ids"), None)
                if isinstance(voice_clone_prompt, dict) and ref_ids is None:
                    ref_ids = _first(voice_clone_prompt.get("ref_ids") or voice_clone_prompt.get("ref_id"), None)

                if ref_ids is None:
                    ref_text = _first(info.get("ref_text"), "")
                    if not isinstance(ref_text, str) or not ref_text.strip():
                        raise ValueError("Base in-context non-streaming requires `ref_text` or tokenized `ref_ids`.")
                    ref_text_ids = tokenize_prompt(StepAudioAR._build_ref_text(ref_text))
                    ref_ids_len = len(ref_text_ids)
                elif hasattr(ref_ids, "shape"):
                    shape = getattr(ref_ids, "shape", None)
                    ref_ids_len = int(shape[-1]) if shape else 0
                elif isinstance(ref_ids, list):
                    ref_ids_len = len(ref_ids)
                else:
                    ref_ids_len = 0

                # model uses ref_ids[:, 3:-2] (strip 5 tokens) and text_id=input_ids[:, 3:-5] (strip 8).
                ref_id_len = max(0, int(ref_ids_len) - 5)
                text_id_len = max(0, int(assistant_len) - 8)
                text_embed_len = ref_id_len + text_id_len + 1  # + eos
                prompt_len += text_embed_len + codec_lens
            else:
                # _generate_icl_prompt(non_streaming_mode=False): aligned to codec_lens.
                prompt_len += codec_lens

        return max(2, int(prompt_len))

    def _build_prompt_embeds(
        self,
        *,
        task_type: str,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None, torch.Tensor | None]:
        logger.info(f"Building prompt embeds for task_type: {task_type}, info_dict: {info_dict}")
        ref_audio = info_dict.get("ref_audio")
        ref_text = info_dict.get("ref_text")
        text = info_dict.get("text")
        sr = info_dict.get("sr") or 16000
        prompt_token, codec_token = self.tokenizer.encode(task_type, audio=ref_audio, prompt=(ref_text, text), sr=sr)
        logger.info(f"prompt_token: {prompt_token}")
        input_ids = torch.tensor(prompt_token.input_ids)
        input_ids = input_ids.to(next(self.model.parameters()).device)
        logger.info(f"input_ids shape: {input_ids.shape}, codec_token shape: {codec_token.shape}")
        input_ids = self.embed_input_ids(input_ids)
        tts_pad_id = self.tokenizer.text_tokenizer.pad_token_id
        tts_pad_embed = self.embed_input_ids(torch.tensor([tts_pad_id]).to(input_ids.device))
        return input_ids, codec_token.shape[1], codec_token, tts_pad_embed

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")
        audio_codes_list: list[torch.Tensor] = []
        ref_code_len_list: list[torch.Tensor] = []
        ref_code_tensor: torch.Tensor | None = None
        codec_streaming_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes = info.get("codes", {})
            meta = info.get("meta", {})
            ac = codes.get("audio")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)
                cs = meta.get("codec_streaming")
                if isinstance(cs, bool):
                    codec_streaming_list.append(
                        torch.full((int(ac.shape[0]),), int(cs), dtype=torch.int8, device=ac.device)
                    )
            ref_code = codes.get("ref")
            if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
                ref_code_tensor = ref_code
            ref_len = meta.get("ref_code_len")
            if ref_len is None:
                continue
            if isinstance(ref_len, torch.Tensor):
                if ref_len.numel() == 0:
                    raise ValueError("ref_code_len is an empty tensor")
                ref_len_val = int(ref_len.reshape(-1)[-1].item())
            elif isinstance(ref_len, list):
                if len(ref_len) != 1:
                    raise ValueError(f"ref_code_len must be scalar or 1-element list, got len={len(ref_len)}")
                ref_len_val = int(ref_len[0])
            else:
                ref_len_val = int(ref_len)
            if isinstance(ac, torch.Tensor):
                # Emit ref_code_len per-token span for runner slicing (consumer takes the last value).
                ref_code_len_list.append(
                    torch.full((int(ac.shape[0]),), ref_len_val, dtype=torch.int32, device=ac.device)
                )

        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(audio_codes_list, dim=0)
        span_len = int(audio_codes.shape[0])
        # logger.info(f"span_len: {span_len}, hidden.shape: {hidden.shape}")
        # hidden = hidden[:span_len]
        mm: OmniPayload = {"codes": {"audio": audio_codes}}
        if ref_code_len_list:
            mm.setdefault("meta", {})["ref_code_len"] = torch.cat(ref_code_len_list, dim=0)[:span_len]
        if ref_code_tensor is not None:
            mm.setdefault("codes", {})["ref"] = [ref_code_tensor]
        if codec_streaming_list:
            mm.setdefault("meta", {})["codec_streaming"] = torch.cat(codec_streaming_list, dim=0)[:span_len]
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._ensure_audio_tokenizer_loaded()
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        payload: OmniPayload = info_dict
        embed = payload.get("embed", {})
        meta = payload.get("meta", {})

        text_list = text_list = info_dict.get("text")

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        text_list = info_dict.get("text")
        if not isinstance(text_list, list) or not text_list or not text_list[0]:
            raise ValueError("Missing additional_information.text for Qwen3-TTS AR talker.")

        task_type = info_dict.get("task_type") or ["clone"]
        codec_streaming = task_type == "clone"

        prompt_embeds_cpu = embed.get("prefill")
        tts_pad_embed_cpu = embed.get("tts_pad")
        tts_pad_embed = None
        if isinstance(tts_pad_embed_cpu, torch.Tensor) and tts_pad_embed_cpu.numel() > 0:
            tts_pad_embed = tts_pad_embed_cpu.to(device=input_ids.device, dtype=torch.bfloat16).reshape(1, -1)

        # Subsequent prefill rounds (multi-chunk): prompt_embeds_cpu is a Tensor stored by the first round.
        is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2
        if is_first_prefill:
            full_prompt_embeds, ref_code_len, ref_code, tts_pad_embed = self._build_prompt_embeds(
                task_type=task_type, info_dict=info_dict
            )
            # Store full prompt embeddings on CPU (large, prefill-only).
            # tailing_text_hidden and tts_pad_embed stay on GPU (gpu_resident_buffer_keys).
            prompt_embeds_cpu = full_prompt_embeds.detach().to("cpu").contiguous()
            info_update: OmniPayload = {
                "embed": {
                    "prefill": prompt_embeds_cpu,
                    "tts_pad": tts_pad_embed.detach(),
                },
                "meta": {
                    "talker_prefill_offset": 0,
                    "talker_text_offset": 0,
                    "codec_streaming": codec_streaming,
                },
            }
            if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
                info_update.setdefault("codes", {})["ref"] = ref_code.detach().to("cpu").contiguous()
            if ref_code_len is not None:
                info_update["meta"]["ref_code_len"] = int(ref_code_len)
            # Always return a span_len slice; if the scheduled placeholder is longer, pad with tts_pad_embed.
            # This preserves placeholder/embedding alignment.
            offset = 0
            s = 0
            e = span_len
            take = prompt_embeds_cpu[s:e]
            prompt_embeds = take.to(device=input_ids.device, dtype=torch.bfloat16)
            info_update["meta"]["talker_prefill_offset"] = int(offset + span_len)
        else:
            offset = int(meta.get("talker_prefill_offset", 0) or 0)
            if offset < 0:
                offset = 0
            s = max(0, min(offset, int(prompt_embeds_cpu.shape[0])))
            e = max(0, min(offset + span_len, int(prompt_embeds_cpu.shape[0])))
            take = prompt_embeds_cpu[s:e]
            if int(take.shape[0]) < span_len:
                pad_n = int(span_len - int(take.shape[0]))
                pad_rows = tts_pad_embed.reshape(1, -1).to("cpu").expand(pad_n, -1)
                take = torch.cat([take, pad_rows], dim=0)
            # Subsequent prefill chunk: slice from stored embeddings at running offset.
            prompt_embeds = take.to(device=input_ids.device, dtype=torch.bfloat16)
            info_update = {
                "meta": {
                    "talker_prefill_offset": int(offset + span_len),
                    "codec_streaming": codec_streaming,
                }
            }

        # When inputs_embeds is set, token ids are ignored by the model but must stay in-vocab for vLLM bookkeeping.
        input_ids_out = input_ids.clone()
        input_ids_out[:] = 0

        zeros = torch.zeros(
            (prompt_embeds.shape[0], 1),
            device=input_ids.device,
            dtype=torch.long,
        )
        info_update.setdefault("codes", {})["audio"] = zeros
        return input_ids_out, prompt_embeds, info_update

    def _encode_ref_audio_to_code(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        try:
            self._ensure_audio_tokenizer_loaded()
            audio_prompt, codec_token = self.tokenizer._audio_tokenize(wav, sr=int(sr), return_dict=True)
            return audio_prompt, codec_token
        except Exception as e:
            logger.error("Failed to tokenize audio prompt", exc_info=e)

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
                f"Hidden states token count mismatch: b*s={b * s}, num_tokens={num_tokens}, shape={hidden_states.shape}"
            )
            hidden_states = hidden_states.reshape(b * s, h)

        assert hidden_states.dim() == 2, f"Expected 2D hidden_states, got {hidden_states.shape}"
        assert hidden_states.shape[0] == num_tokens, (
            f"Hidden states first dim mismatch: {hidden_states.shape[0]} vs num_tokens={num_tokens}"
        )

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
