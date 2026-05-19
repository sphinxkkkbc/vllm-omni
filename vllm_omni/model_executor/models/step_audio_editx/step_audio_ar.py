import math
from dataclasses import dataclass
import logging
from dataclasses import dataclass
from typing import Iterable, Optional
import torch
import torch.nn as nn
from transformers import LlamaConfig
from vllm import SamplingParams
from transformers.modeling_outputs import CausalLMOutputWithPast
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from transformers import AutoTokenizer
from vllm_omni.model_executor.models.step_audio_editx.utils import AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL, AUDIO_EDIT_SYSTEM_PROMPT

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
        **kwargs
    ):

        kwargs.update({
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_attention_heads': num_attention_heads,
            'num_hidden_layers': num_hidden_layers,
            'vocab_size': vocab_size,
            'rms_norm_eps': rms_norm_eps,
            'torch_dtype': torch_dtype,
            'max_position_embeddings': max_seq_len,
        })
        
        super().__init__(**kwargs)
        
        self.num_attention_groups = num_attention_groups
        self.max_seq_len = max_seq_len


def build_alibi_cache(n_heads):
    # get slopes
    n = 2 ** math.floor(math.log2(n_heads))  # nearest 2**n to n_heads
    m0 = 2.0 ** (-8.0 / n)
    # 2^(-8/n), 2^(-8*2/n), 2^(-8*3/n), ...
    slopes = torch.pow(m0, torch.arange(1, n+1))
    if n < n_heads:
        m1 = 2.0 ** (-4.0 / n)
        # 2^(-8/(2n)), 2^(-8*3/(2n)), 2^(-8*5/(2n)), ...
        mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        slopes = torch.cat([slopes, mm])
    return slopes.tolist()



#Group Query Attention
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
            disable_tp=True,
            prefix=f"{prefix}.qkv_proj",
            bias=False
        )

        self.o_proj = RowParallelLinear(
            input_size = self.hidden_size,
            output_size = self.hidden_size,
            disable_tp=True,
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
            scale=self.head_dim ** -0.5,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, -1, self.head_dim)
        # QKVParallelLinear output: [q_heads_local, k_heads_local, v_heads_local]
        num_local_heads = self.qkv_proj.num_heads
        num_local_kv_heads = self.qkv_proj.num_kv_heads
        split_sizes = [num_local_heads, num_local_kv_heads, num_local_kv_heads]
        q, k, v = qkv.split(split_sizes, dim=2)

        q = q.view(batch_size, seq_len, self.num_heads*self.head_dim)
        k = k.view(batch_size, seq_len, self.num_groups*self.head_dim)
        v = v.view(batch_size, seq_len, self.num_groups*self.head_dim)

        attn_output = self.attn(
            query=q,
            key=k, 
            value=v,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
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
            disable_tp=True,
            params_dtype=dtype,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            disable_tp=True,
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
    def __init__(self, config:Step1CausalLMConfig, dtype=torch.float32) -> None:
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
        self.layers = nn.Sequential(*[Step1Layer(self.config, dtype=dtype, prefix=f"layers.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(hidden_size=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=dtype)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            dtype=dtype
        )

    def forward(
        self,
        input_ids=None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.layers(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        name_mapping = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
        ]
        for name, loaded_weight in weights:
            if name.startswith("model.layers."):
                name = name[len("model."):]
            mapped_name = name_mapping.get(name, name)
            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                    # print(f"loaded weight from {name} to {mapped_name}")
                except AssertionError as err:
                    raise AssertionError(
                        f"Failed to load weight {name!r} as {mapped_name!r}"
                    ) from err
                loaded_params.add(mapped_name)
            else:
                Found=False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in mapped_name:
                        continue
                    stacked_name = mapped_name.replace(weight_name, param_name)
                    if stacked_name not in params_dict:
                        continue
                    param = params_dict[stacked_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    # print(f"loaded weight from {stacked_name} to {name}")
                    loaded_params.add(stacked_name)
                    Found = True
                    break  
                if not Found:  
                    logger.debug(f"Skipping weight {name} -> {mapped_name} - not found in model")
        expected = set(params_dict.keys())
        missing = expected - loaded_params

        print("missing:")
        for name in sorted(missing):
            print(name)

        return loaded_params
    
class StepAudioAR:
    def __init__(self, config, vllm_config, **kwargs) -> None:
        self.text_tokenizer = AutoTokenizer.from_pretrained(**kwargs)
        self.llm = Step1ForCausalLM(Step1CausalLMConfig())
        self.edit_clone_sys_prompt_tpl = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL
        self.edit_sys_prompt = AUDIO_EDIT_SYSTEM_PROMPT


    def forward(self, prompt, audio_tokens, task_type):
        if task_type == "edit":  
            prompt_text, edit_type, edit_info, target_text = prompt
            instruct_prefix = self._build_audio_edit_instruction(prompt_text, edit_type, edit_info, target_text)

            token_ids = self._encode_edit_prompt(
                self.edit_sys_prompt,
                instruct_prefix, 
                audio_tokens
            )

        if task_type == "clone":
            prompt_text, target_text = prompt
            prompt_speaker = "debug"
            token_ids = self._encode_clone_prompt(
                target_text,
                prompt_text,
                prompt_speaker,
                audio_tokens,
            )
        output = self._generate(token_ids, max_tokens=8192 - len(token_ids))
        # return output
        return OmniOutput(
            text_hidden_states=output,
        )
    
    def _generate(self, token_ids: list[int], max_tokens: int = 4096, temperature: float = 0.7) -> torch.Tensor:
        audio_in = sum(1 for t in token_ids if 65536 <= t < 67584)
        text_in = sum(1 for t in token_ids if t < 65536)
        other_in = sum(1 for t in token_ids if t >= 67584)
        logger.info(f"INPUT tokens: total={len(token_ids)}, audio(65536-67583)={audio_in}, text(<65536)={text_in}, other(>=67584)={other_in}")
        if token_ids:
            logger.info(f"INPUT range: min={min(token_ids)}, max={max(token_ids)}")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=False,
        )
        
        # Use prompt_token_ids directly instead of decoding to text
        # This preserves audio tokens (65536+) which would be corrupted by decode
        prompt = {"prompt_token_ids": token_ids}
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)

        # Extract output token IDs (vLLM only returns generated tokens, not input)
        output_token_ids = list(outputs[0].outputs[0].token_ids)
        
        # Debug: analyze token distribution
        if output_token_ids:
            min_tok = min(output_token_ids)
            max_tok = max(output_token_ids)
            audio_count = sum(1 for t in output_token_ids if 65536 <= t < 67584)
            text_count = sum(1 for t in output_token_ids if t < 65536)
            other_count = sum(1 for t in output_token_ids if t >= 67584)
            logger.info(f"Generated {len(output_token_ids)} tokens: min={min_tok}, max={max_tok}, "
                       f"audio(65536-67583)={audio_count}, text(<65536)={text_count}, other(>=67584)={other_count}")
        
        # Remove eos token if present
        if len(output_token_ids) > 0 and output_token_ids[-1] == 3: # <|EOT|>
            output_token_ids = output_token_ids[:-1]
        
        output_ids = torch.tensor(output_token_ids, dtype=torch.long)

        return output_ids

    def _encode_edit_prompt(
        self, sys_prompt: str, instruct_prefix: str, audio_token_str: str
    ) -> list[int]:
        """Encode audio edit prompt to token sequence"""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{instruct_prefix}\n{audio_token_str}\n"}
        ]

        return self.text_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


    def _encode_clone_prompt(
        self, text: str, prompt_text: str, prompt_speaker: str, prompt_wav_tokens: str
    ):
        
        sys_prompt = self.edit_clone_sys_prompt_tpl.format(
            speaker=prompt_speaker,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{text}"}
        ]

        return self.text_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def _build_audio_edit_instruction(
        self,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None
    ) -> str:
        """Build audio editing instruction based on request"""
        audio_text = audio_text.strip() if audio_text else ""
        if edit_type in {"emotion", "speed"}:
            if edit_info == "remove":
                instruct_prefix = f"Remove any emotion in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {edit_info}. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "style":
            if edit_info == "remove":
                instruct_prefix = f"Remove any speaking styles in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {edit_info} style. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "denoise":
            instruct_prefix = f"Remove any noise from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all noise from the audio.\n"
        elif edit_type == "vad":
            instruct_prefix = f"Remove any silent portions from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all silence from the audio.\n"
        elif edit_type == "paralinguistic":
            instruct_prefix = f"Add some non-verbal sounds to make the audio more natural, the new text is : {text}\n  The text corresponding to the audio is: {audio_text}\n"
        else:
            logger.error(f"Unsupported audio editing type: {edit_type}")
        return instruct_prefix
