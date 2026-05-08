import math
from dataclasses import dataclass
from einops import rearrange
import torch.nn.functional
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.nn import init
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm_omni.diffusion.attention.layer import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

@dataclass
class Step1Config:
    hidden_size: int = 12288
    intermediate_size: int = 31232
    num_attention_heads: int = 96
    num_attention_groups: int = 8
    num_hidden_layers: int = 88
    vocab_size: int = 65536
    rms_norm_eps: int = 1e-5


@dataclass
class ModelDimensions:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 32
    n_codebook_size: int = 4096
    LLM_dim: int = 3072
    kernel_size: int = 3
    # stride: int = 4
    adapter_stride: int = 2


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.
    
    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    1 for non-padded part and 0 for padded part.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask

def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B, ?).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, ?).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        >>> new_masks = s3tokenizer.mask_to_bias(masks, torch.float32)
        new_masks = [[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],                                                                                                                                                                                  
                    [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10],                                                                                                                                                                                  
                    [-0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10, -1.0000e+10]]
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask

def build_alibi_cache(block_size, n_heads, dtype, device):
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
    slopes = slopes.to(device)

    tril = torch.tril(torch.ones(1, 1, block_size, block_size, device=device))

    bias_rows = torch.arange(block_size, device=device).view(1, -1)
    bias_cols = torch.arange(block_size, device=device).view(-1, 1)
    bias = -torch.sqrt(bias_cols - bias_rows)
    bias = bias.view(1, block_size, block_size) * slopes.view(-1, 1, 1)
    bias = bias.masked_fill(tril == 0, float('-inf'))

    return bias.type(dtype)

### Replace with vllm's implementation for better performance and stability
class Step1RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, dtype=torch.float32, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(x.dtype)
        x = x * self.weight
        return x

## Replace with vllm's implementation for better performance and stability
class Step1Attention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups, dtype=torch.float32):
        super().__init__()

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.k_proj = torch.nn.Linear(hidden_size, num_groups * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = torch.nn.Linear(hidden_size, num_groups * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor):
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (g d) -> b s g d", g=self.num_groups)
        v = rearrange(v, "b s (g d) -> b s g d", g=self.num_groups)

        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=-2)
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=-2)

        mask = build_alibi_cache(q.size(1), self.num_heads, dtype=q.dtype, device=q.device)
        o: torch.Tensor = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask
        )
        o = o.transpose(1, 2).flatten(-2, -1)

        o = self.o_proj(o)
        return o


class Step1MLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype=torch.float32):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        return x


class Step1Layer(torch.nn.Module):
    def __init__(self, config: Step1Config, dtype=torch.float32):
        super().__init__()
        self.self_attn = Step1Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_groups=config.num_attention_groups,
            dtype=dtype
        )
        self.mlp = Step1MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=dtype
        )
        self.input_layernorm = Step1RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = Step1RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

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


class Step1Model(torch.nn.Module):
    def __init__(self, config: Step1Config, dtype=torch.float32) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = torch.nn.Sequential(*[Step1Layer(config, dtype=dtype) for _ in range(config.num_hidden_layers)])
        self.norm = Step1RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.gradient_checkpointing = False

    def forward(self, hidden_states):
        x = hidden_states
        if self.gradient_checkpointing and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x)
        else:
            x = self.layers(x)
        x = self.norm(x)
        return x


class Step1AudioModel(torch.nn.Module):
    def __init__(self, config: Step1Config, encoder_config: ModelDimensions, dtype=torch.float32):
        super().__init__()
        self.model = Step1Model(config, dtype=dtype)
        self.bf16 = dtype==torch.bfloat16
        self.fp16 = dtype==torch.float16
        self.dims = encoder_config
        self.encoder = AudioEncoder(self.dims.n_mels, self.dims.n_audio_ctx, self.dims.n_audio_state, self.dims.n_audio_head, self.dims.n_audio_layer)
        self.adapter = Adaptor(self.dims.n_audio_state, self.dims.LLM_dim, self.dims.kernel_size, self.dims.adapter_stride)
        if self.bf16:
            self.encoder=self.encoder.bfloat16()
            self.adapter=self.adapter.bfloat16()
        if self.fp16:
            self.encoder=self.encoder.half()
            self.adapter=self.adapter.half()
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

    def forward(self, input_ids=None, wavs=None, **kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        # encode wavs
        if self.bf16:
            wavs = wavs.bfloat16()
        if self.fp16:
            wavs = wavs.half()

        wav_lens = kwargs['wav_lens']
        out, _ = self.encoder(wavs, wav_lens)
        assert out.size(2) == 1280
        # out = self.encoderout_ln(out)
        out = self.adapter(out)

        # replace audio tokens in hidden_states
        insert_location = torch.nonzero(input_ids == 31)
        insert_location[:,1] += 1
        feat_lens = out.shape[1]
        for idx in range(len(insert_location)):
            i,s = insert_location[idx]
            hidden_states[i][s : s+feat_lens] = out[idx]
        # hidden_states = patch_scatter(hidden_states, out, insert_location.contiguous())
        x = self.model(hidden_states)
        x = self.lm_head(x)
        return x



class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Embedding(n_ctx, n_state)
        self.positional_embedding.requires_grad_(False)
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = LayerNorm(n_state)
        self.gradient_checkpointing = False

    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        T = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask_to_bias(mask[:, :, (T + 1) % 2::2], x.dtype)  # (B, 1, T // 2)
        x = (x + self.positional_embedding.weight[:x.shape[1], :]).to(x.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask.unsqueeze(1))
            else:
                x = block(x, mask.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x.contiguous())
        return x, x_len

class Adaptor(nn.Module):
    def __init__(
        self, 
        n_state: int = 1280, 
        n_hidden: int = 3072,
        kernel_size: int = 7,
        stride: int = 4
    ):
        super().__init__()
        self.stride = stride
        if self.stride != -1:
            # print("self.stride: {}".format(self.stride))
            self.conv = Conv1d(n_state, n_state, kernel_size, stride, padding=1)
        self.linear1 = nn.Linear(n_state, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, n_hidden)
        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        T = x.size(-1)
        if self.stride != -1:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.conv, x.permute(0, 2, 1))
                x = x.permute(0, 2, 1)
            else:
                x = x.permute(0, 2, 1)
                x = F.gelu(self.conv(x))
                x = x.permute(0, 2, 1)
        if self.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.linear1, x)
            x = torch.utils.checkpoint.checkpoint(self.relu, x)
            x = torch.utils.checkpoint.checkpoint(self.linear2, x)
        else:
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
        return x


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).type(x.dtype)

# class LayerNorm(torch.nn.LayerNorm):
#     def forward(self, x):
#         # Force the weight and bias to the same dtype as the input
#         if self.weight is not None:
#             self.weight = self.weight.to(dtype=x.dtype)
#         if self.bias is not None:
#             self.bias = self.bias.to(dtype=x.dtype)
#         return super().forward(x).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        _, T, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x.contiguous()), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x.contiguous()))
        return x