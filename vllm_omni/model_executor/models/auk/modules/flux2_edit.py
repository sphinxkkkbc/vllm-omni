# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def is_package_available(package_name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(package_name) is not None
    except Exception:
        return False


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B 1 N]
        x = x.permute(0, 2, 1)  # [B D N]

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # [B N D]

        return x


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward (SwiGLU)


class SwiGLU(nn.Module):
    """
    Flux 2 uses a SwiGLU-style activation in the transformer feedforward sub-blocks, but with the linear projection
    layer fused into the first linear layer of the FF sub-block. Thus, this module has no trainable parameters.
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x = self.gate_fn(x1) * x2
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # SwiGLU will reduce the dimension by half
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act_fn = SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: int | None = None,  # if not None -> joint attention
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim

        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim)

        self.q_norm = torch.nn.RMSNorm(dim_head, elementwise_affine=True)
        self.k_norm = torch.nn.RMSNorm(dim_head, elementwise_affine=True)

        if self.context_dim is not None:
            self.to_qkv_c = nn.Linear(context_dim, 3 * self.inner_dim)
            self.c_q_norm = torch.nn.RMSNorm(dim_head, elementwise_affine=True)
            self.c_k_norm = torch.nn.RMSNorm(dim_head, elementwise_affine=True)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_dim is not None:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x
        c: float["b n d"] = None,  # context c
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
        c_mask: bool["b nt"] | None = None,  # text mask
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor

if is_package_available("flash_attn"):
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input


class AttnProcessor:
    def __init__(
        self,
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ):
        if attn_backend == "flash_attn":
            assert is_package_available("flash_attn"), "Please install flash-attn first."

        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections
        query, key, value = attn.to_qkv(x).chunk(3, dim=-1)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        query = attn.q_norm(query)
        key = attn.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if self.attn_backend == "torch":
            # mask. e.g. inference got a batch with different target durations, mask out the padding
            if self.attn_mask_enabled and mask is not None:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                total_seq_len = query.shape[1]
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(query, mask)
                key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(key, mask)
                value, _, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    q_cu_seqlens,
                    k_cu_seqlens,
                    q_max_seqlen_in_batch,
                    k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, total_seq_len)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(
        self,
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ):
        if attn_backend == "flash_attn":
            assert is_package_available("flash_attn"), "Please install flash-attn first."

        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        c: float["b nt d"] = None,  # context c, here text
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
        c_mask: bool["b nt"] | None = None,  # text mask
    ) -> torch.FloatTensor:
        residual = x
        audio_mask = mask

        batch_size = c.shape[0]

        # `sample` projections
        query, key, value = attn.to_qkv(x).chunk(3, dim=-1)

        # `context` projections
        c_query, c_key, c_value = attn.to_qkv_c(c).chunk(3, dim=-1)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        query = attn.q_norm(query)
        key = attn.k_norm(key)
        c_query = attn.c_q_norm(c_query)
        c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # build combined mask for joint attention: audio mask + text mask
        if self.attn_mask_enabled and mask is not None:
            if c_mask is not None:
                mask = torch.cat([mask, c_mask], dim=1)
            else:
                mask = F.pad(mask, (0, c.shape[1]), value=True)

        if self.attn_backend == "torch":
            # mask. e.g. inference got a batch with different target durations, mask out the padding
            if self.attn_mask_enabled and mask is not None:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                total_seq_len = query.shape[1]
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(query, mask)
                key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(key, mask)
                value, _, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    q_cu_seqlens,
                    k_cu_seqlens,
                    q_max_seqlen_in_batch,
                    k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, total_seq_len)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        c = attn.to_out_c(c)

        if audio_mask is not None:
            x = x.masked_fill(~audio_mask.unsqueeze(-1), 0.0)
        if c_mask is not None:
            c = c.masked_fill(~c_mask.unsqueeze(-1), 0.0)

        return x, c


# DiT Block


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        attn_backend="torch",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
    ):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    """

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        context_dim=None,
        attn_backend="torch",
        attn_mask_enabled=False,
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim

        self.attn_norm_c = AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
        )

        self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
        self.ff_c = SwiGLUFeedForward(dim=context_dim, mult=ff_mult)
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(
        self, x, c, t, mask=None, rope=None, c_rope=None, c_mask=None
    ):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask)

        # process attention output for context c
        c = c + c_gate_msa.unsqueeze(1) * c_attn_output
        norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        c_ff_output = self.ff_c(norm_c)
        c = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float[b]):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


"""
Flux2Edit backbone — Flux2Audio variant using pre-encoded LLM text embeddings.

Two-phase architecture:
  1. Double-stream MMDiTBlock (joint text-audio attention)
  2. Single-stream DiTBlock (concatenated text+audio)

ein notation:
b - batch
n - sequence
nt - text sequence
d - dimension
"""


class AudioPromptEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def _embed(self, x: float["b n d"], mask: bool["b n"] | None = None):
        x = self.linear(x)
        x = self.conv_pos_embed(x, mask=mask) + x
        return x

    def forward(
        self,
        x: float["b n d"],
        ref: float["b np d"] | None = None,
        drop_audio_cond: bool = False,
        mask: bool["b n"] | None = None,
        ref_mask: bool["b np"] | None = None,
    ):
        x_emb = self._embed(x, mask=mask)
        if ref is None:
            return x_emb
        if drop_audio_cond:
            ref = torch.zeros_like(ref)
        ref_emb = self._embed(ref, mask=ref_mask)
        return x_emb, ref_emb


@dataclass
class Flux2EditConfig:
    dim: int = 1024
    depth: int = 8
    heads: int = 16
    dim_head: int = 64
    dropout: float = 0.1
    ff_mult: float = 2.0
    text_hidden_dim: int = 2048
    checkpoint_activations: bool = False
    checkpoint_every_n_layers: int = 1  # 1=every layer, n>1=every n-th layer (only used when checkpoint_activations)
    attn_backend: str = "torch"  # torch | flash_attn
    attn_mask_enabled: bool = False
    num_layers: int = 8  # double-stream (MMDiT) block count
    num_single_layers: int = 24  # single-stream (DiT) block count

    def __post_init__(self):
        logger.info("Flux2Edit Config:")
        for f in fields(self):
            logger.info(f"  {f.name}: {getattr(self, f.name)}")

    @classmethod
    def from_dict(cls, config_dict: dict):
        valid_fields = {f.name for f in fields(cls)}
        valid_params = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**valid_params)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


class Flux2Edit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        latent_dim=100,
        text_hidden_dim=2048,
        checkpoint_activations=False,
        checkpoint_every_n_layers=1,
        attn_backend="torch",
        attn_mask_enabled=False,
        num_layers=8,
        num_single_layers=24,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.time_embed = TimestepEmbedding(dim)

        # text projection: Linear(text_hidden_dim, dim) -> RMSNorm(dim)
        self.txt_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)

        self.audio_embed = AudioPromptEmbedding(latent_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        # Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for i in range(num_layers)
            ]
        )

        # Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, latent_dim)

        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_every_n_layers = max(1, checkpoint_every_n_layers)

        # text cache (mirrors Flux2Audio interface)
        self.text_cond, self.text_uncond = None, None

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def project_text(self, text: float["b nt h"], drop_text: bool = False):
        """Project LLM hidden states to model dim: Linear -> RMSNorm."""
        c = self.txt_norm(self.txt_proj(text))
        if drop_text:
            c = torch.zeros_like(c)
        return c

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def _embed_audio(
        self,
        x: float["b n d"],
        ref: float["b np d"] | None,
        drop_audio_cond: bool,
        mask: bool["b n"] | None,
        ref_mask: bool["b np"] | None,
    ):
        if ref is not None and ref.shape[1] == 0:
            ref = None
        out = self.audio_embed(
            x,
            ref=ref,
            drop_audio_cond=drop_audio_cond,
            mask=mask,
            ref_mask=ref_mask,
        )
        if isinstance(out, tuple):
            x_emb, ref_emb = out
            prompt_len = ref_emb.shape[1]
            audio = torch.cat([ref_emb, x_emb], dim=1)  # TODO：ref_emb 和 noised latent拼接在一起
            if mask is not None or ref_mask is not None:
                B, N = x_emb.shape[:2]
                if mask is None:
                    mask = torch.ones(B, N, dtype=torch.bool, device=x_emb.device)
                if ref_mask is None:
                    ref_mask = torch.ones(B, prompt_len, dtype=torch.bool, device=ref_emb.device)
                audio_mask = torch.cat([ref_mask, mask], dim=1)
            else:
                audio_mask = None
            return audio, audio_mask, prompt_len
        # ref is None — allow target-only forward for symmetry
        return out, mask, 0

    def forward(
        self,
        x: float["b n d"],  # noised input audio
        text: float["b nt h"] = None,  # pre-encoded text embeddings from LLM
        time: float[b] | float[""] = None,  # time step
        mask: bool["b n"] | None = None,
        c_mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        ref: float["b np d"] | None = None,  # prompt audio latent
        ref_mask: bool["b np"] | None = None,
    ):
        # print(x.shape)
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)  # B * D

        # text mask: padding positions are all-zero in LLM output
        if c_mask is None:
            c_mask = text.abs().sum(-1) > 0  # [B, nt], True = valid

        if cfg_infer:
            # cond branch
            if cache and self.text_cond is not None:
                c_cond = self.text_cond
            else:
                c_cond = self.project_text(text, drop_text=False)
                if cache:
                    self.text_cond = c_cond
            x_cond, a_mask_cond, prompt_len = self._embed_audio(
                x, ref, drop_audio_cond=False, mask=mask, ref_mask=ref_mask
            )

            # uncond branch
            if cache and self.text_uncond is not None:
                c_uncond = self.text_uncond
            else:
                c_uncond = self.project_text(text, drop_text=True)
                if cache:
                    self.text_uncond = c_uncond
            x_uncond, a_mask_uncond, _ = self._embed_audio(x, ref, drop_audio_cond=True, mask=mask, ref_mask=ref_mask)

            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)

            if a_mask_cond is not None and a_mask_uncond is not None:
                audio_mask = torch.cat((a_mask_cond, a_mask_uncond), dim=0)
            else:
                audio_mask = None
            c_mask = torch.cat((c_mask, c_mask), dim=0)
        else:
            c = self.project_text(text, drop_text=drop_text)  # b, seq, d
            x, audio_mask, prompt_len = self._embed_audio(
                x, ref, drop_audio_cond=drop_audio_cond, mask=mask, ref_mask=ref_mask
            )

        seq_len = x.shape[1]  # reference audio prompt | noised audio latent
        text_len = c.shape[1]  # nt

        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        # Phase 1: Double-stream MMDiTBlock
        for i, block in enumerate(self.transformer_blocks):
            if self.checkpoint_activations and i % self.checkpoint_every_n_layers == 0:
                c, x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    c,
                    t,
                    audio_mask,
                    rope_audio,
                    rope_text,
                    c_mask,
                    use_reentrant=False,
                )
            else:
                c, x = block(x, c, t, mask=audio_mask, rope=rope_audio, c_rope=rope_text, c_mask=c_mask)

        # Phase 2: Concatenate text+audio -> single-stream DiTBlock
        x = torch.cat([c, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(text_len + seq_len)

        if audio_mask is not None:
            single_mask = torch.cat([c_mask, audio_mask], dim=1)
        else:
            single_mask = None

        for i, block in enumerate(self.single_transformer_blocks):
            if self.checkpoint_activations and i % self.checkpoint_every_n_layers == 0:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, single_mask, rope, use_reentrant=False
                )
            else:
                x = block(x, t, mask=single_mask, rope=rope)

        # extract noised-target portion: drop text prefix and (seq_prepend) prompt prefix
        x = x[:, text_len + prompt_len :]
        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
