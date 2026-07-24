import html
import math
import string
from typing import Literal

import ftfy
import regex as re
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer

from .wan_dit import DiTBlock


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)

        # arguments
        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update({"padding": "max_length", "truncation": True, "max_length": self.seq_len})
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text


class WanPrompter:
    def __init__(self, tokenizer_path, text_encoder, text_len=512):
        super().__init__()
        self.text_len = text_len
        self.text_encoder = text_encoder
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=self.text_len, clean="whitespace")

    def encode_prompt(self, prompt, positive=True, device="cuda"):
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        # Zero the tail of each sample by its valid length so a shorter sample
        # in a batch is not truncated to the longest sample's length.
        for i, v in enumerate(seq_lens.tolist()):
            prompt_emb[i, v:] = 0
        return prompt_emb


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def legacy_precompute_freqs_cis_1d(
    dim: int, end: int = 16384, theta: float = 10000.0, base_tps=4.0, target_tps=44100 / 2048
):
    s = float(base_tps) / float(target_tps)
    # 1d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, s)
    # No positional encoding applied to the remaining dimensions.
    no_freqs_cis = precompute_freqs_cis(dim // 3, end, theta, s)
    no_freqs_cis = torch.ones_like(no_freqs_cis)
    return f_freqs_cis, no_freqs_cis, no_freqs_cis


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim, end, theta)
    return f_freqs_cis.chunk(3, dim=-1)


def precompute_freqs_cis(dim: int, end: int = 16384, theta: float = 10000.0, s: float = 1.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64, device=freqs.device) * s
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        # print(f"{t_mod.shape = }")
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            # t_mod is originally [B, C]; broadcasting works for B=1 but not for
            # B>1 against [1, 2, C], so reshape explicitly here.
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)).chunk(
                2, dim=1
            )
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanAudioModel(nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        vae_type: Literal["oobleck", "dac"] = "oobleck",
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.vae_type = vae_type
        # self.patch_embedding = nn.Conv3d(
        #     in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding = nn.Conv1d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        if vae_type == "oobleck":
            freqs = legacy_precompute_freqs_cis_1d(head_dim, base_tps=4.0, target_tps=44100 / 2048)
        elif vae_type == "dac":
            freqs = precompute_freqs_cis_1d(head_dim)
        else:
            raise ValueError(f"Invalid VAE type: {vae_type}")
        # Register RoPE freqs as buffers so model.to(device) / accelerate move
        # them with the model, and so forward / torch.compile graphs do not
        # mutate Python attributes mid-trace.
        self.register_buffer("freqs_cis_0", freqs[0], persistent=False)
        self.register_buffer("freqs_cis_1", freqs[1], persistent=False)
        self.register_buffer("freqs_cis_2", freqs[2], persistent=False)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

    @property
    def freqs(self):
        # Backwards-compatible accessor: external code can still use self.freqs[i].
        return (self.freqs_cis_0, self.freqs_cis_1, self.freqs_cis_2)

    def patchify(self, x: torch.Tensor, control_camera_latents_input: torch.Tensor | None = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f -> b f c").contiguous()
        return x, grid_size  # x, grid_size: (f)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(x, "b f (p c) -> b c (f p)", f=grid_size[0], p=self.patch_size[0])

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        **kwargs,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        x, (f,) = self.patchify(x)

        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, -1).expand(f, -1),
                self.freqs[1][:f].view(f, -1).expand(f, -1),
                self.freqs[2][:f].view(f, -1).expand(f, -1),
            ],
            dim=-1,
        ).reshape(f, 1, -1)

        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f,))
        return x
