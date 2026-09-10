# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK configuration for vLLM-Omni."""

import math
from pathlib import Path

from transformers import PretrainedConfig


class AukConfig(PretrainedConfig):
    """Configuration for AuK model."""

    model_type = "auk"

    def __init__(
        self,
        auk_checkpoint="auk_flash.safetensors",
        qwen_path="Qwen/Qwen2.5-Omni-3B",
        transformer_dim=1536,
        transformer_heads=24,
        transformer_head_dim=64,
        transformer_ff_mult=2.0,
        transformer_num_layers=10,
        transformer_num_single_layers=20,
        text_hidden_dim=2048,
        latent_dim=64,
        attn_backend="torch",
        attn_mask_enabled=True,
        sample_rate=24000,
        downsample_rate=480,
        flash_timesteps=None,
        dtype="float32",
        vae_config=None,
        vae_checkpoint="vae.safetensors",
        bucket_granularity=1,
        graph_buckets=None,
        **kwargs,
    ):
        # Standard HF config fields
        defaults = dict(
            dtype=dtype,
            # vLLM's generic AR runner requires this compatibility field for
            # Stage0 buffer sizing. Stage1 deliberately uses text_hidden_dim.
            hidden_size=text_hidden_dim,
            vocab_size=151936,  # Qwen vocab size
            max_position_embeddings=8192,
            architectures=["AukConditionModel", "AukFlowModel"],
        )
        defaults.update(kwargs)
        super().__init__(**defaults)

        # AuK-specific paths
        self.auk_checkpoint = auk_checkpoint
        self.qwen_path = qwen_path
        if latent_dim != 64:
            raise ValueError("AuK flow latents must have dimension 64")
        if sample_rate != 24000 or downsample_rate != 480:
            raise ValueError("AuK-Flash requires 24 kHz audio and a 480-sample hop")
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.transformer_head_dim = transformer_head_dim
        self.transformer_ff_mult = transformer_ff_mult
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_single_layers = transformer_num_single_layers
        self.text_hidden_dim = text_hidden_dim
        self.latent_dim = latent_dim
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled
        self.sample_rate = sample_rate
        self.downsample_rate = downsample_rate
        self.flash_timesteps = flash_timesteps or [
            0.0,
            0.07612049579620361,
            0.2928932309150696,
            0.6173166036605835,
            1.0,
        ]
        self.vae_config = vae_config or {
            "upsample_rates": [5, 4, 3, 2, 2, 2],
            "upsample_kernel_sizes": [10, 8, 6, 4, 4, 4],
            "upsample_initial_channel": 1536,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "downsample_rates": [2, 2, 2, 3, 4, 5],
            "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
            "snake_logscale": True,
            "latent_dim": 64,
            "use_vae": True,
            "causal": True,
            "flow_hidden_channels": 256,
            "act_causal": True,
        }
        if math.prod(self.vae_config["downsample_rates"]) != downsample_rate:
            raise ValueError("VAE downsample_rates must multiply to downsample_rate")
        self.vae_checkpoint = vae_checkpoint
        self.bucket_granularity = bucket_granularity
        self.graph_buckets = graph_buckets or []


def resolve(root: str, filename: str) -> str:
    """Resolve model file path."""
    if Path(root).is_dir():
        path = Path(root) / filename
        if path.is_file():
            return str(path)
    return filename  # Will be handled by weight loading


# Configuration only
__all__ = ["AukConfig"]
