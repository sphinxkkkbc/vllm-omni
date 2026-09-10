# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.auk.auk_flow import (
    FLASH_TIMES,
    AukInputBatch,
    stage1_checkpoint_name,
)
from vllm_omni.model_executor.models.auk.configuration_auk import AukConfig
from vllm_omni.model_executor.models.auk.modules.vae import BigVGANFlowVAE, BigVGANFlowVAEConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_auk_config_round_trip(tmp_path):
    config = AukConfig(transformer_dim=768, transformer_num_layers=3)
    config.save_pretrained(tmp_path)
    restored = AukConfig.from_pretrained(tmp_path)
    fields = (
        "transformer_dim",
        "transformer_heads",
        "transformer_head_dim",
        "transformer_ff_mult",
        "transformer_num_layers",
        "transformer_num_single_layers",
        "text_hidden_dim",
        "latent_dim",
        "attn_backend",
        "attn_mask_enabled",
        "auk_checkpoint",
        "vae_checkpoint",
        "vae_config",
    )
    assert {name: getattr(restored, name) for name in fields} == {name: getattr(config, name) for name in fields}


def test_flash_defaults_match_official_release():
    config = AukConfig()
    assert (
        config.transformer_dim,
        config.transformer_heads,
        config.transformer_head_dim,
        config.transformer_ff_mult,
        config.text_hidden_dim,
        config.transformer_num_layers,
        config.transformer_num_single_layers,
        config.latent_dim,
        config.attn_backend,
        config.attn_mask_enabled,
    ) == (1536, 24, 64, 2.0, 2048, 10, 20, 64, "torch", True)
    assert tuple(config.flash_timesteps) == FLASH_TIMES


def test_flow_time_tensor_matches_backbone_dtype():
    condition = SimpleNamespace(
        semantic_embeddings=torch.ones(1, 2, 4),
        semantic_mask=torch.ones(1, 2, dtype=torch.bool),
        reference_latents=torch.ones(1, 3, 64),
        reference_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    batch = AukInputBatch.materialize(
        [condition],
        [torch.ones(1, 5, 64)],
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert batch.inputs["time"].dtype == torch.bfloat16
    assert batch.inputs["x"].dtype == torch.bfloat16
    assert batch.inputs["text"].dtype == torch.bfloat16


def test_flow_constructor_uses_only_auk_config(monkeypatch):
    import vllm_omni.model_executor.models.auk.auk_flow as flow

    captured = {}

    class Backbone(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    class VAE(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            captured["vae_config"] = config

    monkeypatch.setattr(flow, "Flux2Edit", Backbone)
    monkeypatch.setattr(flow, "BigVGANFlowVAE", VAE)
    config = AukConfig()
    flow.AukFlowModel(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config)))
    assert {k: captured[k] for k in ("dim", "heads", "dim_head", "ff_mult", "text_hidden_dim")} == {
        "dim": 1536,
        "heads": 24,
        "dim_head": 64,
        "ff_mult": 2.0,
        "text_hidden_dim": 2048,
    }
    assert captured["num_layers"] == 10 and captured["num_single_layers"] == 20


def test_checkpoint_namespace_mapping_and_consumption_basis():
    assert stage1_checkpoint_name("backbone.txt_proj.weight") == "transformer.txt_proj.weight"
    assert stage1_checkpoint_name("vae.conv_pre.weight_g") == "conv_pre.weight_g"
    consumed_checkpoint_keys = {stage1_checkpoint_name("backbone.txt_proj.weight")}
    assert "transformer.txt_proj.weight" not in {"transformer.txt_proj.weight"} - consumed_checkpoint_keys


def test_loader_tracks_parameter_names_separately_from_checkpoint_keys():
    import vllm_omni.model_executor.models.auk.auk_flow as flow

    model = flow.AukFlowModel.__new__(flow.AukFlowModel)
    torch.nn.Module.__init__(model)
    model.backbone = torch.nn.Linear(2, 2, bias=False)
    model.vae = torch.nn.Linear(2, 1, bias=True)
    incoming = [
        ("transformer.weight", torch.full((2, 2), 3.0)),
        ("weight", torch.full((1, 2), 4.0)),
        ("bias", torch.full((1,), 5.0)),
        ("layer_weights", torch.zeros(2)),
        ("text_encoder.anything", torch.zeros(1)),
    ]
    assert model.load_weights(incoming) == {"backbone.weight", "vae.weight", "vae.bias"}
    torch.testing.assert_close(model.backbone.weight, torch.full((2, 2), 3.0))


def test_bigvgan_config_downsample_is_480():
    vae = BigVGANFlowVAEConfig.from_dict(AukConfig().vae_config)
    assert math.prod(vae.downsample_rates) == 480
    assert math.prod(vae.upsample_rates) == 480
    assert vae.latent_dim == 64


def test_bigvgan_latent_shape_contract():
    config = BigVGANFlowVAEConfig(
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        upsample_initial_channel=32,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        downsample_rates=[2],
        downsample_channels=[12, 24],
        latent_dim=8,
        flow_hidden_channels=16,
    )
    vae = BigVGANFlowVAE(config).eval()
    assert vae.inference_from_latents(torch.randn(1, 8, 3)).shape == (1, 1, 6)
    with pytest.raises(ValueError, match=r"\[B,D,T\]"):
        vae.inference_from_latents(torch.randn(1, 3, 8))


def test_flash_euler_loop_parity():
    x = torch.randn(2, 7, 64)
    expected = x.clone()
    actual = x.clone()

    def velocity(value, time):
        return value.square() * 0.01 + time

    for left, right in zip(FLASH_TIMES[:-1], FLASH_TIMES[1:]):
        expected = expected + (right - left) * velocity(expected, left)
    for i in range(4):
        actual.add_((FLASH_TIMES[i + 1] - FLASH_TIMES[i]) * velocity(actual, FLASH_TIMES[i]))
    torch.testing.assert_close(actual, expected)
