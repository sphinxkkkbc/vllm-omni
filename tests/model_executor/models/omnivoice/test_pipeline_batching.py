# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight batching invariants for the OmniVoice pipeline."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import OmniVoicePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minimal_pipeline() -> OmniVoicePipeline:
    pipeline = OmniVoicePipeline.__new__(OmniVoicePipeline)
    nn.Module.__init__(pipeline)
    pipeline.config = SimpleNamespace(audio_mask_id=1024)
    return pipeline


def _state(request_id: str, seq_len: int, fill: int):
    return SimpleNamespace(
        request_id=request_id,
        latents=torch.full((2, 8, seq_len), fill, dtype=torch.long),
        extra={
            "audio_mask": torch.ones(2, seq_len, dtype=torch.bool),
            "attn_mask": torch.ones(2, 1, seq_len, seq_len, dtype=torch.bool),
        },
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_cfg_layout_round_trip(batch_size: int) -> None:
    request_major = torch.arange(2 * batch_size).reshape(2 * batch_size, 1)

    cfg_major = OmniVoicePipeline._request_major_to_cfg_major(request_major, batch_size)
    restored = OmniVoicePipeline._cfg_major_to_request_major(cfg_major, batch_size)

    expected_cfg = torch.tensor([*range(0, 2 * batch_size, 2), *range(1, 2 * batch_size, 2)]).reshape(2 * batch_size, 1)
    assert torch.equal(cfg_major, expected_cfg)
    assert torch.equal(restored, request_major)


def test_new_longer_request_repads_existing_state_and_masks() -> None:
    pipeline = _minimal_pipeline()
    old = _state("old", seq_len=5, fill=11)
    new = _state("new", seq_len=8, fill=22)

    old_latents = old.latents.clone()
    old_audio_mask = old.extra["audio_mask"].clone()
    old_attn_mask = old.extra["attn_mask"].clone()

    states = pipeline.prepare_state_batch([old, new], new_request_ids=["new"])

    assert states[0] is old
    assert states[1] is new
    assert old.latents.shape == new.latents.shape == (2, 8, 8)
    assert old.extra["audio_mask"].shape == new.extra["audio_mask"].shape == (2, 8)
    assert old.extra["attn_mask"].shape == new.extra["attn_mask"].shape == (2, 1, 8, 8)

    torch.testing.assert_close(old.latents[..., :5], old_latents)
    assert torch.all(old.latents[..., 5:] == 1024)
    assert torch.equal(old.extra["audio_mask"][..., :5], old_audio_mask)
    assert not torch.any(old.extra["audio_mask"][..., 5:])
    assert torch.equal(old.extra["attn_mask"][..., :5, :5], old_attn_mask)
    assert not torch.any(old.extra["attn_mask"][..., 5:, :])
    assert not torch.any(old.extra["attn_mask"][..., :, 5:])


def test_later_short_request_is_padded_to_existing_active_length() -> None:
    pipeline = _minimal_pipeline()
    first = _state("first", seq_len=8, fill=11)
    second = _state("second", seq_len=8, fill=22)
    newcomer = _state("new", seq_len=6, fill=33)

    pipeline.prepare_state_batch([first, second, newcomer], new_request_ids=["new"])

    assert first.latents.shape == second.latents.shape == newcomer.latents.shape == (2, 8, 8)
    assert torch.all(newcomer.latents[..., :6] == 33)
    assert torch.all(newcomer.latents[..., 6:] == 1024)
    assert not torch.any(newcomer.extra["audio_mask"][..., 6:])
    assert not torch.any(newcomer.extra["attn_mask"][..., 6:, :])
    assert not torch.any(newcomer.extra["attn_mask"][..., :, 6:])


def test_audio_outputs_are_trimmed_to_each_target_length() -> None:
    audio = torch.arange(2 * 10 * 960, dtype=torch.float32).reshape(2, 1, 10 * 960)

    outputs = OmniVoicePipeline._split_audio_outputs(audio, target_lens=[3, 7])

    assert len(outputs) == 2
    assert outputs[0].output.shape == (1, 1, 3 * 960)
    assert outputs[1].output.shape == (1, 1, 7 * 960)
    assert torch.equal(outputs[0].output, audio[0:1, :, : 3 * 960])
    assert torch.equal(outputs[1].output, audio[1:2, :, : 7 * 960])
