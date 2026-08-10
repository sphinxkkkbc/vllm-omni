# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.step_audio2.cosyvoice2.dit import DiT

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

external_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")


def _make_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    kwargs = {
        "in_channels": 320,
        "out_channels": 80,
        "mlp_ratio": 2.0,
        "depth": 2,
        "num_heads": 2,
        "head_dim": 32,
        "hidden_size": 64,
    }
    torch.manual_seed(0)
    reference = external_dit.DiT(**kwargs).eval()
    vendored = DiT(**kwargs).eval()

    # This is the same strict compatibility guarantee used when loading
    # Step-Audio2's flow.pt checkpoint at runtime.
    vendored.load_state_dict(reference.state_dict(), strict=True)
    return reference, vendored


def _chunk_inputs(model: torch.nn.Module, seq_len: int = 7):
    batch = 2
    depth = len(model.blocks)
    heads = model.blocks[0].attn.num_heads
    head_dim = model.blocks[0].attn.head_dim
    hidden_size = model.blocks[0].conv.in_channels

    x = torch.randn(batch, model.in_channels, seq_len)
    time = model.t_embedder(torch.rand(batch)).unsqueeze(1)
    cnn_out = torch.empty(depth, batch, hidden_size * 2, 2)
    att_out = torch.empty(depth, batch, heads, seq_len, head_dim * 2)
    return x, time, None, [None] * depth, [None] * depth, cnn_out, att_out


def test_vendored_dit_forward_matches_external() -> None:
    reference, vendored = _make_models()
    batch, seq_len = 2, 7
    inputs = {
        "x": torch.randn(batch, 80, seq_len),
        "mask": torch.ones(batch, 1, seq_len),
        "mu": torch.randn(batch, 80, seq_len),
        "t": torch.rand(batch),
        "spks": torch.randn(batch, 80),
        "cond": torch.randn(batch, 80, seq_len),
    }

    with torch.inference_mode():
        expected = reference(**inputs)
        actual = vendored(**inputs)

    torch.testing.assert_close(actual, expected)


def test_vendored_dit_uncached_chunk_matches_external() -> None:
    reference, vendored = _make_models()
    inputs = _chunk_inputs(reference)
    reference_buffers = (inputs[-2].clone(), inputs[-1].clone())
    vendored_buffers = (inputs[-2].clone(), inputs[-1].clone())

    with torch.inference_mode():
        expected = reference.blocks_forward_chunk(
            *inputs[:5], *reference_buffers
        )
        actual = vendored.blocks_forward_chunk(
            *inputs[:5], *vendored_buffers
        )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(vendored_buffers[0], reference_buffers[0])
    torch.testing.assert_close(vendored_buffers[1], reference_buffers[1])


def test_vendored_dit_cached_chunk_matches_external() -> None:
    reference, vendored = _make_models()
    first_inputs = _chunk_inputs(reference)
    reference_first_buffers = (first_inputs[-2].clone(), first_inputs[-1].clone())
    vendored_first_buffers = (first_inputs[-2].clone(), first_inputs[-1].clone())

    with torch.inference_mode():
        reference.blocks_forward_chunk(*first_inputs[:5], *reference_first_buffers)
        vendored.blocks_forward_chunk(*first_inputs[:5], *vendored_first_buffers)

    seq_len = 5
    batch = first_inputs[0].shape[0]
    depth = len(reference.blocks)
    heads = reference.blocks[0].attn.num_heads
    head_dim = reference.blocks[0].attn.head_dim
    hidden_size = reference.blocks[0].conv.in_channels
    x = torch.randn(batch, reference.in_channels, seq_len)
    time = reference.t_embedder(torch.rand(batch)).unsqueeze(1)
    reference_buffers = (
        torch.empty(depth, batch, hidden_size * 2, 2),
        torch.empty(depth, batch, heads, first_inputs[0].shape[-1] + seq_len, head_dim * 2),
    )
    vendored_buffers = tuple(buffer.clone() for buffer in reference_buffers)

    with torch.inference_mode():
        expected = reference.blocks_forward_chunk(
            x,
            time,
            None,
            reference_first_buffers[0],
            reference_first_buffers[1],
            *reference_buffers,
        )
        actual = vendored.blocks_forward_chunk(
            x,
            time,
            None,
            vendored_first_buffers[0],
            vendored_first_buffers[1],
            *vendored_buffers,
        )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(vendored_buffers[0], reference_buffers[0])
    torch.testing.assert_close(vendored_buffers[1], reference_buffers[1])
