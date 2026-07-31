# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.segmented_graph_wrapper import CUDAGraphDecoderWrapper
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix_frames = None

    def _decode_cached_incremental(self, codes, caches, prefix_frames):
        self.prefix_frames = prefix_frames
        return codes


def test_eager_fallback_uses_dynamic_decoder_prefix_length():
    decoder = _RecordingDecoder()
    wrapper = CUDAGraphDecoderWrapper(decoder, enabled=False)
    codes = torch.zeros(1, 2, 2)

    output, incremental = wrapper._decode(
        codes,
        {
            "prefix_frames": 48,
            "decoder_prefix_frames": 48,
            "suffix_quantized": torch.zeros(1, 2, 1),
        },
        clone_graph_output=True,
    )

    assert output is codes
    assert incremental is True
    assert decoder.prefix_frames == 48


def test_graph_prefix_cache_keeps_physical_prefix_length(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_graph = type("_Graph", (), {"replay": lambda self: None})()
    wrapper.prefix_static_input = torch.zeros(1, 2, 73)
    wrapper.prefix_static_output = torch.zeros(1, 1, 73)
    wrapper.prefix_suffix_quantized = torch.zeros(1, 2, 1)
    wrapper.prefix_suffix_conv = torch.zeros(1, 1, 2)
    wrapper.prefix_hidden_mask = torch.ones(1, 1, 73)
    wrapper.prefix_attention_mask = torch.ones(1, 72, dtype=torch.bool)
    wrapper.suffix_attention_mask = torch.ones(1, 73, dtype=torch.bool)
    wrapper.prefix_length = 72
    wrapper.decoder = type("_Decoder", (), {"total_upsample": 1})()
    wrapper.static_caches = {"decoder_prefix_frames": 72}
    wrapper._ensure_suffix_buffers = lambda caches: None

    caches = {"prefix_frames": 48}
    wrapper._decode_icl_prefix(torch.zeros(1, 2, 49), caches)

    assert caches["prefix_frames"] == 48
    assert caches["decoder_prefix_frames"] == 72


def test_incremental_downstream_matches_full_decode_tail():
    torch.manual_seed(0)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    hidden = torch.randn(1, 97, config.latent_dim)
    new_frames = 25

    with torch.no_grad():
        full = hidden.permute(0, 2, 1)
        for blocks in decoder.upsample:
            for block in blocks:
                full = block(full)
        for block in decoder.decoder:
            full = block(full)
        expected = full[..., -new_frames * decoder.total_upsample :].clamp(min=-1, max=1)
        actual = decoder._decode_downstream_incremental(hidden, new_frames)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
