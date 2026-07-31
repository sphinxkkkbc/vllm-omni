# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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
        self.config = SimpleNamespace(sliding_window=72)
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


def test_segmented_capture_sizes_are_derived_from_decoder_and_chunk_config():
    decoder = _RecordingDecoder()
    decoder.config.sliding_window = 10

    wrapper = CUDAGraphDecoderWrapper(
        decoder,
        initial_chunk_frames=2,
        codec_chunk_frames=4,
        enabled=False,
    )

    assert wrapper.prefix_length == 10
    assert wrapper.capture_sizes == [6, 10, 14]
    assert wrapper._previous_frames_by_target == {6: 2, 10: 6, 14: 10}


def test_graph_prefix_cache_keeps_physical_prefix_length(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_graphs = {1: type("_Graph", (), {"replay": lambda self: None})()}
    wrapper.prefix_static_inputs = {1: torch.zeros(1, 2, 73)}
    wrapper.prefix_static_outputs = {1: torch.zeros(1, 1, 73)}
    wrapper.prefix_suffix_quantized = {1: torch.zeros(1, 2, 1)}
    wrapper.prefix_suffix_conv = {1: torch.zeros(1, 1, 2)}
    wrapper.prefix_hidden_masks = {1: torch.ones(1, 1, 73)}
    wrapper.prefix_attention_masks = {1: torch.ones(1, 72, dtype=torch.bool)}
    wrapper.suffix_attention_masks = {1: torch.ones(1, 73, dtype=torch.bool)}
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.decoder = type("_Decoder", (), {"total_upsample": 1})()
    wrapper.prefix_static_caches = {1: {"decoder_prefix_frames": 72}}
    wrapper._ensure_suffix_buffers = lambda caches: None

    caches = {"prefix_frames": 48}
    wrapper._decode_icl_prefix(torch.zeros(1, 2, 49), caches)

    assert caches["prefix_frames"] == 48
    assert caches["decoder_prefix_frames"] == 72


def test_batched_chunked_decode_groups_exact_phases(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper.decoder = SimpleNamespace(_is_suffix_cache_rolling=lambda previous, cached: previous >= 72 and cached == 72)
    calls: list[tuple[str, int, int]] = []

    def _prefix(codes, caches):
        calls.append(("prefix", 0, len(codes)))
        return [torch.full((1, 1, 1), 1.0) for _ in codes]

    def _suffix(target, codes, caches, new_frames):
        calls.append(("suffix", target, len(codes)))
        return [torch.full((1, 1, 1), float(target)) for _ in codes]

    monkeypatch.setattr(wrapper, "_decode_prefix_batch", _prefix)
    monkeypatch.setattr(wrapper, "_decode_suffix_batch", _suffix)

    class _KV:
        @staticmethod
        def get_seq_length():
            return 72

    prefix_frames = 48
    codes_list = [torch.zeros(1, 2, prefix_frames + 1)]
    caches = [{"prefix_frames": prefix_frames}]
    for previous, suffix_frames, cached_frames in (
        (1, 26, 1),
        (26, 51, 26),
        (51, 76, 51),
        (76, 97, 72),
    ):
        codes_list.append(torch.zeros(1, 2, prefix_frames + suffix_frames))
        caches.append(
            {
                "prefix_frames": prefix_frames,
                "decoder_prefix_frames": 72,
                "ref_wav": torch.zeros(1, 1, 1),
                "past_key_values": _KV(),
                "suffix_frames": previous,
                "suffix_quantized": torch.zeros(1, 2, cached_frames),
            }
        )

    lengths = [codes.shape[-1] for codes in codes_list]
    padded_codes = torch.zeros(len(codes_list), 2, max(lengths))
    for row, codes in enumerate(codes_list):
        padded_codes[row, :, : codes.shape[-1]] = codes[0]
    outputs = wrapper.batched_chunked_decode_with_cudagraph(padded_codes, lengths, caches=caches)

    assert calls == [
        ("prefix", 0, 1),
        ("suffix", 26, 1),
        ("suffix", 51, 1),
        ("suffix", 76, 1),
        ("suffix", 97, 1),
    ]
    assert [int(output.item()) for output in outputs] == [1, 26, 51, 76, 97]
    assert [cache["_last_output_incremental_audio"] for cache in caches] == [False, True, True, True, True]


def test_graph_batch_overflow_splits_at_largest_bucket():
    assert CUDAGraphDecoderWrapper._split_for_graph_buckets(16, {1, 2, 4, 8}) == [(0, 8), (8, 16)]
    assert CUDAGraphDecoderWrapper._split_for_graph_buckets(10, {1, 2, 4, 8}) == [(0, 8), (8, 10)]


def test_batched_suffix_graph_updates_ping_pong_buffers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.codec_chunk_frames = 25
    wrapper._previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper.capture_batch_sizes = [1, 2]
    wrapper.combined_graphs = {(2, 26): type("_Graph", (), {"replay": lambda self: None})()}
    wrapper.combined_static_codes = {(2, 26): torch.zeros(2, 2, 25)}
    wrapper.combined_static_quantized = {(2, 26): torch.zeros(2, 2, 1)}
    wrapper.combined_static_conv = {(2, 26): torch.zeros(2, 1, 3)}
    wrapper.combined_static_masks = {(2, 26): torch.ones(2, 98, dtype=torch.bool)}
    wrapper.combined_static_caches = {(2, 26): {}}
    graph_output = torch.stack(
        [
            torch.arange(50, dtype=torch.float32).view(1, -1),
            torch.arange(50, dtype=torch.float32).view(1, -1) + 100,
        ]
    )
    graph_next_quantized = torch.stack([torch.full((2, 26), 1.0), torch.full((2, 26), 2.0)])
    graph_next_conv = torch.stack([torch.full((26, 3), 3.0), torch.full((26, 3), 4.0)])
    wrapper.combined_static_outputs = {(2, 26): (graph_output, graph_next_quantized, graph_next_conv)}
    wrapper.decoder = type("_Decoder", (), {"total_upsample": 2})()
    monkeypatch.setattr(wrapper, "_copy_request_caches_to_batch", lambda caches, static: None)

    caches = [
        {
            "suffix_quantized": torch.zeros(1, 2, 1),
            "suffix_conv": torch.zeros(1, 1, 3),
            "suffix_frames": 1,
        }
        for _ in range(2)
    ]
    codes = [torch.zeros(1, 2, 26) for _ in range(2)]

    outputs = wrapper._decode_suffix_batch(26, codes, caches, [25, 25])

    assert outputs is not None
    assert [cache["_suffix_buffer_index"] for cache in caches] == [1, 1]
    assert caches[0]["suffix_quantized"].data_ptr() == caches[0]["_suffix_quantized_buffers"][1].data_ptr()
    assert caches[1]["suffix_conv"].data_ptr() == caches[1]["_suffix_conv_buffers"][1].data_ptr()
    torch.testing.assert_close(caches[0]["suffix_quantized"], torch.full((1, 2, 26), 1.0))
    torch.testing.assert_close(caches[1]["suffix_quantized"], torch.full((1, 2, 26), 2.0))
    torch.testing.assert_close(caches[0]["suffix_conv"], torch.full((1, 26, 3), 3.0))
    torch.testing.assert_close(caches[1]["suffix_conv"], torch.full((1, 26, 3), 4.0))


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
