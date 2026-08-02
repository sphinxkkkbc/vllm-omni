# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections import Counter
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


def _decoder_stub(**kwargs):
    decoder = SimpleNamespace(**kwargs)
    decoder.batched_request_decode = Qwen3TTSTokenizerV2Decoder.batched_request_decode.__get__(decoder)
    return decoder


class _RecordingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(sliding_window=72)
        self.prefix_frames = None

    def decode_suffix(self, codes, caches, prefix_frames):
        self.prefix_frames = prefix_frames
        return codes


def test_eager_fallback_uses_dynamic_decoder_prefix_length():
    decoder = _RecordingDecoder()
    wrapper = CUDAGraphDecoderWrapper(decoder, enabled=False)
    codes = torch.zeros(1, 2, 2)

    output = wrapper._decode(
        codes,
        {
            "prefix_frames": 48,
            "decoder_prefix_frames": 48,
            "suffix_quantized": torch.zeros(1, 2, 1),
        },
        clone_graph_output=True,
    )

    assert output is codes
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
    assert wrapper.icl_capture_sizes == [6, 10, 14]
    assert wrapper._icl_previous_frames_by_target == {6: 2, 10: 6, 14: 10}
    assert wrapper._xvec_previous_frames_by_target == {6: 2, 10: 6, 14: 10}


def test_stateless_capture_sizes_extend_defaults_with_configured_sizes():
    decoder = _RecordingDecoder()
    wrapper = CUDAGraphDecoderWrapper(
        decoder,
        async_chunk=False,
        decode_chunk_size=300,
        decode_left_context=25,
        stateless_capture_sizes=[97, 400],
        enabled=False,
    )

    assert wrapper.stateless_capture_sizes == [97, 150, 325, 400]


def test_decode_modes_enforce_cache_contracts():
    decoder = _RecordingDecoder()
    async_wrapper = CUDAGraphDecoderWrapper(decoder, async_chunk=True, enabled=False)
    stateless_wrapper = CUDAGraphDecoderWrapper(decoder, async_chunk=False, enabled=False)
    codes = torch.zeros(1, 2, 1)

    with pytest.raises(ValueError, match="caches are required"):
        async_wrapper.chunked_decode_with_cudagraph(codes, caches=None)
    with pytest.raises(ValueError, match="caches must be None"):
        stateless_wrapper.chunked_decode_with_cudagraph(codes, caches={})


def test_stateless_replay_pads_to_bucket_and_trims_output(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    class _Graph:
        def replay(self):
            return None

    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.async_chunk = False
    wrapper.enabled = True
    wrapper._warmed_up = True
    wrapper.stateless_capture_sizes = [4]
    wrapper.stateless_graphs = {(1, 4): _Graph()}
    wrapper.stateless_static_inputs = {(1, 4): torch.full((1, 2, 4), -1, dtype=torch.long)}
    wrapper.stateless_static_outputs = {(1, 4): torch.arange(4, dtype=torch.float32).view(1, 1, 4)}
    wrapper.decoder = SimpleNamespace(total_upsample=1)
    codes = torch.tensor([[[1, 2], [3, 4]]])

    output = wrapper._decode_stateless(codes, clone_graph_output=True)

    torch.testing.assert_close(output, torch.tensor([[[0.0, 1.0]]]))
    torch.testing.assert_close(
        wrapper.stateless_static_inputs[(1, 4)],
        torch.tensor([[[1, 2, 0, 0], [3, 4, 0, 0]]]),
    )


def test_capture_uses_cached_conv_lengths():
    decoder = _RecordingDecoder()
    decoder.config.sliding_window = 72
    wrapper = CUDAGraphDecoderWrapper(
        decoder,
        initial_chunk_frames=1,
        codec_chunk_frames=25,
        enabled=False,
    )

    assert [wrapper._cached_conv_frames(frames) for frames in (1, 26, 51, 72)] == [1, 26, 51, 70]


def test_icl_prefix_graph_cache_keeps_physical_prefix_length(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.icl_prefix_graphs = {1: type("_Graph", (), {"replay": lambda self: None})()}
    wrapper.icl_prefix_static_inputs = {1: torch.zeros(1, 2, 73)}
    wrapper.icl_prefix_static_outputs = {1: torch.zeros(1, 1, 73)}
    wrapper.icl_prefix_suffix_quantized = {1: torch.zeros(1, 2, 1)}
    wrapper.icl_prefix_suffix_conv = {1: torch.zeros(1, 1, 2)}
    wrapper.icl_prefix_hidden_masks = {1: torch.ones(1, 1, 73)}
    wrapper.icl_prefix_attention_masks = {1: torch.ones(1, 72, dtype=torch.bool)}
    wrapper.icl_suffix_attention_masks = {1: torch.ones(1, 73, dtype=torch.bool)}
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.decoder = type("_Decoder", (), {"total_upsample": 1})()
    wrapper.icl_prefix_static_caches = {1: {"decoder_prefix_frames": 72}}
    wrapper._ensure_suffix_buffers = lambda caches: None

    caches = {"prefix_frames": 48}
    wrapper._decode_icl_prefix(torch.zeros(1, 2, 49), caches)

    assert caches["prefix_frames"] == 48
    assert caches["decoder_prefix_frames"] == 72


def test_batched_chunked_decode_groups_exact_phases(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.async_chunk = True
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper._xvec_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper.decoder = _decoder_stub(_is_suffix_cache_rolling=lambda previous, cached: previous >= 72 and cached == 72)
    calls: list[tuple[str, int, int]] = []

    def _icl_prefix(codes, caches):
        calls.append(("icl_prefix", 0, len(codes)))
        return [torch.full((1, 1, 1), 1.0) for _ in codes]

    def _suffix(mode, target, codes, caches, new_frames):
        calls.append((f"suffix:{mode}", target, len(codes)))
        return [torch.full((1, 1, 1), float(target)) for _ in codes]

    monkeypatch.setattr(wrapper, "_decode_icl_prefix_batch", _icl_prefix)
    monkeypatch.setattr(wrapper, "_decode_suffix_batch", _suffix)

    class _KV:
        @staticmethod
        def get_seq_length():
            return 72

    prefix_frames = 48
    codes_list = [torch.zeros(1, 2, prefix_frames + 1)]
    caches = [{"prefix_frames": prefix_frames}]
    for previous, _suffix_frames, cached_frames in (
        (1, 26, 1),
        (26, 51, 26),
        (51, 76, 51),
        (76, 97, 72),
    ):
        codes_list.append(torch.zeros(1, 2, 25))
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
        ("icl_prefix", 0, 1),
        ("suffix:icl", 26, 1),
        ("suffix:icl", 51, 1),
        ("suffix:icl", 76, 1),
        ("suffix:icl", 97, 1),
    ]
    assert [int(output.item()) for output in outputs] == [1, 26, 51, 76, 97]


def test_xvec_delta_chunks_route_through_all_reachable_graph_phases(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.async_chunk = True
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper._xvec_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper.decoder = _decoder_stub(_is_suffix_cache_rolling=lambda previous, cached: previous >= 72 and cached == 72)
    calls: list[tuple[str, int]] = []

    def _suffix(mode, target, codes, caches, new_frames):
        calls.append((mode, target))
        return [torch.zeros(1, 1, 1) for _ in codes]

    monkeypatch.setattr(wrapper, "_decode_suffix_batch", _suffix)

    class _EmptyKV:
        @staticmethod
        def get_seq_length():
            return 0

    caches = [
        {
            "prefix_frames": 0,
            "decoder_prefix_frames": 0,
            "ref_wav": torch.zeros(1, 1, 0),
            "past_key_values": _EmptyKV(),
            "suffix_frames": previous,
            "suffix_quantized": torch.zeros(1, 2, cached_frames),
        }
        for previous, cached_frames in ((1, 1), (26, 26), (51, 51), (76, 72))
    ]
    lengths = [25, 25, 25, 25]
    codes = torch.zeros(4, 2, 25)

    wrapper.batched_chunked_decode_with_cudagraph(codes, lengths, caches=caches)

    assert calls == [("xvec", 26), ("xvec", 51), ("xvec", 76), ("xvec", 97)]


def test_graph_batch_overflow_splits_at_largest_bucket():
    assert CUDAGraphDecoderWrapper._split_for_graph_buckets(16, {1, 2, 4, 8}) == [(0, 8), (8, 16)]
    assert CUDAGraphDecoderWrapper._split_for_graph_buckets(10, {1, 2, 4, 8}) == [(0, 8), (8, 10)]


def test_missing_graph_phase_falls_back_instead_of_returning_empty(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.icl_prefix_graphs = {}
    wrapper.combined_graphs = {}
    wrapper._record_graph_fallback = lambda *_args: None

    codes = [torch.zeros(1, 2, 25)]
    caches = [{}]

    assert wrapper._decode_icl_prefix_batch(codes, caches) is None
    assert wrapper._decode_suffix_batch("xvec", 50, codes, caches, [25]) is None


def test_xvec_dummy_cache_is_initialized_before_graph_capture():
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
        sliding_window=72,
    )
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.decoder = SimpleNamespace(config=config)

    cache = wrapper._make_dummy_xvec_cache(2, torch.device("cpu"), torch.float32)["past_key_values"]

    assert cache.get_seq_length() == 0
    assert all(layer.is_initialized for layer in cache.layers)
    assert all(layer.keys.device.type == "cpu" for layer in cache.layers)


def test_batched_suffix_graph_updates_ping_pong_buffers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper._xvec_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper.capture_batch_sizes = [1, 2]
    key = ("icl", 2, 26)
    wrapper.combined_graphs = {key: type("_Graph", (), {"replay": lambda self: None})()}
    wrapper.combined_static_codes = {key: torch.zeros(2, 2, 25)}
    wrapper.combined_static_quantized = {key: torch.zeros(2, 2, 1)}
    wrapper.combined_static_conv = {key: torch.zeros(2, 1, 3)}
    wrapper.combined_static_masks = {key: torch.ones(2, 98, dtype=torch.bool)}
    wrapper.combined_static_caches = {key: {}}
    graph_output = torch.stack(
        [
            torch.arange(50, dtype=torch.float32).view(1, -1),
            torch.arange(50, dtype=torch.float32).view(1, -1) + 100,
        ]
    )
    graph_next_quantized = torch.stack([torch.full((2, 26), 1.0), torch.full((2, 26), 2.0)])
    graph_next_conv = torch.stack([torch.full((26, 3), 3.0), torch.full((26, 3), 4.0)])
    wrapper.combined_static_outputs = {key: (graph_output, graph_next_quantized, graph_next_conv)}
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

    outputs = wrapper._decode_suffix_batch("icl", 26, codes, caches, [25, 25])

    assert outputs is not None
    assert [cache["_suffix_buffer_index"] for cache in caches] == [1, 1]
    assert caches[0]["suffix_quantized"].data_ptr() == caches[0]["_suffix_quantized_buffers"][1].data_ptr()
    assert caches[1]["suffix_conv"].data_ptr() == caches[1]["_suffix_conv_buffers"][1].data_ptr()
    torch.testing.assert_close(caches[0]["suffix_quantized"], torch.full((1, 2, 26), 1.0))
    torch.testing.assert_close(caches[1]["suffix_quantized"], torch.full((1, 2, 26), 2.0))
    torch.testing.assert_close(caches[0]["suffix_conv"], torch.full((1, 26, 3), 3.0))
    torch.testing.assert_close(caches[1]["suffix_conv"], torch.full((1, 26, 3), 4.0))


def test_xvec_first_chunk_uses_prefixless_initializer(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1}
    wrapper._xvec_previous_frames_by_target = {26: 1}
    calls = []
    wrapper.decoder = _decoder_stub(
        _decode_xvec_first_chunk=lambda codes, cache: calls.append("xvec") or codes[:, :1, :],
    )

    output = wrapper._batched_request_decode(
        [torch.zeros(1, 2, 25)],
        [{"prefix_frames": 0}],
    )

    assert calls == ["xvec"]
    assert output[0].shape == (1, 1, 25)


def test_dummy_request_does_not_record_pre_capture_shape_fallback():
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1}
    wrapper._xvec_previous_frames_by_target = {26: 1}
    wrapper.xvec_prefix_graphs = {}
    wrapper._stats_enabled = True
    wrapper._stats_total_requests = 0
    wrapper._stats_hit_requests = 0
    wrapper._stats_fallback_requests = 0
    wrapper._stats_replays = Counter()
    wrapper._stats_fallbacks = Counter()
    wrapper._maybe_log_stats = lambda: None
    wrapper.decoder = _decoder_stub(
        _decode_xvec_first_chunk=lambda codes, _cache: codes[:, :1, :],
    )

    wrapper._batched_request_decode(
        [torch.zeros(1, 2, 4)],
        [{"prefix_frames": 0, "_is_dummy_run": True}],
    )

    assert wrapper._stats_total_requests == 0
    assert wrapper._stats_fallback_requests == 0
    assert wrapper._stats_fallbacks == Counter()
    assert wrapper._suppress_stats is False


def test_eager_backend_max_batch_size_splits_each_phase_group():
    decoder = _decoder_stub()
    batch_sizes = []

    def decode_xvec(codes, _caches):
        batch_sizes.append(len(codes))
        return [code[:, :1, :].clone() for code in codes]

    outputs = decoder.batched_request_decode(
        [torch.zeros(1, 2, 1) for _ in range(5)],
        [{"prefix_frames": 0} for _ in range(5)],
        prefix_length=72,
        initial_chunk_frames=1,
        codec_chunk_frames=25,
        transitions_by_mode={"icl": {26: 1}, "xvec": {26: 1}},
        decode_icl_prefix_batch=None,
        decode_xvec_prefix_batch=decode_xvec,
        decode_suffix_batch=None,
        decode_eager=lambda *_args: pytest.fail("unexpected per-request fallback"),
        backend_max_batch_size=2,
    )

    assert batch_sizes == [2, 2, 1]
    assert len(outputs) == 5


def test_xvec_first_chunk_replays_prefix_graph(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper.capture_batch_sizes = [1, 2]
    wrapper._icl_previous_frames_by_target = {26: 1}
    wrapper._xvec_previous_frames_by_target = {26: 1}
    replayed = []
    wrapper.xvec_prefix_graphs = {2: SimpleNamespace(replay=lambda: replayed.append(True))}
    wrapper.xvec_prefix_static_inputs = {2: torch.zeros(2, 2, 1)}
    wrapper.xvec_prefix_static_outputs = {2: torch.arange(4, dtype=torch.float32).view(2, 1, 2)}
    wrapper.xvec_prefix_static_caches = {
        2: {
            "ref_hidden": torch.zeros(2, 2, 0),
            "ref_conv": torch.zeros(2, 0, 3),
            "prefix_hidden": torch.zeros(2, 0, 3),
            "ref_upsample": torch.zeros(2, 3, 0),
            "ref_wav": torch.zeros(2, 1, 0),
            "suffix_quantized": torch.ones(2, 2, 1),
            "suffix_conv": torch.ones(2, 1, 3),
            "past_key_values": SimpleNamespace(layers=[]),
        }
    }
    wrapper._record_graph_hit = lambda *_args: None
    wrapper._record_graph_fallback = lambda *_args: None
    wrapper._ensure_suffix_buffers = lambda cache: None
    wrapper.decoder = _decoder_stub(
        _is_suffix_cache_rolling=lambda previous, cached: False,
        _decode_xvec_first_chunk=lambda *_args: pytest.fail("unexpected eager fallback"),
        _slice_dynamic_cache=Qwen3TTSTokenizerV2Decoder._slice_dynamic_cache,
    )

    caches = [{"prefix_frames": 0}, {"prefix_frames": 0}]
    outputs = wrapper._batched_request_decode(
        [torch.full((1, 2, 1), 3), torch.full((1, 2, 1), 4)],
        caches,
    )

    assert replayed == [True]
    torch.testing.assert_close(wrapper.xvec_prefix_static_inputs[2][0], torch.full((2, 1), 3.0))
    torch.testing.assert_close(wrapper.xvec_prefix_static_inputs[2][1], torch.full((2, 1), 4.0))
    assert [cache["decoder_prefix_frames"] for cache in caches] == [0, 0]
    assert [cache["suffix_frames"] for cache in caches] == [1, 1]
    assert len(outputs) == 2


def test_xvec_prefixless_incremental_matches_full_decode_tail():
    torch.manual_seed(1)
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
        sliding_window=72,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    codes = torch.randint(0, config.codebook_size, (1, config.num_quantizers, 50))
    caches = {"prefix_frames": 0}

    with torch.no_grad():
        first = decoder(codes[..., :25], caches=caches)
        incremental = decoder(codes[..., 25:], caches=caches)
        full = decoder._forward_exact(codes)

    assert first.shape[-1] == 25 * decoder.total_upsample
    assert caches["decoder_prefix_frames"] == 0
    assert caches["suffix_frames"] == 50
    assert caches["past_key_values"].get_seq_length() == 0
    assert all(layer.is_initialized for layer in caches["past_key_values"].layers)
    torch.testing.assert_close(
        incremental,
        full[..., -25 * decoder.total_upsample :],
        atol=1e-5,
        rtol=1e-4,
    )


def test_single_frame_xvec_prefix_keeps_all_next_chunk_conv_frames():
    torch.manual_seed(4)
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
        sliding_window=72,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    decoder._incremental_chunk_frames = 25
    codes = torch.randint(0, config.codebook_size, (1, config.num_quantizers, 26))
    caches = {"prefix_frames": 0}

    with torch.no_grad():
        decoder(codes[..., :1], caches=caches)
        incremental = decoder(codes[..., 1:], caches=caches)
        full = decoder._forward_exact(codes)

    assert caches["suffix_frames"] == 26
    assert caches["suffix_quantized"].shape[-1] == 26
    assert caches["suffix_conv"].shape[1] == 26
    torch.testing.assert_close(
        incremental,
        full[..., -25 * decoder.total_upsample :],
        atol=1e-5,
        rtol=1e-4,
    )


def test_xvec_rolling_matches_truncated_suffix_window():
    torch.manual_seed(5)
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
        sliding_window=72,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    decoder._incremental_chunk_frames = 25
    codes = torch.randint(0, config.codebook_size, (1, config.num_quantizers, 101))
    caches = {"prefix_frames": 0}

    with torch.no_grad():
        decoder(codes[..., :1], caches=caches)
        decoder(codes[..., 1:26], caches=caches)
        decoder(codes[..., 26:51], caches=caches)
        decoder(codes[..., 51:76], caches=caches)
        rolling = decoder(codes[..., 76:101], caches=caches)
        truncated = decoder._forward_exact(codes[..., -(config.sliding_window + 25) :])

    assert caches["ref_hidden"].shape[-1] == 0
    assert caches["suffix_quantized"].shape[-1] == config.sliding_window
    assert caches["suffix_conv"].shape[1] == config.sliding_window - 2
    torch.testing.assert_close(
        rolling,
        truncated[..., -25 * decoder.total_upsample :],
        atol=1e-5,
        rtol=1e-4,
    )


def test_stateful_eager_batches_xvec_prefix_and_suffix():
    torch.manual_seed(7)
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
        sliding_window=72,
    )
    batched_decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    eager_decoder = copy.deepcopy(batched_decoder)
    batched_decoder._incremental_chunk_frames = 25
    eager_decoder._incremental_chunk_frames = 25
    codes = torch.randint(0, config.codebook_size, (2, config.num_quantizers, 26))
    batched_caches = [{"prefix_frames": 0}, {"prefix_frames": 0}]
    eager_caches = [{"prefix_frames": 0}, {"prefix_frames": 0}]

    with torch.no_grad():
        batched_prefix = batched_decoder.batched_chunked_decode(
            codes[..., :1],
            [1, 1],
            caches=batched_caches,
        )
        eager_prefix = torch.cat(
            [eager_decoder.chunked_decode(codes[row : row + 1, :, :1], caches=eager_caches[row]) for row in range(2)],
            dim=0,
        )
        batched_suffix = batched_decoder.batched_chunked_decode(
            codes[..., 1:],
            [25, 25],
            caches=batched_caches,
        )
        eager_suffix = torch.cat(
            [eager_decoder.chunked_decode(codes[row : row + 1, :, 1:], caches=eager_caches[row]) for row in range(2)],
            dim=0,
        )

    torch.testing.assert_close(batched_prefix, eager_prefix, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(batched_suffix, eager_suffix, atol=1e-5, rtol=1e-4)
    for batched_cache, eager_cache in zip(batched_caches, eager_caches, strict=True):
        assert batched_cache["suffix_frames"] == eager_cache["suffix_frames"] == 26
        torch.testing.assert_close(batched_cache["suffix_quantized"], eager_cache["suffix_quantized"])
        torch.testing.assert_close(batched_cache["suffix_conv"], eager_cache["suffix_conv"])


def test_stateful_eager_batches_variable_icl_prefixes():
    torch.manual_seed(11)
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
        sliding_window=72,
    )
    batched_decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    eager_decoder = copy.deepcopy(batched_decoder)
    batched_decoder._incremental_chunk_frames = 25
    eager_decoder._incremental_chunk_frames = 25
    lengths = [49, 73]
    codes = torch.zeros(2, config.num_quantizers, max(lengths), dtype=torch.long)
    for row, length in enumerate(lengths):
        codes[row, :, :length] = torch.randint(0, config.codebook_size, (config.num_quantizers, length))
    batched_caches = [{"prefix_frames": 48}, {"prefix_frames": 72}]
    eager_caches = [{"prefix_frames": 48}, {"prefix_frames": 72}]

    with torch.no_grad():
        batched_output = batched_decoder.batched_chunked_decode(codes, lengths, caches=batched_caches)
        eager_output = torch.cat(
            [
                eager_decoder.chunked_decode(
                    codes[row : row + 1, :, :length],
                    caches=eager_caches[row],
                )
                for row, length in enumerate(lengths)
            ],
            dim=0,
        )
        suffix_codes = torch.randint(0, config.codebook_size, (2, config.num_quantizers, 25))
        batched_suffix = batched_decoder.batched_chunked_decode(
            suffix_codes,
            [25, 25],
            caches=batched_caches,
        )
        eager_suffix = torch.cat(
            [
                eager_decoder.chunked_decode(
                    suffix_codes[row : row + 1],
                    caches=eager_caches[row],
                )
                for row in range(2)
            ],
            dim=0,
        )

    torch.testing.assert_close(batched_output, eager_output, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(batched_suffix, eager_suffix, atol=1e-5, rtol=1e-4)
    for batched_cache in batched_caches:
        assert batched_cache["decoder_prefix_frames"] == 72
        assert batched_cache["suffix_frames"] == 26
