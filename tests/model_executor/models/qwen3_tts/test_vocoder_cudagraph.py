# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    VocoderCUDAGraphDescriptor,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.vocoder_cudagraph import (
    Qwen3TTSIclPrefixRoutine,
    Qwen3TTSIclPrefixVariant,
    Qwen3TTSStatelessRoutine,
    Qwen3TTSStatelessVariant,
    Qwen3TTSSuffixVariant,
    Qwen3TTSXvecPrefixRoutine,
    Qwen3TTSXvecPrefixVariant,
    build_stateful_targets,
    build_stateless_target,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_quantizers=2)
        self.total_upsample = 4

    def _forward_exact(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.float().sum(dim=1, keepdim=True)
        return values.repeat_interleave(self.total_upsample, dim=-1)


class _StatefulDecoder(_Decoder):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("_dtype_probe", torch.nn.Parameter(torch.zeros(1)))
        self.config = Qwen3TTSTokenizerV2DecoderConfig(
            codebook_size=32,
            hidden_size=4,
            latent_dim=2,
            codebook_dim=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=8,
            num_hidden_layers=1,
            num_quantizers=2,
            decoder_dim=4,
            upsample_rates=(2,),
            upsampling_ratios=(2,),
            sliding_window=10,
        )

    def _decode_icl_first_chunk(self, codes, cache, prefix_frames, **kwargs):
        del prefix_frames
        cache.update(
            {
                "ref_hidden": codes[:, :1, :],
                "ref_conv": codes[:, :1, :].transpose(1, 2),
                "prefix_hidden": codes[:, :1, :].transpose(1, 2),
                "suffix_quantized": codes[:, :1, :],
                "suffix_conv": codes[:, :1, :].transpose(1, 2),
                "past_key_values": kwargs.get("prefix_cache"),
            }
        )
        return codes.float()

    def _decode_xvec_first_chunk(self, codes, cache):
        cache.update(
            {
                "ref_hidden": codes[:, :1, :0],
                "ref_conv": codes[:, :1, :0].transpose(1, 2),
                "prefix_hidden": codes[:, :1, :0].transpose(1, 2),
                "suffix_quantized": codes[:, :1, :],
                "suffix_conv": codes[:, :1, :].transpose(1, 2),
                "past_key_values": cache.get("past_key_values"),
            }
        )
        return codes.float()

    def _decode_suffix(self, *args, **kwargs):
        del kwargs
        codes = args[2]
        return codes.float(), codes.float(), codes.float()


def _config(*, async_chunk: bool, graph_config=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            async_chunk=async_chunk,
            vocoder_cudagraph_config=graph_config,
        )
    )


def test_stateless_capture_sizes_extend_defaults_with_configured_sizes() -> None:
    target = build_stateless_target(
        decoder=_Decoder(),
        runnable=_Decoder()._forward_exact,
        vllm_config=_config(
            async_chunk=False,
            graph_config={
                "capture_batch_sizes": [1, 3],
                "targets": {
                    "qwen3_tts.stateless": {
                        "capture_bucket_sizes": [17, 29],
                    }
                },
            },
        ),
        num_quantizers=2,
        total_upsample=4,
        capture_batch_sizes=[8],
        decode_chunk_size=300,
        decode_left_context=25,
    )

    assert {descriptor.variant for descriptor in target.descriptors} == {
        Qwen3TTSStatelessVariant(1, 17),
        Qwen3TTSStatelessVariant(1, 29),
        Qwen3TTSStatelessVariant(1, 150),
        Qwen3TTSStatelessVariant(1, 325),
        Qwen3TTSStatelessVariant(3, 17),
        Qwen3TTSStatelessVariant(3, 29),
        Qwen3TTSStatelessVariant(3, 150),
        Qwen3TTSStatelessVariant(3, 325),
    }


def test_qwen3_tts_stateless_target_keeps_default_capture_buckets() -> None:
    target = build_stateless_target(
        decoder=_Decoder(),
        runnable=_Decoder()._forward_exact,
        vllm_config=_config(async_chunk=False),
        num_quantizers=2,
        total_upsample=4,
        capture_batch_sizes=[1],
        decode_chunk_size=300,
        decode_left_context=25,
    )

    assert {descriptor.variant.frames for descriptor in target.descriptors} == {150, 325}


def test_qwen3_tts_async_chunk_keeps_known_stateless_target_without_startup_coverage() -> None:
    target = build_stateless_target(
        decoder=_Decoder(),
        runnable=_Decoder()._forward_exact,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        total_upsample=4,
        capture_batch_sizes=[1],
        decode_chunk_size=300,
        decode_left_context=25,
    )

    assert target.target_id == "qwen3_tts.stateless"
    assert target.descriptors == ()
    codes = torch.ones(1, 2, 3, dtype=torch.long)
    assert torch.equal(target(codes), _Decoder()._forward_exact(codes))


def test_qwen3_tts_runtime_resolution_uses_smallest_safe_padding_bucket() -> None:
    decoder = _Decoder()
    routine = Qwen3TTSStatelessRoutine(
        decoder=decoder,
        runnable=decoder._forward_exact,
        num_quantizers=2,
        total_upsample=4,
    )
    target = build_stateless_target(
        decoder=decoder,
        runnable=decoder._forward_exact,
        vllm_config=_config(
            async_chunk=False,
            graph_config={
                "capture_batch_sizes": [1, 2],
                "targets": {"qwen3_tts.stateless": {"capture_bucket_sizes": [10]}},
            },
        ),
        num_quantizers=2,
        total_upsample=4,
        capture_batch_sizes=None,
        decode_chunk_size=300,
        decode_left_context=25,
    )
    # Both fit [B=1,F=8], but (1, 10) has the smaller padded area.
    available = set(target.descriptors)
    codes = torch.ones(1, 2, 8, dtype=torch.long)
    resolution = routine.resolve_runtime((codes,), {}, available)
    assert resolution.descriptor is not None
    assert resolution.descriptor.variant == Qwen3TTSStatelessVariant(1, 10)


def test_stateless_replay_pads_to_bucket_and_trims_output() -> None:
    decoder = _Decoder()
    routine = Qwen3TTSStatelessRoutine(
        decoder=decoder,
        runnable=decoder._forward_exact,
        num_quantizers=2,
        total_upsample=4,
    )
    descriptor = VocoderCUDAGraphDescriptor(Qwen3TTSStatelessVariant(1, 4))
    buffers = routine.allocate_buffers(descriptor, torch.device("cpu"))
    buffers.codes.fill_(-1)
    codes = torch.tensor([[[1, 2], [3, 4]]])
    captured_output = torch.arange(16, dtype=torch.float32).view(1, 1, 16)
    routine.copy_runtime_inputs((codes,), {}, buffers)
    output = routine.output_after_replay((codes,), {}, buffers, captured_output)

    torch.testing.assert_close(buffers.codes, torch.tensor([[[1, 2, 0, 0], [3, 4, 0, 0]]]))
    torch.testing.assert_close(output, captured_output[..., :8])


def test_icl_prefix_graph_cache_keeps_physical_prefix_length() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[2],
        initial_frames=2,
        chunk_frames=3,
        chunk_ramp=None,
    )
    target = targets[0]
    routine = target.routine
    assert isinstance(routine, Qwen3TTSIclPrefixRoutine)
    descriptor = target.descriptors[0]
    buffers = routine.allocate_buffers(descriptor, torch.device("cpu"))
    routine.prepare_for_capture(buffers)
    captured_output = routine.forward_for_capture(buffers)
    runtime_codes = torch.ones(1, 2, 7, dtype=torch.long)
    runtime_cache: dict[str, object] = {}
    hidden_mask = torch.ones(1, 1, 12)
    prefix_mask = torch.ones(1, 10, dtype=torch.bool)
    suffix_mask = torch.ones(1, 12, dtype=torch.bool)
    routine.copy_runtime_inputs(
        (runtime_codes, runtime_cache),
        {
            "prefix_hidden_mask": hidden_mask,
            "prefix_attention_mask": prefix_mask,
            "suffix_attention_mask": suffix_mask,
        },
        buffers,
    )
    output, state = routine.output_after_replay((runtime_codes, runtime_cache), {}, buffers, captured_output)

    assert buffers.codes.shape == (2, 2, 12)
    torch.testing.assert_close(buffers.codes[0, :, :7], runtime_codes[0])
    torch.testing.assert_close(buffers.codes[0, :, 7:], torch.zeros(2, 5, dtype=torch.long))
    assert runtime_cache == {}
    assert state["decoder_prefix_frames"] == 10
    assert state["past_key_values"].get_seq_length() == 0
    assert output.shape == (1, 2, 12)


def test_qwen3_tts_icl_capture_delegates_through_eager_call() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[1],
        initial_frames=2,
        chunk_frames=3,
        chunk_ramp=None,
    )
    routine = targets[0].routine
    assert isinstance(routine, Qwen3TTSIclPrefixRoutine)
    buffers = routine.allocate_buffers(targets[0].descriptors[0], torch.device("cpu"))
    routine.prepare_for_capture(buffers)
    prepared_prefix_cache = buffers.prefix_cache
    assert prepared_prefix_cache is not None

    original_eager_call = routine.eager_call
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def recording_eager_call(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        return original_eager_call(*args, **kwargs)  # type: ignore[arg-type]

    setattr(routine, "eager_call", recording_eager_call)
    routine.forward_for_capture(buffers)

    assert len(calls) == 1
    assert calls[0][1]["prefix_cache"] is prepared_prefix_cache


def test_xvec_dummy_cache_is_initialized_before_graph_capture() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[2],
        initial_frames=2,
        chunk_frames=3,
        chunk_ramp=None,
    )
    routine = targets[1].routine
    assert isinstance(routine, Qwen3TTSXvecPrefixRoutine)
    buffers = routine.allocate_buffers(targets[1].descriptors[0], torch.device("cpu"))
    routine.prepare_for_capture(buffers)

    cache = buffers.cache["past_key_values"]
    assert cache.get_seq_length() == 0
    assert all(layer.is_initialized for layer in cache.layers)
    assert all(layer.keys.device.type == "cpu" for layer in cache.layers)


def test_xvec_first_chunk_replays_prefix_graph() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[2],
        initial_frames=2,
        chunk_frames=3,
        chunk_ramp=None,
    )
    target = targets[1]
    routine = target.routine
    assert isinstance(routine, Qwen3TTSXvecPrefixRoutine)
    descriptor = target.descriptors[0]
    buffers = routine.allocate_buffers(descriptor, torch.device("cpu"))
    routine.prepare_for_capture(buffers)
    captured_output = routine.forward_for_capture(buffers)

    runtime_codes = torch.ones(1, 2, 2, dtype=torch.long)
    runtime_cache: dict[str, object] = {}
    routine.copy_runtime_inputs((runtime_codes, runtime_cache), {}, buffers)
    output, state = routine.output_after_replay((runtime_codes, runtime_cache), {}, buffers, captured_output)

    assert runtime_cache == {}
    assert state["decoder_prefix_frames"] == 0
    assert state["past_key_values"].get_seq_length() == 0
    assert output.shape == (1, 2, 2)


def test_qwen3_tts_suffix_routine_uses_physical_cache_lengths() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[1],
        initial_frames=2,
        chunk_frames=3,
        chunk_ramp=None,
    )
    target = targets[2]
    descriptor = next(
        descriptor for descriptor in target.descriptors if descriptor.variant == Qwen3TTSSuffixVariant("icl", 1, 11)
    )
    buffers = target.routine.allocate_buffers(descriptor, torch.device("cpu"))

    assert buffers.quantized.shape == (1, 3, 8)
    assert buffers.conv.shape == (1, 8, 2)


def test_capture_uses_cached_conv_lengths() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[1],
        initial_frames=1,
        chunk_frames=3,
        chunk_ramp=None,
    )
    suffix_target = targets[2]
    expected_by_target = {
        4: (1, 1),
        7: (4, 4),
        10: (7, 7),
        13: (10, 8),
    }

    for target_frames, (expected_quantized, expected_conv) in expected_by_target.items():
        descriptor = next(
            item
            for item in suffix_target.descriptors
            if item.variant == Qwen3TTSSuffixVariant("xvec", 1, target_frames)
        )
        buffers = suffix_target.routine.allocate_buffers(descriptor, torch.device("cpu"))
        assert buffers.quantized.shape[-1] == expected_quantized
        assert buffers.conv.shape[1] == expected_conv


def test_suffix_routine_copies_request_state_without_batched_cat(monkeypatch) -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[2],
        initial_frames=2,
        chunk_frames=3,
        chunk_ramp=None,
    )
    suffix_target = targets[2]
    descriptor = next(item for item in suffix_target.descriptors if item.variant == Qwen3TTSSuffixVariant("xvec", 2, 5))
    routine = suffix_target.routine
    buffers = routine.allocate_buffers(descriptor, torch.device("cpu"))
    dtype = next(decoder.parameters()).dtype
    caches = [routine._make_dummy_cache("xvec", 1, torch.device("cpu"), dtype) for _ in range(2)]
    for row, cache in enumerate(caches):
        cache["suffix_quantized"] = torch.full((1, 3, 2), float(row + 1))
        cache["suffix_conv"] = torch.full((1, 2, 2), float(row + 1))
        cache["suffix_frames"] = 2
        cache["decoder_prefix_frames"] = 0
    codes_list = [torch.full((1, 2, 3), row + 1, dtype=torch.long) for row in range(2)]

    monkeypatch.setattr(torch, "cat", lambda *_args, **_kwargs: pytest.fail("runtime suffix copy must not batch-cat"))
    routine.copy_runtime_inputs(("xvec", 5, codes_list, caches, [3, 3]), {}, buffers)

    torch.testing.assert_close(buffers.codes[0], codes_list[0][0])
    torch.testing.assert_close(buffers.codes[1], codes_list[1][0])
    torch.testing.assert_close(buffers.quantized[0], caches[0]["suffix_quantized"][0])
    torch.testing.assert_close(buffers.quantized[1], caches[1]["suffix_quantized"][0])


def test_suffix_output_after_replay_does_not_mutate_request_cache() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[1],
        initial_frames=2,
        chunk_frames=3,
        chunk_ramp=None,
    )
    routine = targets[2].routine
    cache = {
        "suffix_frames": 2,
        "suffix_quantized": torch.ones(1, 3, 2),
        "suffix_conv": torch.ones(1, 2, 2),
    }
    quantized_before = cache["suffix_quantized"].clone()
    conv_before = cache["suffix_conv"].clone()
    codes = torch.zeros(1, 2, 3)
    captured_output = (
        torch.full((1, 1, 12), 1.0),
        torch.full((1, 3, 5), 2.0),
        torch.full((1, 3, 2), 3.0),
    )

    output = routine.output_after_replay(
        ("xvec", 5, [codes], [cache], [3]),
        {},
        object(),
        captured_output,
    )

    for actual, expected in zip(output, captured_output, strict=True):
        torch.testing.assert_close(actual, expected)
    assert cache["suffix_frames"] == 2
    torch.testing.assert_close(cache["suffix_quantized"], quantized_before)
    torch.testing.assert_close(cache["suffix_conv"], conv_before)


def test_qwen3_tts_stateful_targets_are_config_derived_and_target_local() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True, graph_config={"capture_batch_sizes": [1, 2]}),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=None,
        initial_frames=2,
        chunk_frames=3,
        chunk_ramp=None,
    )

    assert [target.target_id for target in targets] == [
        "qwen3_tts.icl_prefix",
        "qwen3_tts.xvec_prefix",
        "qwen3_tts.suffix",
    ]
    assert {descriptor.variant for descriptor in targets[0].descriptors} == {
        Qwen3TTSIclPrefixVariant(1, 12),
        Qwen3TTSIclPrefixVariant(2, 12),
    }
    assert {descriptor.variant for descriptor in targets[1].descriptors} == {
        Qwen3TTSXvecPrefixVariant(1, 2),
        Qwen3TTSXvecPrefixVariant(2, 2),
    }
    suffix_variants = {descriptor.variant for descriptor in targets[2].descriptors}
    assert Qwen3TTSSuffixVariant("icl", 1, 5) in suffix_variants
    assert Qwen3TTSSuffixVariant("xvec", 2, 13) in suffix_variants


def test_segmented_capture_sizes_are_derived_from_decoder_and_chunk_config() -> None:
    decoder = _StatefulDecoder()
    targets = build_stateful_targets(
        decoder=decoder,
        vllm_config=_config(async_chunk=True),
        num_quantizers=2,
        prefix_length=10,
        capture_batch_sizes=[1],
        initial_frames=2,
        chunk_frames=4,
        chunk_ramp=None,
    )
    suffix_target = targets[2]
    captured_targets = {
        item.variant.target_frames
        for item in suffix_target.descriptors
        if item.variant.mode == "icl" and item.variant.batch_size == 1
    }

    assert captured_targets == {6, 10, 14}


def test_target_owns_descriptor_namespace_and_binding_is_reversible() -> None:
    from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
        VocoderCUDAGraphTarget,
        VocoderGraphHandle,
    )

    routine = Qwen3TTSStatelessRoutine(
        decoder=_Decoder(),
        runnable=_Decoder()._forward_exact,
        num_quantizers=2,
        total_upsample=4,
    )
    target = VocoderCUDAGraphTarget(
        "qwen3_tts.stateless",
        routine,
        [VocoderCUDAGraphDescriptor(Qwen3TTSStatelessVariant(1, 4))],
    )
    handle = VocoderGraphHandle(lambda codes: codes + 1)
    target._bind_handle(handle)
    assert torch.equal(target(torch.zeros(1)), torch.ones(1))
    with pytest.raises(TypeError, match="VocoderGraphHandle"):
        target._bind_handle(lambda codes: codes)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="already bound"):
        target._bind_handle(handle)
    target._restore_eager()
