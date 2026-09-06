# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Runner-owned decoder Target for Qwen3-TTS Code2Wav."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    BaseVocoderCUDAGraphRoutine,
    VocoderCUDAGraphDescriptor,
    VocoderCUDAGraphTarget,
    VocoderRuntimeKey,
    VocoderRuntimeResolution,
)
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import parse_chunk_ramp

from .stateful_chunking import resolve_stateful_chunk_contract
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    _materialize_suffix_batch,
    _resolve_suffix_batch,
)

STATELESS_TARGET_ID = "qwen3_tts.stateless"
SHARED_CONFIG_KEYS = frozenset(
    {
        "capture_batch_sizes",
        "decode_batch_max_size",
        "decode_enable_tf32",
    }
)
TARGET_CONFIG_KEYS = frozenset({"capture_bucket_sizes", "decode_chunk_frames", "decode_left_context_frames"})


@dataclass(frozen=True)
class Qwen3TTSStatelessVariant:
    batch_size: int
    frames: int


@dataclass(frozen=True)
class Qwen3TTSExecutionSettings:
    """Resolved Qwen decode and capture settings shared by all owners."""

    async_chunk: bool
    capture_batch_sizes: tuple[int, ...] | None
    stateless_capture_sizes: tuple[int, ...]
    decode_chunk_frames: int
    decode_left_context_frames: int
    decode_batch_max_size: int
    decode_enable_tf32: bool
    codec_chunk_frames: int
    codec_left_context_frames: int
    initial_codec_chunk_frames: int
    codec_chunk_ramp: tuple[int, ...]


@dataclass
class Qwen3TTSStatelessBuffers:
    codes: torch.Tensor


class Qwen3TTSStatelessRoutine(BaseVocoderCUDAGraphRoutine):
    def __init__(
        self,
        *,
        decoder: torch.nn.Module,
        runnable: Callable[[torch.Tensor], torch.Tensor],
        num_quantizers: int,
        total_upsample: int,
    ) -> None:
        self.decoder = decoder
        self.num_quantizers = num_quantizers
        self.total_upsample = total_upsample
        self._runnable = runnable

    def eager_call(self, codes: torch.Tensor) -> torch.Tensor:
        return self._runnable(codes)

    def validate_runtime_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if kwargs or len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise TypeError("Qwen3-TTS stateless decoder expects exactly one Tensor argument")
        codes = args[0]
        if codes.ndim != 3:
            raise ValueError(f"Expected Qwen3-TTS codes [B, Q, F], got {tuple(codes.shape)}")
        if int(codes.shape[1]) != self.num_quantizers:
            raise ValueError(f"Expected {self.num_quantizers} codebooks, got {codes.shape[1]}")

    def resolve_runtime(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        available: Set[VocoderCUDAGraphDescriptor],
    ) -> VocoderRuntimeResolution:
        del kwargs
        codes = args[0]
        runtime_variant = Qwen3TTSStatelessVariant(
            batch_size=int(codes.shape[0]),
            frames=int(codes.shape[-1]),
        )
        candidates = [
            descriptor
            for descriptor in available
            if isinstance(descriptor.variant, Qwen3TTSStatelessVariant)
            and descriptor.variant.batch_size >= runtime_variant.batch_size
            and descriptor.variant.frames >= runtime_variant.frames
        ]
        descriptor = min(
            candidates,
            key=lambda item: (
                item.variant.batch_size * item.variant.frames,
                item.variant.batch_size,
                item.variant.frames,
            ),
            default=None,
        )
        return VocoderRuntimeResolution(
            runtime_key=VocoderRuntimeKey(runtime_variant),
            descriptor=descriptor,
        )

    def allocate_buffers(
        self,
        descriptor: VocoderCUDAGraphDescriptor,
        device: torch.device,
    ) -> Qwen3TTSStatelessBuffers:
        variant = descriptor.variant
        if not isinstance(variant, Qwen3TTSStatelessVariant):
            raise TypeError(f"Unexpected Qwen3-TTS stateless Descriptor: {descriptor!r}")
        return Qwen3TTSStatelessBuffers(
            codes=torch.zeros(
                variant.batch_size,
                self.num_quantizers,
                variant.frames,
                dtype=torch.long,
                device=device,
            )
        )

    def forward_for_capture(self, buffers: object) -> torch.Tensor:
        if not isinstance(buffers, Qwen3TTSStatelessBuffers):
            raise TypeError("Unexpected Qwen3-TTS stateless capture buffers")
        return self.eager_call(buffers.codes)

    def copy_runtime_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
    ) -> None:
        del kwargs
        if not isinstance(buffers, Qwen3TTSStatelessBuffers):
            raise TypeError("Unexpected Qwen3-TTS stateless replay buffers")
        codes = args[0]
        buffers.codes.zero_()
        buffers.codes[: codes.shape[0], :, : codes.shape[-1]].copy_(codes)

    def output_after_replay(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
        captured_output: object,
    ) -> torch.Tensor:
        del kwargs, buffers
        codes = args[0]
        output = captured_output
        if not isinstance(output, torch.Tensor):
            raise TypeError("Qwen3-TTS stateless graph output must be a Tensor")
        # Runtime args only recover logical extent from descriptor-sized output.
        return output[
            : codes.shape[0],
            ...,
            : codes.shape[-1] * self.total_upsample,
        ]


@dataclass(frozen=True)
class Qwen3TTSIclPrefixVariant:
    batch_size: int
    total_frames: int


@dataclass(frozen=True)
class Qwen3TTSXvecPrefixVariant:
    batch_size: int
    frames: int


@dataclass(frozen=True)
class Qwen3TTSSuffixVariant:
    mode: str
    batch_size: int
    target_frames: int


@dataclass
class _IclPrefixBuffers:
    codes: torch.Tensor
    hidden_mask: torch.Tensor
    prefix_mask: torch.Tensor
    suffix_mask: torch.Tensor
    cache: dict[str, Any]
    prefix_cache: DynamicCache | None = None


@dataclass
class _XvecPrefixBuffers:
    codes: torch.Tensor
    cache: dict[str, Any]


@dataclass
class _SuffixBuffers:
    codes: torch.Tensor
    quantized: torch.Tensor
    conv: torch.Tensor
    mask: torch.Tensor
    cache: dict[str, Any]
    mode: str
    target_frames: int


def _slice_dynamic_cache_batch(cache: DynamicCache, batch_size: int) -> DynamicCache:
    """Clone graph-owned static cache into detached result data.

    The caller/model may later commit this data into request-owned state; this
    helper itself never receives or mutates a request cache.
    """
    result = copy.deepcopy(cache)
    for layer in result.layers:
        if layer.keys is not None:
            layer.keys = layer.keys[:batch_size].clone()
        if layer.values is not None:
            layer.values = layer.values[:batch_size].clone()
    return result


def _slice_dynamic_cache_view(cache: DynamicCache, row: int) -> DynamicCache:
    result = copy.copy(cache)
    result.layers = [copy.copy(layer) for layer in cache.layers]
    for layer in result.layers:
        if layer.keys is not None:
            layer.keys = layer.keys[row : row + 1]
        if layer.values is not None:
            layer.values = layer.values[row : row + 1]
    return result


def _copy_dynamic_cache_row(source: DynamicCache, target: DynamicCache, row: int) -> None:
    """Copy request KV contents into one graph-owned static cache row.

    This is replay-input preparation only and must not commit replay output
    into request-owned decoder state.
    """
    for source_layer, target_layer in zip(source.layers, target.layers, strict=True):
        if source_layer.keys is None or source_layer.values is None:
            if target_layer.keys is not None and target_layer.keys.numel() != 0:
                raise RuntimeError("Uninitialized request KV cache cannot populate a non-empty graph cache")
            continue
        if target_layer.keys is None or target_layer.values is None:
            raise RuntimeError("Graph KV cache is not initialized")
        target_layer.keys[row : row + 1].copy_(source_layer.keys)
        target_layer.values[row : row + 1].copy_(source_layer.values)


def _materialize_cache_bookkeeping(
    cache: DynamicCache,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
    batch_size: int | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
    num_layers: int | None = None,
) -> None:
    """Materialize graph-owned DynamicCache metadata before capture.

    This prepares dummy/static state required by the captured callable and
    does not represent or advance request-level decoder state.
    """

    for layer in cache.layers:
        if (
            not getattr(layer, "is_initialized", True)
            and batch_size is not None
            and num_heads is not None
            and head_dim is not None
            and dtype is not None
        ):
            fake_kv = torch.empty(
                batch_size,
                num_heads,
                0,
                head_dim,
                dtype=dtype,
                device=device,
            )
            layer.lazy_initialization(fake_kv, fake_kv)
        sliding_window = getattr(layer, "_sliding_window_tensor", None)
        if sliding_window is not None and sliding_window.device != device:
            layer._sliding_window_tensor = sliding_window.to(device)
    if (
        batch_size is not None
        and num_heads is not None
        and head_dim is not None
        and dtype is not None
        and num_layers is not None
    ):
        # Some Transformers configurations leave DynamicCache.layers lazy even
        # after early_initialization(). Zero-length updates force creation of
        # every configured layer outside capture without adding logical KV.
        fake_kv = torch.empty(batch_size, num_heads, 0, head_dim, dtype=dtype, device=device)
        for layer_index in range(num_layers):
            cache.update(fake_kv, fake_kv, layer_index)


class _Qwen3TTSStatefulRoutineBase(BaseVocoderCUDAGraphRoutine):
    def __init__(self, *, decoder: torch.nn.Module, num_quantizers: int) -> None:
        self.decoder = decoder
        self.num_quantizers = num_quantizers

    def validate_runtime_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if not args or not isinstance(args[0], torch.Tensor) or args[0].ndim != 3:
            raise ValueError(f"{type(self).__name__} expects batched codes with shape [B,Q,F]")
        if int(args[0].shape[1]) != self.num_quantizers:
            raise ValueError(f"Expected {self.num_quantizers} codebooks, got {args[0].shape[1]}")

    def make_lazy_descriptor(self, runtime_key: VocoderRuntimeKey) -> VocoderCUDAGraphDescriptor | None:
        # Stateful shape domains are derived from connector transitions. Do not
        # invent a new stateful graph specialization at runtime by default.
        del runtime_key
        return None

    def prepare_for_capture(self, buffers: object) -> None:
        del buffers


class Qwen3TTSIclPrefixRoutine(_Qwen3TTSStatefulRoutineBase):
    def __init__(
        self, *, decoder: torch.nn.Module, num_quantizers: int, prefix_length: int, initial_frames: int
    ) -> None:
        super().__init__(decoder=decoder, num_quantizers=num_quantizers)
        self.prefix_length = prefix_length
        self.initial_frames = initial_frames
        self._runnable = decoder._decode_icl_first_chunk

    def eager_call(
        self,
        codes: torch.Tensor,
        cache: dict[str, Any],
        prefix_frames: int,
        *,
        prefix_cache: DynamicCache | None = None,
        prefix_hidden_mask: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        suffix_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._runnable(
            codes,
            cache,
            prefix_frames,
            prefix_cache=prefix_cache,
            prefix_hidden_mask=prefix_hidden_mask,
            prefix_attention_mask=prefix_attention_mask,
            suffix_attention_mask=suffix_attention_mask,
        )

    def resolve_runtime(self, args, kwargs, available: Set[VocoderCUDAGraphDescriptor]) -> VocoderRuntimeResolution:
        del kwargs
        codes = args[0]
        variant = Qwen3TTSIclPrefixVariant(int(codes.shape[0]), int(codes.shape[-1]))
        candidates = [
            d
            for d in available
            if isinstance(d.variant, Qwen3TTSIclPrefixVariant)
            and d.variant.batch_size >= variant.batch_size
            and d.variant.total_frames >= variant.total_frames
        ]
        descriptor = min(
            candidates,
            key=lambda d: (d.variant.batch_size * d.variant.total_frames, d.variant.batch_size),
            default=None,
        )
        return VocoderRuntimeResolution(VocoderRuntimeKey(variant), descriptor)

    def allocate_buffers(self, descriptor, device):
        variant = descriptor.variant
        if not isinstance(variant, Qwen3TTSIclPrefixVariant):
            raise TypeError(f"Unexpected ICL Descriptor: {descriptor!r}")
        dtype = next(self.decoder.parameters()).dtype
        return _IclPrefixBuffers(
            codes=torch.zeros(
                variant.batch_size, self.num_quantizers, variant.total_frames, dtype=torch.long, device=device
            ),
            hidden_mask=torch.ones(variant.batch_size, 1, variant.total_frames, dtype=dtype, device=device),
            prefix_mask=torch.ones(variant.batch_size, self.prefix_length, dtype=torch.bool, device=device),
            suffix_mask=torch.ones(variant.batch_size, variant.total_frames, dtype=torch.bool, device=device),
            cache={},
        )

    def prepare_for_capture(self, buffers: object) -> None:
        if not isinstance(buffers, _IclPrefixBuffers):
            raise TypeError("Unexpected ICL capture buffers")
        config = self.decoder.config
        dtype = next(self.decoder.parameters()).dtype
        prefix_cache = DynamicCache(config=config)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        prefix_cache.early_initialization(
            batch_size=buffers.codes.shape[0],
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=buffers.codes.device,
        )
        # Match the legacy ICL capture path: prefix_cache is empty but fully
        # initialized. Do not populate logical prefix KV here; the prefix
        # forward writes it exactly once during warmup/capture.
        _materialize_cache_bookkeeping(
            prefix_cache,
            buffers.codes.device,
            dtype=dtype,
            batch_size=buffers.codes.shape[0],
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            num_layers=config.num_hidden_layers,
        )
        buffers.prefix_cache = prefix_cache
        buffers.cache = {}

    def forward_for_capture(self, buffers: object) -> torch.Tensor:
        if not isinstance(buffers, _IclPrefixBuffers) or buffers.prefix_cache is None:
            raise TypeError("ICL capture buffers are not prepared")
        # Capture uses the pre-materialized empty prefix cache, matching the
        # legacy _capture_icl_prefix path. Runtime eager calls may omit this
        # optional argument, in which case the model creates its ordinary
        # request cache.
        return self.eager_call(
            buffers.codes,
            buffers.cache,
            self.prefix_length,
            prefix_cache=buffers.prefix_cache,
            prefix_hidden_mask=buffers.hidden_mask,
            prefix_attention_mask=buffers.prefix_mask,
            suffix_attention_mask=buffers.suffix_mask,
        )

    def copy_runtime_inputs(self, args, kwargs, buffers: object) -> None:
        if not isinstance(buffers, _IclPrefixBuffers):
            raise TypeError("Unexpected ICL replay buffers")
        codes, runtime_cache = args[0], args[1]
        buffers.codes.zero_()
        buffers.codes[: codes.shape[0], :, : codes.shape[-1]].copy_(codes)
        buffers.hidden_mask.fill_(1)
        buffers.prefix_mask.fill_(1)
        buffers.suffix_mask.fill_(1)
        for name, source in (
            ("prefix_hidden_mask", buffers.hidden_mask),
            ("prefix_attention_mask", buffers.prefix_mask),
            ("suffix_attention_mask", buffers.suffix_mask),
        ):
            value = kwargs.get(name)
            if value is not None:
                source[: value.shape[0], ...].copy_(value)
        del runtime_cache

    def output_after_replay(
        self, args, kwargs, buffers: object, captured_output: object
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs
        codes = args[0]
        static_cache = buffers.cache
        state = {
            key: static_cache[key][: codes.shape[0]]
            for key in ("ref_hidden", "ref_conv", "prefix_hidden", "suffix_quantized", "suffix_conv")
        }
        detached = _slice_dynamic_cache_batch(static_cache["past_key_values"], int(codes.shape[0]))
        state["past_key_values"] = [_slice_dynamic_cache_view(detached, row) for row in range(int(codes.shape[0]))]
        state["decoder_prefix_frames"] = self.prefix_length
        if not isinstance(captured_output, torch.Tensor):
            raise TypeError("Qwen3-TTS ICL graph output must be a Tensor")
        return captured_output[: codes.shape[0]], state


class Qwen3TTSXvecPrefixRoutine(_Qwen3TTSStatefulRoutineBase):
    def __init__(self, *, decoder: torch.nn.Module, num_quantizers: int, initial_frames: int) -> None:
        super().__init__(decoder=decoder, num_quantizers=num_quantizers)
        self.initial_frames = initial_frames
        self._runnable = decoder._decode_xvec_first_chunk

    def eager_call(self, codes: torch.Tensor, cache: dict[str, Any]) -> torch.Tensor:
        return self._runnable(codes, cache)

    def resolve_runtime(self, args, kwargs, available: Set[VocoderCUDAGraphDescriptor]) -> VocoderRuntimeResolution:
        del kwargs
        codes = args[0]
        variant = Qwen3TTSXvecPrefixVariant(int(codes.shape[0]), int(codes.shape[-1]))
        candidates = [
            d
            for d in available
            if isinstance(d.variant, Qwen3TTSXvecPrefixVariant)
            and d.variant.batch_size >= variant.batch_size
            and d.variant.frames >= variant.frames
        ]
        descriptor = min(
            candidates, key=lambda d: (d.variant.batch_size * d.variant.frames, d.variant.batch_size), default=None
        )
        return VocoderRuntimeResolution(VocoderRuntimeKey(variant), descriptor)

    def allocate_buffers(self, descriptor, device):
        variant = descriptor.variant
        if not isinstance(variant, Qwen3TTSXvecPrefixVariant):
            raise TypeError(f"Unexpected xvec Descriptor: {descriptor!r}")
        return _XvecPrefixBuffers(
            codes=torch.zeros(variant.batch_size, self.num_quantizers, variant.frames, dtype=torch.long, device=device),
            cache={},
        )

    def prepare_for_capture(self, buffers: object) -> None:
        if not isinstance(buffers, _XvecPrefixBuffers):
            raise TypeError("Unexpected xvec capture buffers")
        config = self.decoder.config
        dtype = next(self.decoder.parameters()).dtype
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        prefix_cache = DynamicCache(config=config)
        prefix_cache.early_initialization(
            batch_size=buffers.codes.shape[0],
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=buffers.codes.device,
        )
        _materialize_cache_bookkeeping(
            prefix_cache,
            buffers.codes.device,
            dtype=dtype,
            batch_size=buffers.codes.shape[0],
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            num_layers=config.num_hidden_layers,
        )
        buffers.cache = {"past_key_values": prefix_cache}

    def forward_for_capture(self, buffers: object) -> torch.Tensor:
        if not isinstance(buffers, _XvecPrefixBuffers):
            raise TypeError("Unexpected xvec capture buffers")
        return self.eager_call(buffers.codes, buffers.cache)

    def copy_runtime_inputs(self, args, kwargs, buffers: object) -> None:
        del kwargs
        if not isinstance(buffers, _XvecPrefixBuffers):
            raise TypeError("Unexpected xvec replay buffers")
        codes = args[0]
        buffers.codes.zero_()
        buffers.codes[: codes.shape[0], :, : codes.shape[-1]].copy_(codes)

    def output_after_replay(
        self, args, kwargs, buffers: object, captured_output: object
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs
        codes = args[0]
        static_cache = buffers.cache
        state = {
            key: static_cache[key][: codes.shape[0]]
            for key in ("ref_hidden", "ref_conv", "prefix_hidden", "suffix_quantized", "suffix_conv")
        }
        detached = _slice_dynamic_cache_batch(static_cache["past_key_values"], int(codes.shape[0]))
        state["past_key_values"] = [_slice_dynamic_cache_view(detached, row) for row in range(int(codes.shape[0]))]
        state["decoder_prefix_frames"] = 0
        if not isinstance(captured_output, torch.Tensor):
            raise TypeError("Qwen3-TTS XVec graph output must be a Tensor")
        return captured_output[: codes.shape[0]], state


class Qwen3TTSSuffixRoutine(_Qwen3TTSStatefulRoutineBase):
    def __init__(
        self,
        *,
        decoder: torch.nn.Module,
        num_quantizers: int,
        prefix_length: int,
        transitions: dict[str, dict[int, int]],
    ) -> None:
        super().__init__(decoder=decoder, num_quantizers=num_quantizers)
        self.prefix_length = prefix_length
        self.transitions = transitions

        def run(mode, target_frames, codes, old_quantized, old_conv, cache, mask):
            previous = self.transitions[mode][target_frames]
            new_frames = target_frames - previous
            rolling = self.decoder._is_suffix_cache_rolling(previous, int(old_quantized.shape[-1]))
            return self.decoder._decode_suffix(
                codes,
                old_quantized,
                old_conv,
                cache,
                self.prefix_length if mode == "icl" else 0,
                new_frames,
                rolling,
                attention_mask=mask,
            )

        self._runnable = run

    def eager_call(self, mode, target_frames, codes_list, request_caches, new_frames_list):
        """Handle only the runtime Target signature."""
        return self._eager_request_batch(
            str(mode),
            int(target_frames),
            codes_list,
            request_caches,
            new_frames_list,
        )

    def validate_runtime_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if kwargs or len(args) != 5 or not isinstance(args[2], list) or not isinstance(args[3], list):
            raise ValueError("qwen3_tts.suffix received malformed runtime inputs")
        codes_list, request_caches, new_frames_list = args[2], args[3], args[4]
        if len(codes_list) != len(request_caches) or len(codes_list) != len(new_frames_list):
            raise ValueError("qwen3_tts.suffix request lists must have equal lengths")
        if not codes_list or any(not isinstance(codes, torch.Tensor) or codes.ndim != 3 for codes in codes_list):
            raise ValueError("qwen3_tts.suffix expects non-empty [B,Q,F] code tensors")
        if any(int(codes.shape[0]) != 1 or int(codes.shape[1]) != self.num_quantizers for codes in codes_list):
            raise ValueError(f"Expected single-request {self.num_quantizers}-codebook suffix inputs")

    def resolve_runtime(self, args, kwargs, available: Set[VocoderCUDAGraphDescriptor]) -> VocoderRuntimeResolution:
        del kwargs
        mode, target_frames, codes_list = str(args[0]), int(args[1]), args[2]
        variant = Qwen3TTSSuffixVariant(mode, len(codes_list), target_frames)
        candidates = [
            d
            for d in available
            if isinstance(d.variant, Qwen3TTSSuffixVariant)
            and d.variant.mode == mode
            and d.variant.target_frames == target_frames
            and d.variant.batch_size >= variant.batch_size
        ]
        descriptor = min(candidates, key=lambda d: d.variant.batch_size, default=None)
        return VocoderRuntimeResolution(VocoderRuntimeKey(variant), descriptor)

    def allocate_buffers(self, descriptor, device):
        variant = descriptor.variant
        if not isinstance(variant, Qwen3TTSSuffixVariant):
            raise TypeError(f"Unexpected suffix Descriptor: {descriptor!r}")
        previous = self.transitions[variant.mode][variant.target_frames]
        new_frames = variant.target_frames - previous
        dtype = next(self.decoder.parameters()).dtype
        cache = self._make_dummy_cache(variant.mode, variant.batch_size, device, dtype)
        return _SuffixBuffers(
            codes=torch.zeros(variant.batch_size, self.num_quantizers, new_frames, dtype=torch.long, device=device),
            quantized=torch.zeros(
                variant.batch_size,
                self.decoder.config.codebook_dim,
                min(previous, self.prefix_length),
                dtype=dtype,
                device=device,
            ),
            conv=torch.zeros(
                variant.batch_size,
                min(previous, self.prefix_length - 2),
                self.decoder.config.latent_dim,
                dtype=dtype,
                device=device,
            ),
            mask=torch.ones(
                variant.batch_size,
                (self.prefix_length if variant.mode == "icl" else 0) + variant.target_frames,
                dtype=torch.bool,
                device=device,
            ),
            cache=cache,
            mode=variant.mode,
            target_frames=variant.target_frames,
        )

    def _make_dummy_cache(self, mode: str, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        config = self.decoder.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        cache = DynamicCache(config=config)
        cache.early_initialization(
            batch_size=batch_size, num_heads=config.num_key_value_heads, head_dim=head_dim, dtype=dtype, device=device
        )
        if mode == "icl":
            prefix_shape = (batch_size, config.num_key_value_heads, self.prefix_length, head_dim)
            prefix_keys = torch.zeros(prefix_shape, dtype=dtype, device=device)
            prefix_values = torch.zeros_like(prefix_keys)
            for layer_index in range(config.num_hidden_layers):
                cache.update(prefix_keys, prefix_values, layer_index)
        _materialize_cache_bookkeeping(
            cache,
            device,
            dtype=dtype,
            batch_size=batch_size,
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            num_layers=config.num_hidden_layers,
        )
        return {
            "ref_hidden": torch.zeros(
                batch_size, config.codebook_dim, self.prefix_length if mode == "icl" else 0, dtype=dtype, device=device
            ),
            "ref_conv": torch.zeros(
                batch_size, self.prefix_length if mode == "icl" else 0, config.latent_dim, dtype=dtype, device=device
            ),
            "prefix_hidden": torch.zeros(
                batch_size, self.prefix_length if mode == "icl" else 0, config.latent_dim, dtype=dtype, device=device
            ),
            "past_key_values": cache,
        }

    def forward_for_capture(self, buffers: object):
        if not isinstance(buffers, _SuffixBuffers):
            raise TypeError("Unexpected suffix capture buffers")
        return self._runnable(
            buffers.mode,
            buffers.target_frames,
            buffers.codes,
            buffers.quantized,
            buffers.conv,
            buffers.cache,
            buffers.mask,
        )

    def copy_runtime_inputs(self, args, kwargs, buffers: object) -> None:
        del kwargs
        if not isinstance(buffers, _SuffixBuffers):
            raise TypeError("Unexpected suffix replay buffers")
        mode, target_frames, codes_list, request_caches, new_frames_list = args
        buffers.codes.zero_()
        buffers.quantized.zero_()
        buffers.conv.zero_()
        buffers.mask.fill_(1)
        for row, (codes, cache, new_frames) in enumerate(zip(codes_list, request_caches, new_frames_list, strict=True)):
            buffers.codes[row, :, :new_frames].copy_(codes[0, :, -new_frames:])
            buffers.quantized[row : row + 1].copy_(cache["suffix_quantized"])
            buffers.conv[row : row + 1].copy_(cache["suffix_conv"])
            for key in ("ref_hidden", "ref_conv", "prefix_hidden"):
                buffers.cache[key][row : row + 1].copy_(cache[key])
            _copy_dynamic_cache_row(cache["past_key_values"], buffers.cache["past_key_values"], row)
            if mode == "icl":
                buffers.mask[row, : int(cache.get("prefix_pad_frames", 0))] = 0
        if buffers.mode != str(mode) or buffers.target_frames != int(target_frames):
            raise ValueError("Suffix runtime mode/target does not match captured Descriptor")

    def output_after_replay(self, args, kwargs, buffers: object, captured_output: object):
        del kwargs, buffers
        _mode, _target_frames, _codes_list, _request_caches, new_frames_list = args
        if (
            not isinstance(captured_output, tuple)
            or len(captured_output) != 3
            or not all(isinstance(value, torch.Tensor) for value in captured_output)
        ):
            raise TypeError("Qwen3-TTS suffix graph output must be a 3-tensor tuple")
        wav, next_quantized, next_conv = captured_output
        actual_batch = len(new_frames_list)
        return wav[:actual_batch], next_quantized[:actual_batch], next_conv[:actual_batch]

    def _eager_request_batch(
        self,
        mode: str,
        target_frames: int,
        codes_list: list[torch.Tensor],
        request_caches: list[dict[str, Any]],
        new_frames_list: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        metadata = _resolve_suffix_batch(
            self.decoder,
            mode,
            target_frames,
            codes_list,
            request_caches,
            new_frames_list,
            transitions_by_mode=self.transitions,
        )
        if metadata is None:
            return None
        prepared = _materialize_suffix_batch(
            self.decoder,
            mode,
            codes_list,
            request_caches,
            new_frames_list,
            metadata,
        )
        if prepared is None:
            return None
        return self._runnable(
            mode,
            target_frames,
            prepared.codes,
            prepared.old_quantized,
            prepared.old_conv,
            prepared.cache,
            prepared.attention_mask,
        )


def _positive_ints(value: object, *, path: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence of positive integers")
    result: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{path} must contain only positive integers")
        result.add(item)
    if not result:
        raise ValueError(f"{path} must not be empty")
    return tuple(sorted(result))


def _positive_ints_or_none(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, Sequence):
        raise TypeError("vocoder graph capture sizes must be a sequence")
    result = []
    for item in value:
        parsed = int(item)
        if parsed <= 0:
            raise ValueError("vocoder graph capture sizes must be positive")
        result.append(parsed)
    return tuple(sorted(set(result)))


def resolve_qwen3_tts_execution_settings(vllm_config: VllmConfig) -> Qwen3TTSExecutionSettings:
    """Resolve Qwen config once for decoder, Targets, and GraphExecutor."""
    model_config = vllm_config.model_config
    async_chunk = bool(getattr(model_config, "async_chunk", False))
    graph_value = getattr(model_config, "vocoder_cudagraph_config", None)
    graph_config = {} if graph_value is None else dict(graph_value) if isinstance(graph_value, Mapping) else None
    if graph_config is None:
        raise TypeError("vocoder_cudagraph must be a mapping")
    target_configs = graph_config.get("targets", {})
    if target_configs is None:
        target_configs = {}
    if not isinstance(target_configs, Mapping):
        raise TypeError("vocoder_cudagraph.targets must be a mapping")
    stateless_value = target_configs.get(STATELESS_TARGET_ID, {})
    if not isinstance(stateless_value, Mapping):
        raise TypeError(f"vocoder_cudagraph.targets.{STATELESS_TARGET_ID} must be a mapping")
    stateless_config = dict(stateless_value)
    connector_config = getattr(model_config, "stage_connector_config", None)
    connector_value = (
        connector_config.get("extra", connector_config)
        if isinstance(connector_config, Mapping)
        else getattr(connector_config, "extra", None)
    )
    connector = dict(connector_value) if isinstance(connector_value, Mapping) else {}

    def int_config(config: Mapping[str, Any], name: str, default: int) -> int:
        value = config.get(name, default)
        try:
            return default if value is None else int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}") from exc

    decode_chunk = int_config(stateless_config, "decode_chunk_frames", 300)
    decode_left = int_config(stateless_config, "decode_left_context_frames", 25)
    decode_batch = int_config(graph_config, "decode_batch_max_size", 0)
    if decode_chunk <= 0 or decode_left < 0:
        raise ValueError(
            "Invalid Qwen3-TTS Code2Wav decode chunk config: "
            f"decode_chunk_frames={decode_chunk}, decode_left_context_frames={decode_left}"
        )
    if decode_batch < 0:
        raise ValueError(f"Invalid Qwen3-TTS Code2Wav config decode_batch_max_size={decode_batch}")

    tf32 = graph_config.get("decode_enable_tf32", False)
    if isinstance(tf32, str):
        lowered = tf32.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            tf32 = True
        elif lowered in ("0", "false", "no", "off"):
            tf32 = False
        else:
            raise ValueError(f"Invalid Qwen3-TTS Code2Wav config decode_enable_tf32={tf32!r}")
    if not isinstance(tf32, (bool, int)):
        raise ValueError(f"Invalid Qwen3-TTS Code2Wav config decode_enable_tf32={tf32!r}")

    capture_batch_sizes = _positive_ints_or_none(graph_config.get("capture_batch_sizes"))
    capture_bucket_value = stateless_config.get("capture_bucket_sizes")
    if capture_bucket_value is None:
        stateless_capture_sizes = (150, decode_chunk + decode_left)
    else:
        stateless_capture_sizes = tuple(
            sorted(
                {
                    150,
                    decode_chunk + decode_left,
                    *_positive_ints(
                        capture_bucket_value,
                        path=f"vocoder_cudagraph.targets.{STATELESS_TARGET_ID}.capture_bucket_sizes",
                    ),
                }
            )
        )
    codec_chunk = int(connector.get("codec_chunk_frames", 0) or 0)
    initial_chunk = int(connector.get("initial_codec_chunk_frames", 1) or 1)
    ramp = parse_chunk_ramp(connector, steady=codec_chunk) if async_chunk else None
    return Qwen3TTSExecutionSettings(
        async_chunk=async_chunk,
        capture_batch_sizes=capture_batch_sizes,
        stateless_capture_sizes=stateless_capture_sizes,
        decode_chunk_frames=decode_chunk,
        decode_left_context_frames=decode_left,
        decode_batch_max_size=decode_batch,
        decode_enable_tf32=bool(tf32),
        codec_chunk_frames=codec_chunk,
        codec_left_context_frames=int(connector.get("codec_left_context_frames", 0) or 0),
        initial_codec_chunk_frames=initial_chunk,
        codec_chunk_ramp=tuple(ramp or ()),
    )


def build_qwen3_tts_targets(
    *,
    decoder: torch.nn.Module,
    settings: Qwen3TTSExecutionSettings,
    num_quantizers: int,
    total_upsample: int,
) -> tuple[VocoderCUDAGraphTarget, ...]:
    """Build Qwen3-TTS Targets without CUDA allocation or runnable invocation."""

    stateless = build_stateless_target(
        decoder=decoder,
        runnable=decoder._forward_exact,
        settings=settings,
        num_quantizers=num_quantizers,
        total_upsample=total_upsample,
    )
    stateful = build_stateful_targets(
        decoder=decoder,
        settings=settings,
        num_quantizers=num_quantizers,
        prefix_length=int(getattr(getattr(decoder, "config", None), "sliding_window", 0) or 0),
    )
    return (stateless, *stateful)


def build_stateless_target(
    *,
    decoder: torch.nn.Module,
    runnable: Callable[[torch.Tensor], torch.Tensor],
    settings: Qwen3TTSExecutionSettings,
    num_quantizers: int,
    total_upsample: int,
) -> VocoderCUDAGraphTarget:
    """Resolve config into one immutable stateless Target, CPU-only."""

    if settings.async_chunk:
        # Keep the known Target in the stable registry so overrides can be
        # validated, but declare no stateless startup coverage; the stateful
        # ICL/x-vector/suffix Targets are built separately below.
        batch_sizes: tuple[int, ...] = ()
        capture_sizes: tuple[int, ...] = ()
    else:
        batch_sizes = settings.capture_batch_sizes or (1,)
        capture_sizes = settings.stateless_capture_sizes

    descriptors = tuple(
        VocoderCUDAGraphDescriptor(Qwen3TTSStatelessVariant(batch_size, frames))
        for batch_size in batch_sizes
        for frames in capture_sizes
    )
    routine = Qwen3TTSStatelessRoutine(
        decoder=decoder,
        runnable=runnable,
        num_quantizers=num_quantizers,
        total_upsample=total_upsample,
    )
    return VocoderCUDAGraphTarget(
        target_id=STATELESS_TARGET_ID,
        routine=routine,
        descriptors=descriptors,
        clone_output=True,
        supported_config_keys=TARGET_CONFIG_KEYS,
    )


def build_stateful_targets(
    *,
    decoder: torch.nn.Module,
    settings: Qwen3TTSExecutionSettings,
    num_quantizers: int,
    prefix_length: int,
) -> tuple[VocoderCUDAGraphTarget, ...]:
    """Build Qwen3-TTS stateful Targets from connector transitions."""

    batch_sizes = settings.capture_batch_sizes or (1,)

    contract = resolve_stateful_chunk_contract(
        prefix_length=prefix_length,
        initial_codec_chunk_frames=settings.initial_codec_chunk_frames,
        codec_chunk_frames=settings.codec_chunk_frames or 25,
        codec_chunk_ramp=settings.codec_chunk_ramp,
    )
    resolved_initial_frames = contract.resolved_initial_frames
    transitions = (
        contract.transitions
        if prefix_length > 2 and resolved_initial_frames > 0 and settings.codec_chunk_frames > 0
        else {}
    )

    if not settings.async_chunk:
        batch_sizes = ()
        transitions = {}

    icl_routine = Qwen3TTSIclPrefixRoutine(
        decoder=decoder,
        num_quantizers=num_quantizers,
        prefix_length=prefix_length,
        initial_frames=resolved_initial_frames,
    )
    xvec_routine = Qwen3TTSXvecPrefixRoutine(
        decoder=decoder,
        num_quantizers=num_quantizers,
        initial_frames=resolved_initial_frames,
    )
    suffix_routine = Qwen3TTSSuffixRoutine(
        decoder=decoder,
        num_quantizers=num_quantizers,
        prefix_length=prefix_length,
        transitions={"icl": transitions, "xvec": transitions},
    )
    icl_descriptors = tuple(
        VocoderCUDAGraphDescriptor(Qwen3TTSIclPrefixVariant(batch, prefix_length + resolved_initial_frames))
        for batch in batch_sizes
    )
    xvec_descriptors = tuple(
        VocoderCUDAGraphDescriptor(Qwen3TTSXvecPrefixVariant(batch, resolved_initial_frames)) for batch in batch_sizes
    )
    suffix_descriptors = tuple(
        VocoderCUDAGraphDescriptor(Qwen3TTSSuffixVariant(mode, batch, target))
        for mode in ("icl", "xvec")
        for target in transitions
        for batch in batch_sizes
    )
    return (
        VocoderCUDAGraphTarget("qwen3_tts.icl_prefix", icl_routine, icl_descriptors, clone_output=True),
        VocoderCUDAGraphTarget("qwen3_tts.xvec_prefix", xvec_routine, xvec_descriptors, clone_output=True),
        VocoderCUDAGraphTarget("qwen3_tts.suffix", suffix_routine, suffix_descriptors, clone_output=True),
    )
