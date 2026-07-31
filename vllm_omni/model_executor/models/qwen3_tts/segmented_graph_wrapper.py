# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import copy
import os
from collections import Counter

import torch
from torch.cuda import CUDAGraph
from transformers.cache_utils import DynamicCache
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(decoder, capture_sizes=[25, 50, 100, 200, 300])
        wrapper.warmup(device)

        # During inference:
        output = wrapper.decode(codes, caches)  # Uses CUDA graph if possible
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        capture_batch_sizes: list[int] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
        initial_chunk_frames: int = 1,
        codec_chunk_frames: int = 25,
    ):
        self.decoder = decoder
        self._configured_capture_sizes = sorted(set(capture_sizes or []))
        self.capture_batch_sizes = sorted(set(capture_batch_sizes or [1, 2, 4, 8]))
        self.num_quantizers = num_quantizers
        self.enabled = enabled
        self._warmed_up = False

        self.combined_graphs: dict[tuple[str, int, int], CUDAGraph] = {}
        self.combined_static_codes: dict[tuple[str, int, int], torch.Tensor] = {}
        self.combined_static_quantized: dict[tuple[str, int, int], torch.Tensor] = {}
        self.combined_static_conv: dict[tuple[str, int, int], torch.Tensor] = {}
        self.combined_static_masks: dict[tuple[str, int, int], torch.Tensor] = {}
        self.combined_static_caches: dict[tuple[str, int, int], dict] = {}
        self.combined_static_outputs: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.prefix_graphs: dict[int, CUDAGraph] = {}
        self.prefix_static_inputs: dict[int, torch.Tensor] = {}
        self.prefix_static_outputs: dict[int, torch.Tensor] = {}
        self.prefix_suffix_quantized: dict[int, torch.Tensor] = {}
        self.prefix_suffix_conv: dict[int, torch.Tensor] = {}
        self.prefix_hidden_masks: dict[int, torch.Tensor] = {}
        self.prefix_attention_masks: dict[int, torch.Tensor] = {}
        self.suffix_attention_masks: dict[int, torch.Tensor] = {}
        self.prefix_static_caches: dict[int, dict] = {}

        self._device = None
        self.prefix_length = int(getattr(self.decoder.config, "sliding_window", 0) or 0)
        self.initial_chunk_frames = int(initial_chunk_frames)
        self.codec_chunk_frames = int(codec_chunk_frames)
        if self.prefix_length <= 2:
            raise ValueError(f"decoder sliding_window must be greater than 2, got {self.prefix_length}")
        if self.initial_chunk_frames <= 0 or self.codec_chunk_frames <= 0:
            raise ValueError(
                "initial_chunk_frames and codec_chunk_frames must be positive, "
                f"got {self.initial_chunk_frames} and {self.codec_chunk_frames}"
            )
        self._previous_frames_by_target = self._derive_suffix_transitions()
        self._xvec_previous_frames_by_target = self._derive_xvec_suffix_transitions()
        derived_capture_sizes = sorted(self._previous_frames_by_target)
        if self._configured_capture_sizes:
            configured = set(self._configured_capture_sizes)
            self.capture_sizes = [size for size in derived_capture_sizes if size in configured]
            ignored = sorted(configured - set(derived_capture_sizes))
            if ignored:
                logger.warning(
                    "Ignoring unreachable segmented Code2Wav capture sizes %s; valid sizes for "
                    "initial_chunk_frames=%d codec_chunk_frames=%d sliding_window=%d are %s",
                    ignored,
                    self.initial_chunk_frames,
                    self.codec_chunk_frames,
                    self.prefix_length,
                    derived_capture_sizes,
                )
        else:
            self.capture_sizes = derived_capture_sizes
        self._stats_enabled = os.environ.get(
            "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS",
            "",
        ).lower() in ("1", "true", "yes", "on")
        self._stats_log_every = int(
            os.environ.get(
                "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS_LOG_EVERY",
                "100",
            )
            or 0
        )
        self._stats_file = os.environ.get(
            "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS_FILE",
            "",
        )
        self._stats_total_requests = 0
        self._stats_hit_requests = 0
        self._stats_fallback_requests = 0
        self._stats_replays: Counter[tuple[str, int, int]] = Counter()
        self._stats_fallbacks: Counter[tuple[str, int]] = Counter()

    def _derive_suffix_transitions(self) -> dict[int, int]:
        transitions: dict[int, int] = {}
        previous = self.initial_chunk_frames
        while previous < self.prefix_length:
            target = previous + self.codec_chunk_frames
            transitions[target] = previous
            previous = target
        transitions[self.prefix_length + self.codec_chunk_frames] = self.prefix_length
        return transitions

    def _derive_xvec_suffix_transitions(self) -> dict[int, int]:
        transitions: dict[int, int] = {}
        previous = self.codec_chunk_frames
        while previous < self.prefix_length:
            target = previous + self.codec_chunk_frames
            transitions[target] = previous
            previous = target
        transitions[self.prefix_length + self.codec_chunk_frames] = self.prefix_length
        return transitions

    def _transitions_for_mode(self, mode: str) -> dict[int, int]:
        return self._xvec_previous_frames_by_target if mode == "xvec" else self._previous_frames_by_target

    def _record_graph_hit(self, phase: str, batch_size: int, request_count: int) -> None:
        if not getattr(self, "_stats_enabled", False):
            return
        self._stats_total_requests += request_count
        self._stats_hit_requests += request_count
        self._stats_replays[(phase, batch_size, request_count)] += 1
        self._maybe_log_stats()

    def _record_graph_fallback(self, phase: str, request_count: int) -> None:
        if not getattr(self, "_stats_enabled", False):
            return
        self._stats_total_requests += request_count
        self._stats_fallback_requests += request_count
        self._stats_fallbacks[(phase, request_count)] += 1
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        if self._stats_log_every > 0 and self._stats_total_requests % self._stats_log_every == 0:
            self.log_decode_stats()

    def log_decode_stats(self) -> None:
        if not getattr(self, "_stats_enabled", False) or self._stats_total_requests == 0:
            return
        hit_rate = 100.0 * self._stats_hit_requests / self._stats_total_requests
        message = (
            "Segmented Code2Wav CUDA Graph stats: "
            f"requests={self._stats_total_requests} hits={self._stats_hit_requests} "
            f"fallbacks={self._stats_fallback_requests} hit_rate={hit_rate:.2f}% "
            f"replays={self._stats_replays.most_common(20)} "
            f"fallback_groups={self._stats_fallbacks.most_common(20)}"
        )
        logger.warning("%s", message)
        stats_file = getattr(self, "_stats_file", "")
        if stats_file:
            fd = os.open(stats_file, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"pid={os.getpid()} {message}\n".encode())
            finally:
                os.close(fd)

    def _get_capture_shapes(self) -> list[tuple[int, int]]:
        shapes = {(batch_size, size) for batch_size in self.capture_batch_sizes for size in self.capture_sizes}
        return sorted(shapes)

    def _make_dummy_cache(
        self,
        batch_size: int,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> dict:
        config = self.decoder.config
        dummy_hidden = torch.zeros(
            batch_size,
            config.codebook_dim,
            self.prefix_length,
            device=device,
            dtype=model_dtype,
        )
        dummy_conv = torch.zeros(
            batch_size,
            self.prefix_length,
            config.latent_dim,
            device=device,
            dtype=model_dtype,
        )
        dummy_kv = DynamicCache(config=config)
        assert self.prefix_length == config.sliding_window
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        prefix_kv_shape = (batch_size, config.num_key_value_heads, self.prefix_length, head_dim)

        prefix_keys = torch.zeros(prefix_kv_shape, device=device, dtype=model_dtype)
        prefix_values = torch.zeros_like(prefix_keys)

        for layer_idx in range(config.num_hidden_layers):
            dummy_kv.update(prefix_keys, prefix_values, layer_idx)

        expected_kv_length = self.prefix_length - 1
        if dummy_kv.get_seq_length() != self.prefix_length or any(
            layer.keys.shape[-2] != expected_kv_length for layer in dummy_kv.layers
        ):
            raise RuntimeError(
                "Dummy sliding-window cache does not match the captured prefix "
                f"state: expected logical length {self.prefix_length} and "
                f"physical KV length {expected_kv_length}"
            )

        dummy_prefix_hidden = torch.zeros(
            batch_size,
            self.prefix_length,
            config.latent_dim,
            device=device,
            dtype=model_dtype,
        )
        dummy_upsample = torch.zeros(
            batch_size,
            config.latent_dim,
            self.prefix_length * self.decoder.upsample_factor,
            device=device,
            dtype=model_dtype,
        )
        dummy_wav = torch.zeros(
            batch_size,
            1,
            self.prefix_length * self.decoder.total_upsample,
            device=device,
            dtype=model_dtype,
        )
        return {
            "ref_hidden": dummy_hidden,
            "ref_conv": dummy_conv,
            "past_key_values": dummy_kv,
            "prefix_hidden": dummy_prefix_hidden,
            "ref_upsample": dummy_upsample,
            "ref_wav": dummy_wav,
        }

    def _make_dummy_xvec_cache(
        self,
        batch_size: int,
        previous_frames: int,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> dict:
        config = self.decoder.config
        boundary_frames = 2 if previous_frames >= self.prefix_length else 0
        return {
            "ref_hidden": torch.zeros(
                batch_size, config.codebook_dim, boundary_frames, device=device, dtype=model_dtype
            ),
            "ref_conv": torch.zeros(batch_size, 0, config.latent_dim, device=device, dtype=model_dtype),
            "past_key_values": DynamicCache(config=config),
            "prefix_hidden": torch.zeros(batch_size, 0, config.latent_dim, device=device, dtype=model_dtype),
            "ref_upsample": torch.zeros(batch_size, config.latent_dim, 0, device=device, dtype=model_dtype),
            "ref_wav": torch.zeros(batch_size, 1, 0, device=device, dtype=model_dtype),
        }

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
    ):
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        self._device = device
        self.decoder.eval()

        if not self.capture_sizes:
            raise ValueError("No reachable segmented capture sizes were configured")

        self.capture_batch_sizes = [bs for bs in self.capture_batch_sizes if bs > 0]
        if not self.capture_batch_sizes:
            self.capture_batch_sizes = [1]
        capture_shapes = self._get_capture_shapes()

        for batch_size in self.capture_batch_sizes:
            try:
                self._capture_icl_prefix(batch_size, device, dtype)
                logger.info("  Captured prefix CUDA Graph for batch=%d", batch_size)
            except Exception:
                logger.warning("  Failed to capture prefix CUDA Graph for batch=%d", batch_size, exc_info=True)

        logger.info(
            "Starting CUDA Graph warmup for %d shapes: batch_sizes=%s seq_lens=%s",
            len(capture_shapes),
            self.capture_batch_sizes,
            self.capture_sizes,
        )

        for batch_size, size in capture_shapes:
            try:
                caches = self._make_dummy_cache(batch_size, device, next(self.decoder.parameters()).dtype)
                self._capture_combined_suffix("icl", batch_size, size, caches, device, dtype)
                logger.info("  Captured combined suffix CUDA Graph for batch=%d size=%d", batch_size, size)
            except Exception:
                logger.warning(
                    "  Failed to capture combined suffix graph for batch=%d size=%d",
                    batch_size,
                    size,
                    exc_info=True,
                )

        model_dtype = next(self.decoder.parameters()).dtype
        for batch_size in self.capture_batch_sizes:
            for size, previous_frames in self._xvec_previous_frames_by_target.items():
                try:
                    caches = self._make_dummy_xvec_cache(batch_size, previous_frames, device, model_dtype)
                    self._capture_combined_suffix("xvec", batch_size, size, caches, device, dtype)
                    logger.info("  Captured xvec suffix CUDA Graph for batch=%d size=%d", batch_size, size)
                except Exception:
                    logger.warning(
                        "  Failed to capture xvec suffix graph for batch=%d size=%d",
                        batch_size,
                        size,
                        exc_info=True,
                    )

        self._warmed_up = bool(self.prefix_graphs) or bool(self.combined_graphs)

    def _run_combined_suffix(
        self,
        mode: str,
        codes: torch.Tensor,
        old_quantized: torch.Tensor,
        old_conv: torch.Tensor,
        caches: dict,
        target_frames: int,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_frames = self._transitions_for_mode(mode)[target_frames]
        rolling = self.decoder._is_suffix_cache_rolling(previous_frames, int(old_quantized.shape[-1]))
        return self.decoder._decode_suffix_incremental(
            codes,
            old_quantized,
            old_conv,
            caches,
            self.prefix_length if mode == "icl" else 0,
            self.codec_chunk_frames,
            rolling,
            attention_mask=attention_mask,
        )

    def _capture_combined_suffix(
        self,
        mode: str,
        batch_size: int,
        target_frames: int,
        caches: dict,
        device: torch.device,
        code_dtype: torch.dtype,
    ) -> None:
        previous_frames = self._transitions_for_mode(mode)[target_frames]
        model_dtype = next(self.decoder.parameters()).dtype
        key = (mode, batch_size, target_frames)
        static_codes = torch.zeros(
            batch_size, self.num_quantizers, self.codec_chunk_frames, dtype=code_dtype, device=device
        )
        static_quantized = torch.zeros(
            batch_size,
            self.decoder.config.codebook_dim,
            min(previous_frames, self.prefix_length),
            dtype=model_dtype,
            device=device,
        )
        static_conv = torch.zeros(
            batch_size,
            min(previous_frames, self.prefix_length - 2),
            self.decoder.config.latent_dim,
            dtype=model_dtype,
            device=device,
        )
        prefix_frames = self.prefix_length if mode == "icl" else 0
        static_mask = torch.ones(batch_size, prefix_frames + target_frames, dtype=torch.bool, device=device)

        with torch.no_grad():
            _ = self._run_combined_suffix(
                mode,
                static_codes,
                static_quantized,
                static_conv,
                caches,
                target_frames,
                static_mask,
            )
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                outputs = self._run_combined_suffix(
                    mode,
                    static_codes,
                    static_quantized,
                    static_conv,
                    caches,
                    target_frames,
                    static_mask,
                )

        self.combined_graphs[key] = graph
        self.combined_static_codes[key] = static_codes
        self.combined_static_quantized[key] = static_quantized
        self.combined_static_conv[key] = static_conv
        self.combined_static_masks[key] = static_mask
        self.combined_static_caches[key] = caches
        self.combined_static_outputs[key] = outputs

    def _capture_icl_prefix(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        size = self.prefix_length + self.initial_chunk_frames
        caches: dict = {}
        static_input = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.decoder._decode_icl_first_chunk(static_input, caches, self.prefix_length)
        torch.accelerator.synchronize(device)

        config = self.decoder.config
        model_dtype = next(self.decoder.parameters()).dtype
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        prefix_cache = DynamicCache(config=config)
        prefix_cache.early_initialization(
            batch_size=batch_size,
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=model_dtype,
            device=device,
        )
        suffix_quantized = torch.zeros(
            batch_size,
            config.codebook_dim,
            self.prefix_length,
            dtype=model_dtype,
            device=device,
        )
        suffix_conv = torch.zeros(
            batch_size,
            self.prefix_length - 2,
            config.latent_dim,
            dtype=model_dtype,
            device=device,
        )
        prefix_hidden_mask = torch.ones(batch_size, 1, size, dtype=model_dtype, device=device)
        prefix_attention_mask = torch.ones(batch_size, self.prefix_length, dtype=torch.bool, device=device)
        suffix_attention_mask = torch.ones(batch_size, size, dtype=torch.bool, device=device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder._decode_icl_first_chunk(
                    static_input,
                    caches,
                    self.prefix_length,
                    prefix_cache=prefix_cache,
                    prefix_hidden_mask=prefix_hidden_mask,
                    prefix_attention_mask=prefix_attention_mask,
                    suffix_attention_mask=suffix_attention_mask,
                )
                suffix_quantized[:, :, : self.initial_chunk_frames].copy_(caches["suffix_quantized"])
                suffix_conv[:, : self.initial_chunk_frames, :].copy_(caches["suffix_conv"])

        self.prefix_graphs[batch_size] = graph
        self.prefix_static_inputs[batch_size] = static_input
        self.prefix_static_outputs[batch_size] = static_output
        self.prefix_suffix_quantized[batch_size] = suffix_quantized
        self.prefix_suffix_conv[batch_size] = suffix_conv
        self.prefix_hidden_masks[batch_size] = prefix_hidden_mask
        self.prefix_attention_masks[batch_size] = prefix_attention_mask
        self.suffix_attention_masks[batch_size] = suffix_attention_mask
        self.prefix_static_caches[batch_size] = caches

    def _decode_icl_prefix(self, codes: torch.Tensor, caches: dict) -> torch.Tensor:
        batch_size = int(codes.shape[0])
        if batch_size != 1 or batch_size not in self.prefix_graphs or torch.cuda.is_current_stream_capturing():
            return self.decoder._decode_icl_first_chunk(codes, caches, int(caches["prefix_frames"]))

        actual_prefix_frames = int(caches["prefix_frames"])
        suffix_frames = int(codes.shape[-1]) - actual_prefix_frames
        if not 0 < actual_prefix_frames <= self.prefix_length or suffix_frames != self.initial_chunk_frames:
            return self.decoder._decode_icl_first_chunk(codes, caches, actual_prefix_frames)

        prefix_static_input = self.prefix_static_inputs[batch_size]
        prefix_hidden_mask = self.prefix_hidden_masks[batch_size]
        prefix_attention_mask = self.prefix_attention_masks[batch_size]
        suffix_attention_mask = self.suffix_attention_masks[batch_size]
        prefix_pad_frames = self.prefix_length - actual_prefix_frames
        prefix_static_input.zero_()
        prefix_static_input[:, :, prefix_pad_frames : self.prefix_length].copy_(codes[:, :, :actual_prefix_frames])
        prefix_static_input[:, :, self.prefix_length :].copy_(codes[:, :, actual_prefix_frames:])
        prefix_hidden_mask.fill_(1)
        prefix_hidden_mask[:, :, :prefix_pad_frames] = 0
        prefix_attention_mask.fill_(1)
        prefix_attention_mask[:, :prefix_pad_frames] = 0
        suffix_attention_mask.fill_(1)
        suffix_attention_mask[:, :prefix_pad_frames] = 0

        self.prefix_graphs[batch_size].replay()
        caches.update(self.prefix_static_caches[batch_size])
        caches["prefix_pad_frames"] = prefix_pad_frames
        caches["suffix_quantized"] = self.prefix_suffix_quantized[batch_size][:, :, : self.initial_chunk_frames]
        caches["suffix_conv"] = self.prefix_suffix_conv[batch_size][:, : self.initial_chunk_frames, :]
        self._ensure_suffix_buffers(caches)
        return self.prefix_static_outputs[batch_size][..., prefix_pad_frames * self.decoder.total_upsample :]

    def _ensure_suffix_buffers(self, caches: dict) -> None:
        if "_suffix_quantized_buffers" in caches:
            return

        suffix_quantized = caches["suffix_quantized"]
        suffix_conv = caches["suffix_conv"]
        quantized_capacity = self.prefix_length
        conv_capacity = self.prefix_length - 2
        quantized_buffers = [
            suffix_quantized.new_zeros((*suffix_quantized.shape[:-1], quantized_capacity)) for _ in range(2)
        ]
        conv_buffers = [
            suffix_conv.new_zeros((suffix_conv.shape[0], conv_capacity, suffix_conv.shape[-1])) for _ in range(2)
        ]
        quantized_valid = min(int(suffix_quantized.shape[-1]), quantized_capacity)
        conv_valid = min(int(suffix_conv.shape[1]), conv_capacity)
        quantized_buffers[0][:, :, :quantized_valid].copy_(suffix_quantized[:, :, -quantized_valid:])
        conv_buffers[0][:, :conv_valid, :].copy_(suffix_conv[:, -conv_valid:, :])
        caches["_suffix_quantized_buffers"] = quantized_buffers
        caches["_suffix_conv_buffers"] = conv_buffers
        caches["_suffix_buffer_index"] = 0
        caches.setdefault("suffix_frames", int(suffix_quantized.shape[-1]))
        caches["suffix_quantized"] = quantized_buffers[0][:, :, :quantized_valid]
        caches["suffix_conv"] = conv_buffers[0][:, :conv_valid, :]

    def _copy_caches(self, caches: dict, static_caches: dict):
        tensor_cache_keys = {
            "ref_hidden",
            "ref_conv",
            "prefix_hidden",
            "ref_upsample",
            "ref_wav",
        }
        for key in tensor_cache_keys:
            static_caches[key].copy_(caches[key])

        static_kv = static_caches["past_key_values"]
        request_kv = caches["past_key_values"]
        for static_layer, request_layer in zip(
            static_kv.layers,
            request_kv.layers,
            strict=True,
        ):
            static_layer.keys.copy_(request_layer.keys)
            static_layer.values.copy_(request_layer.values)

    def _copy_request_caches_to_batch(self, request_caches: list[dict], static_caches: dict) -> None:
        tensor_cache_keys = ("ref_hidden", "ref_conv", "prefix_hidden", "ref_upsample", "ref_wav")
        for row, request_cache in enumerate(request_caches):
            for key in tensor_cache_keys:
                static_caches[key][row : row + 1].copy_(request_cache[key])

            for static_layer, request_layer in zip(
                static_caches["past_key_values"].layers,
                request_cache["past_key_values"].layers,
                strict=True,
            ):
                static_layer.keys[row : row + 1].copy_(request_layer.keys)
                static_layer.values[row : row + 1].copy_(request_layer.values)

    @staticmethod
    def _slice_dynamic_cache(cache: DynamicCache, row: int) -> DynamicCache:
        request_cache = copy.deepcopy(cache)
        for layer in request_cache.layers:
            layer.keys = layer.keys[row : row + 1].clone()
            layer.values = layer.values[row : row + 1].clone()
        return request_cache

    def _select_batch_bucket(self, request_count: int, available: set[int]) -> int | None:
        return next((size for size in self.capture_batch_sizes if size >= request_count and size in available), None)

    @staticmethod
    def _split_for_graph_buckets(request_count: int, available: set[int]) -> list[tuple[int, int]]:
        if request_count <= 0 or not available:
            return []
        largest = max(available)
        ranges: list[tuple[int, int]] = []
        start = 0
        while request_count - start > largest:
            ranges.append((start, start + largest))
            start += largest
        ranges.append((start, request_count))
        return ranges

    def _decode_prefix_batch(
        self,
        codes_list: list[torch.Tensor],
        request_caches: list[dict],
    ) -> list[torch.Tensor] | None:
        available = set(self.prefix_graphs)
        if torch.cuda.is_current_stream_capturing():
            self._record_graph_fallback("prefix:capture", len(codes_list))
            return None
        if len(codes_list) > max(available, default=0):
            outputs: list[torch.Tensor] = []
            for start, end in self._split_for_graph_buckets(len(codes_list), available):
                chunk_outputs = self._decode_prefix_batch(
                    codes_list[start:end],
                    request_caches[start:end],
                )
                if chunk_outputs is None:
                    return None
                outputs.extend(chunk_outputs)
            return outputs

        batch_size = self._select_batch_bucket(len(codes_list), available)
        if batch_size is None:
            self._record_graph_fallback("prefix:no_graph", len(codes_list))
            return None

        static_input = self.prefix_static_inputs[batch_size]
        hidden_mask = self.prefix_hidden_masks[batch_size]
        prefix_mask = self.prefix_attention_masks[batch_size]
        suffix_mask = self.suffix_attention_masks[batch_size]
        static_input.zero_()
        hidden_mask.fill_(1)
        prefix_mask.fill_(1)
        suffix_mask.fill_(1)

        prefix_pads: list[int] = []
        for row, (codes, cache) in enumerate(zip(codes_list, request_caches, strict=True)):
            prefix_frames = int(cache["prefix_frames"])
            suffix_frames = int(codes.shape[-1]) - prefix_frames
            if not 0 < prefix_frames <= self.prefix_length or suffix_frames != self.initial_chunk_frames:
                return None
            prefix_pad = self.prefix_length - prefix_frames
            prefix_pads.append(prefix_pad)
            static_input[row, :, prefix_pad : self.prefix_length].copy_(codes[0, :, :prefix_frames])
            static_input[row, :, self.prefix_length :].copy_(codes[0, :, prefix_frames:])
            hidden_mask[row, :, :prefix_pad] = 0
            prefix_mask[row, :prefix_pad] = 0
            suffix_mask[row, :prefix_pad] = 0

        self.prefix_graphs[batch_size].replay()
        self._record_graph_hit("prefix", batch_size, len(codes_list))
        static_caches = self.prefix_static_caches[batch_size]
        outputs: list[torch.Tensor] = []
        tensor_keys = ("ref_hidden", "ref_conv", "prefix_hidden", "ref_upsample", "ref_wav")
        for row, (cache, prefix_pad) in enumerate(zip(request_caches, prefix_pads, strict=True)):
            for key in tensor_keys:
                cache[key] = static_caches[key][row : row + 1].clone()
            cache["past_key_values"] = self._slice_dynamic_cache(static_caches["past_key_values"], row)
            cache["decoder_prefix_frames"] = self.prefix_length
            cache["prefix_pad_frames"] = prefix_pad
            cache["suffix_quantized"] = self.prefix_suffix_quantized[batch_size][
                row : row + 1, :, : self.initial_chunk_frames
            ].clone()
            cache["suffix_conv"] = self.prefix_suffix_conv[batch_size][
                row : row + 1, : self.initial_chunk_frames, :
            ].clone()
            cache["suffix_frames"] = self.initial_chunk_frames
            start = prefix_pad * self.decoder.total_upsample
            outputs.append(self.prefix_static_outputs[batch_size][row : row + 1, :, start:].clone())
        return outputs

    def _decode_suffix_batch(
        self,
        mode: str,
        target_frames: int,
        codes_list: list[torch.Tensor],
        request_caches: list[dict],
        new_frames_list: list[int],
    ) -> list[torch.Tensor] | None:
        available = {
            batch
            for graph_mode, batch, target in self.combined_graphs
            if graph_mode == mode and target == target_frames
        }
        if torch.cuda.is_current_stream_capturing():
            self._record_graph_fallback(f"suffix_{target_frames}:capture", len(codes_list))
            return None
        if len(codes_list) > max(available, default=0):
            outputs: list[torch.Tensor] = []
            for start, end in self._split_for_graph_buckets(len(codes_list), available):
                chunk_outputs = self._decode_suffix_batch(
                    mode,
                    target_frames,
                    codes_list[start:end],
                    request_caches[start:end],
                    new_frames_list[start:end],
                )
                if chunk_outputs is None:
                    return None
                outputs.extend(chunk_outputs)
            return outputs

        batch_size = self._select_batch_bucket(len(codes_list), available)
        if batch_size is None:
            self._record_graph_fallback(f"suffix_{target_frames}:no_graph", len(codes_list))
            return None
        key = (mode, batch_size, target_frames)
        static_codes = self.combined_static_codes[key]
        static_quantized = self.combined_static_quantized[key]
        static_conv = self.combined_static_conv[key]
        static_mask = self.combined_static_masks[key]
        static_caches = self.combined_static_caches[key]
        static_codes.zero_()
        static_mask.fill_(1)

        xvec_boundaries: list[torch.Tensor | None] = []
        for row, (codes, cache, new_frames) in enumerate(zip(codes_list, request_caches, new_frames_list, strict=True)):
            self._ensure_suffix_buffers(cache)
            xvec_boundaries.append(
                self._get_xvec_boundary(cache, self._transitions_for_mode(mode)[target_frames], new_frames)
                if mode == "xvec"
                else None
            )
            static_codes[row, :, :new_frames].copy_(codes[0, :, -new_frames:])
            static_quantized[row : row + 1].copy_(cache["suffix_quantized"])
            static_conv[row : row + 1].copy_(cache["suffix_conv"])
            if mode == "icl":
                static_mask[row, : int(cache.get("prefix_pad_frames", 0))] = 0

        self._copy_request_caches_to_batch(request_caches, static_caches)
        self.combined_graphs[key].replay()
        self._record_graph_hit(f"suffix_{target_frames}", batch_size, len(codes_list))
        graph_output, graph_next_quantized, graph_next_conv = self.combined_static_outputs[key]
        outputs: list[torch.Tensor] = []
        for row, (cache, new_frames) in enumerate(zip(request_caches, new_frames_list, strict=True)):
            previous_frames = self._transitions_for_mode(mode)[target_frames]
            actual_suffix_frames = previous_frames + new_frames
            output = self._trim_replay_output(
                graph_output[row : row + 1],
                actual_suffix_frames,
                target_frames,
            ).clone()
            outputs.append(output)
            if new_frames == self.codec_chunk_frames:
                current_index = int(cache["_suffix_buffer_index"])
                next_index = 1 - current_index
                next_quantized = cache["_suffix_quantized_buffers"][next_index]
                next_conv = cache["_suffix_conv_buffers"][next_index]
                quantized_valid = int(graph_next_quantized.shape[-1])
                conv_valid = int(graph_next_conv.shape[1])
                next_quantized[:, :, :quantized_valid].copy_(graph_next_quantized[row : row + 1])
                next_conv[:, :conv_valid, :].copy_(graph_next_conv[row : row + 1])
                cache["_suffix_buffer_index"] = next_index
                cache["suffix_frames"] = target_frames
                cache["suffix_quantized"] = next_quantized[:, :, :quantized_valid]
                cache["suffix_conv"] = next_conv[:, :conv_valid, :]
            if xvec_boundaries[row] is not None:
                cache["ref_hidden"] = xvec_boundaries[row]
        return outputs

    def _get_xvec_boundary(self, cache: dict, cached_frames: int, new_frames: int) -> torch.Tensor | None:
        suffix_cache_length = self.prefix_length
        boundary_start = cached_frames + new_frames - suffix_cache_length - 2
        if 0 <= boundary_start and boundary_start + 2 <= cached_frames:
            return cache["suffix_quantized"][:, :, boundary_start : boundary_start + 2].clone()
        return None

    def _batched_request_decode(
        self,
        codes_list: list[torch.Tensor],
        request_caches: list[dict],
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor | None] = [None] * len(codes_list)
        incremental: list[bool] = [False] * len(codes_list)
        groups: dict[tuple[str, int], list[int]] = {}
        for index, (codes, cache) in enumerate(zip(codes_list, request_caches, strict=True)):
            if "ref_wav" not in cache:
                prefix_frames = int(cache["prefix_frames"])
                suffix_frames = int(codes.shape[-1]) - prefix_frames
                if prefix_frames == 0:
                    phase = "xvec_first"
                elif 0 < prefix_frames <= self.prefix_length and suffix_frames == self.initial_chunk_frames:
                    phase = "prefix"
                else:
                    phase = "eager"
                groups.setdefault((phase, 0), []).append(index)
                continue
            request_kv = cache["past_key_values"]
            decoder_prefix = int(cache.get("decoder_prefix_frames", cache["prefix_frames"]))
            mode = "xvec" if decoder_prefix == 0 else "icl"
            expected_kv_length = 0 if mode == "xvec" else self.prefix_length
            if decoder_prefix != expected_kv_length or request_kv.get_seq_length() != expected_kv_length:
                groups.setdefault(("eager", 0), []).append(index)
                continue
            logical_prefix = int(cache["prefix_frames"])
            suffix_frames = int(codes.shape[-1]) - logical_prefix
            previous = int(cache.get("suffix_frames", cache["suffix_quantized"].shape[-1]))
            rolling = self.decoder._is_suffix_cache_rolling(
                previous,
                int(cache["suffix_quantized"].shape[-1]),
            )
            retained = self.prefix_length if rolling else previous
            new_frames = suffix_frames - retained
            target = (
                self.prefix_length + self.codec_chunk_frames
                if rolling
                else next(
                    (target for target, source in self._transitions_for_mode(mode).items() if source == previous),
                    None,
                )
            )
            if target is None or not 0 < new_frames <= self.codec_chunk_frames:
                groups.setdefault(("eager", 0), []).append(index)
            else:
                groups.setdefault((f"suffix:{mode}", target), []).append(index)

        for (phase, target), indices in groups.items():
            group_codes = [codes_list[i] for i in indices]
            group_caches = [request_caches[i] for i in indices]
            group_outputs: list[torch.Tensor] | None = None
            if phase == "prefix":
                group_outputs = self._decode_prefix_batch(group_codes, group_caches)
            elif phase.startswith("suffix:"):
                mode = phase.removeprefix("suffix:")
                suffix_codes = [
                    codes[:, :, int(cache["prefix_frames"]) :]
                    for codes, cache in zip(group_codes, group_caches, strict=True)
                ]
                new_frames = []
                for codes, cache in zip(suffix_codes, group_caches, strict=True):
                    previous = int(cache.get("suffix_frames", cache["suffix_quantized"].shape[-1]))
                    rolling = self.decoder._is_suffix_cache_rolling(
                        previous,
                        int(cache["suffix_quantized"].shape[-1]),
                    )
                    retained = self.prefix_length if rolling else previous
                    new_frames.append(int(codes.shape[-1]) - retained)
                group_outputs = self._decode_suffix_batch(mode, target, suffix_codes, group_caches, new_frames)
                if group_outputs is not None:
                    for index in indices:
                        incremental[index] = True

            if group_outputs is None:
                group_outputs = []
                for index in indices:
                    cache = request_caches[index]
                    codes = codes_list[index]
                    if "ref_wav" not in cache:
                        if int(cache["prefix_frames"]) == 0:
                            group_outputs.append(self.decoder._decode_xvec_first_chunk(codes, cache))
                        else:
                            group_outputs.append(self._decode_icl_prefix(codes, cache))
                    else:
                        suffix_codes = codes[:, :, int(cache["prefix_frames"]) :]
                        group_outputs.append(self._decode_eager(suffix_codes, cache))
                        incremental[index] = True
            for index, output in zip(indices, group_outputs, strict=True):
                outputs[index] = output
                request_caches[index]["_last_output_incremental_audio"] = incremental[index]
                request_caches[index]["_last_output_audio_length"] = int(output.shape[-1])

        return [output for output in outputs if output is not None]

    def batched_chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        lengths: list[int],
        caches: list[dict] | None = None,
        chunk_size: int = 300,
        left_context_size: int = 25,
        max_batch_size: int = 0,
    ) -> torch.Tensor:
        if caches is None:
            from .cuda_graph_decoder_wrapper import _batched_chunked_decode

            return _batched_chunked_decode(
                codes,
                lengths,
                decode_fn=self.decoder._forward_exact,
                total_upsample=self.decoder.total_upsample,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                max_batch_size=max_batch_size,
            )

        if len(caches) != codes.shape[0] or len(lengths) != codes.shape[0]:
            raise ValueError("codes, lengths, and caches must have the same batch size")
        codes_list = [codes[row : row + 1, :, :length] for row, length in enumerate(lengths)]
        outputs = self._batched_request_decode(codes_list, caches)
        max_length = max((int(output.shape[-1]) for output in outputs), default=0)
        padded = codes.new_zeros((len(outputs), 1, max_length), dtype=outputs[0].dtype if outputs else torch.float32)
        for row, output in enumerate(outputs):
            padded[row, :, : output.shape[-1]].copy_(output[0])
        return padded

    def _decode_eager(self, codes: torch.Tensor, caches: dict) -> torch.Tensor:
        for key in (
            "_suffix_quantized_buffers",
            "_suffix_conv_buffers",
            "_suffix_buffer_index",
        ):
            caches.pop(key, None)
        decoder_prefix_frames = int(caches.get("decoder_prefix_frames", caches["prefix_frames"]))
        return self.decoder._decode_cached_incremental(codes, caches, decoder_prefix_frames)

    def _decode(
        self,
        codes: torch.Tensor,
        caches: dict,
        *,
        clone_graph_output: bool,
    ) -> tuple[torch.Tensor, bool]:
        if not self.enabled or not self._warmed_up:
            return self._decode_eager(codes, caches), True

        # Inner CUDA graph replay is illegal while an outer stream capture is
        # active (e.g. vLLM's cudagraph_mode=FULL warmup on Stage 1). Fall back
        # to eager in that case so the outer capture can complete. The guard is
        # a no-op at runtime: is_current_stream_capturing() returns False
        # outside the startup capture window, so normal inference still hits
        # the graph fast path.
        if torch.cuda.is_current_stream_capturing():
            return self._decode_eager(codes, caches), True

        if int(codes.shape[0]) != 1:
            return self._decode_eager(codes, caches), True

        suffix_frames = int(codes.shape[-1])
        old_quantized = caches.get("suffix_quantized")
        old_conv = caches.get("suffix_conv")
        if old_quantized is None or old_conv is None:
            return self._decode_eager(codes, caches), True

        self._ensure_suffix_buffers(caches)
        old_quantized = caches["suffix_quantized"]
        old_conv = caches["suffix_conv"]

        previous_suffix_frames = int(caches["suffix_frames"])
        decoder_prefix_frames = int(caches.get("decoder_prefix_frames", caches["prefix_frames"]))
        mode = "xvec" if decoder_prefix_frames == 0 else "icl"
        cached_frames = int(old_quantized.shape[-1])
        rolling = self.decoder._is_suffix_cache_rolling(previous_suffix_frames, cached_frames)
        new_frames = suffix_frames - (self.prefix_length if rolling else previous_suffix_frames)
        target_frames = (
            self.prefix_length + self.codec_chunk_frames
            if rolling
            else next(
                (
                    target
                    for target, source in self._transitions_for_mode(mode).items()
                    if source == previous_suffix_frames
                ),
                None,
            )
        )
        graph_key = (mode, int(codes.shape[0]), target_frames) if target_frames is not None else None
        if graph_key is None or not 0 < new_frames <= self.codec_chunk_frames or graph_key not in self.combined_graphs:
            return self._decode_eager(codes, caches), True

        request_kv = caches["past_key_values"]
        expected_kv_length = 0 if mode == "xvec" else self.prefix_length
        if request_kv.get_seq_length() != expected_kv_length:
            logger.warning_once(
                "CUDA Graph decoder expected a prefix cache with logical length "
                "%d, got %d; falling back to eager decoding",
                expected_kv_length,
                request_kv.get_seq_length(),
            )
            return self._decode_eager(codes, caches), True

        new_codes = codes[:, :, -new_frames:]
        xvec_boundary = self._get_xvec_boundary(caches, previous_suffix_frames, new_frames) if mode == "xvec" else None
        static_codes = self.combined_static_codes[graph_key]
        if new_frames == self.codec_chunk_frames:
            static_codes.copy_(new_codes)
        else:
            static_codes.zero_()
            static_codes[:, :, :new_frames].copy_(new_codes)

        self.combined_static_quantized[graph_key].copy_(old_quantized)
        self.combined_static_conv[graph_key].copy_(old_conv)
        static_mask = self.combined_static_masks[graph_key]
        static_mask.fill_(1)
        if mode == "icl":
            static_mask[:, : int(caches.get("prefix_pad_frames", 0))] = 0

        static_caches = self.combined_static_caches[graph_key]
        if caches is not static_caches:
            self._copy_caches(caches, static_caches)
        self.combined_graphs[graph_key].replay()

        output, graph_next_quantized, graph_next_conv = self.combined_static_outputs[graph_key]
        if new_frames == self.codec_chunk_frames:
            current_index = int(caches["_suffix_buffer_index"])
            next_index = 1 - current_index
            next_quantized = caches["_suffix_quantized_buffers"][next_index]
            next_conv = caches["_suffix_conv_buffers"][next_index]
            quantized_valid = int(graph_next_quantized.shape[-1])
            conv_valid = int(graph_next_conv.shape[1])
            next_quantized[:, :, :quantized_valid].copy_(graph_next_quantized)
            next_conv[:, :conv_valid, :].copy_(graph_next_conv)
            caches["_suffix_buffer_index"] = next_index
            caches["suffix_frames"] = target_frames
            caches["suffix_quantized"] = next_quantized[:, :, :quantized_valid]
            caches["suffix_conv"] = next_conv[:, :conv_valid, :]
        if xvec_boundary is not None:
            caches["ref_hidden"] = xvec_boundary

        actual_suffix_frames = (self.prefix_length if rolling else previous_suffix_frames) + new_frames
        output = self._trim_replay_output(output, actual_suffix_frames, target_frames)
        if clone_graph_output:
            return output.clone(), True
        return output, True

    def _trim_replay_output(self, static_output: torch.Tensor, actual_size: int, padded_size: int) -> torch.Tensor:
        """Trim a graph/compiled replay output to the eager-equivalent length.

        The captured ``static_output`` already reflects the decoder's TRUE output
        length for ``padded_size`` frames, which for causal decoders (e.g.
        Qwen3-Omni Code2Wav) is shorter than ``padded_size * total_upsample`` by a
        fixed amount. Each zero-padded frame beyond ``actual_size`` contributes
        ``total_upsample`` trailing samples that are stale buffer content, so trim
        relative to the captured length instead of the nominal length. This is a
        no-op for decoders whose output equals the nominal length (e.g. Qwen3-TTS).
        """
        drop = (padded_size - actual_size) * self.decoder.total_upsample
        actual_out_len = max(0, static_output.shape[-1] - drop)
        return static_output[..., :actual_out_len]

    def decode(self, codes: torch.Tensor, caches: dict) -> torch.Tensor:
        output, _ = self._decode(codes, caches, clone_graph_output=True)
        return output

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        caches: dict | None = None,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample
        incremental_flags: list[bool] = []

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            incremental_output = False
            if caches is None or codes_chunk.shape[-1] == 4:
                wav_chunk = self.decoder._forward_exact(codes_chunk)
            elif "ref_wav" not in caches:
                wav_chunk = self._decode_icl_prefix(codes_chunk, caches)
            else:
                actual_prefix_frames = int(caches.get("prefix_frames", self.prefix_length))
                codes_chunk = codes_chunk[:, :, actual_prefix_frames:]
                wav_chunk, incremental_output = self._decode(
                    codes_chunk,
                    caches,
                    clone_graph_output=False,
                )

            # Keep origin/main's concat semantics: Qwen3-Omni can return a chunk
            # that is shorter than the nominal code_len * total_upsample length.
            # Clone each slice because graph outputs are static buffers that later
            # replays may overwrite.
            if incremental_output:
                wavs.append(wav_chunk.clone())
            else:
                wavs.append(wav_chunk[..., context_size * total_upsample :].clone())
            incremental_flags.append(incremental_output)
            start_index = end_index

        if not wavs:
            return self.decoder._forward_exact(codes)
        output = torch.cat(wavs, dim=-1)
        if caches is not None:
            caches["_last_output_incremental_audio"] = bool(incremental_flags) and all(incremental_flags)
        if output.is_cuda and not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(output.device).synchronize()
        return output
