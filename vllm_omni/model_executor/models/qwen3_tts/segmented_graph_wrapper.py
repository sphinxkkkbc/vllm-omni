# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import bisect

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
    ):
        self.decoder = decoder
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        self.capture_batch_sizes = sorted(set(capture_batch_sizes or [1]))
        self._bucket_sizes = self.capture_sizes
        self.num_quantizers = num_quantizers
        self.enabled = enabled
        self._warmed_up = False

        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_caches: dict = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self.prefix_graph: CUDAGraph | None = None
        self.prefix_static_input: torch.Tensor | None = None
        self.prefix_static_output: torch.Tensor | None = None

        self._device = None
        self.prefix_length = 72
        self.suffix_length = [26, 51, 76, 97]
        self._bucket_sizes = self.suffix_length

    def _get_padded_size(self, actual_size: int) -> int | None:
        # bisect_left over the pre-sorted _bucket_sizes is O(log n) vs the linear scan;
        # this matters because _decode invokes it on every per-chunk replay.
        idx = bisect.bisect_left(self._bucket_sizes, actual_size)
        if idx < len(self._bucket_sizes):
            return self._bucket_sizes[idx]
        return None

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
            raise ValueError("capture_sizes must be provided")

        # self.capture_batch_sizes = [bs for bs in self.capture_batch_sizes if bs > 0]
        self.capture_batch_sizes = [1]
        self.capture_sizes = self.suffix_length
        capture_shapes = self._get_capture_shapes()

        try:
            self._capture_icl_prefix(device, dtype)
            logger.info("  Captured CUDA Graph for first chunk")
        except Exception:
            logger.warning("  Failed to capture CUDA Graph for first chunk", exc_info=True)

        if not self.static_caches:
            model_dtype = next(self.decoder.parameters()).dtype
            self.static_caches = self._make_dummy_cache(
                self.capture_batch_sizes[0],
                device,
                model_dtype,
            )

        logger.info(
            "Starting CUDA Graph warmup for %d shapes: batch_sizes=%s seq_lens=%s",
            len(capture_shapes),
            self.capture_batch_sizes,
            self.capture_sizes,
        )

        # Warmup runs to ensure CUDA memory is allocated
        for batch_size, size in capture_shapes:
            dummy = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
            with torch.no_grad():
                _ = self.decoder._decode_cached(dummy, self.static_caches, self.prefix_length)

        torch.accelerator.synchronize(device)

        for batch_size, size in capture_shapes:
            try:
                self._capture(batch_size, self.static_caches, size, device, dtype)
                logger.info("  Captured CUDA Graph for batch=%d size=%d", batch_size, size)
            except Exception:
                logger.warning("  Failed to capture graph for batch=%d size=%d", batch_size, size, exc_info=True)

        self._warmed_up = self.prefix_graph is not None or bool(self.graphs)

    def _capture(self, batch_size: int, caches: dict, size: int, device: torch.device, dtype: torch.dtype):
        key = (batch_size, size)
        static_input = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.decoder._decode_cached(static_input, caches, self.prefix_length)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder._decode_cached(static_input, caches, self.prefix_length)

        self.graphs[key] = graph
        self.static_inputs[key] = static_input
        self.static_outputs[key] = static_output

    def _capture_icl_prefix(self, device: torch.device, dtype: torch.dtype):
        size = self.prefix_length + 1
        caches: dict = {}
        static_input = torch.zeros(1, self.num_quantizers, size, dtype=dtype, device=device)
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
            batch_size=1,
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=model_dtype,
            device=device,
        )

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder._decode_icl_first_chunk(
                    static_input,
                    caches,
                    self.prefix_length,
                    prefix_cache=prefix_cache,
                )

        self.prefix_graph = graph
        self.prefix_static_input = static_input
        self.prefix_static_output = static_output
        self.static_caches = caches

    def _decode_icl_prefix(self, codes: torch.Tensor, caches: dict) -> torch.Tensor:
        if (
            self.prefix_graph is None
            or self.prefix_static_input is None
            or self.prefix_static_output is None
            or codes.shape[0] != 1
            or codes.shape[-1] != self.prefix_length + 1
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.decoder._decode_icl_first_chunk(codes, caches, self.prefix_length)

        self.prefix_static_input.copy_(codes)
        self.prefix_graph.replay()
        caches.update(self.static_caches)
        return self.prefix_static_output

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

    def _decode(self, codes: torch.Tensor, caches: dict, *, clone_graph_output: bool) -> torch.Tensor:
        if not self.enabled or not self._warmed_up:
            return self.decoder._decode_cached(codes, caches, self.prefix_length)

        # Inner CUDA graph replay is illegal while an outer stream capture is
        # active (e.g. vLLM's cudagraph_mode=FULL warmup on Stage 1). Fall back
        # to eager in that case so the outer capture can complete. The guard is
        # a no-op at runtime: is_current_stream_capturing() returns False
        # outside the startup capture window, so normal inference still hits
        # the graph fast path.
        if torch.cuda.is_current_stream_capturing():
            return self.decoder._decode_cached(codes, caches, self.prefix_length)

        batch_size = int(codes.shape[0])
        actual_size = int(codes.shape[-1])
        padded_size = self._get_padded_size(actual_size)
        graph_key = (batch_size, padded_size) if padded_size is not None else None

        if graph_key is None or graph_key not in self.graphs:
            return self.decoder._decode_cached(codes, caches, self.prefix_length)

        request_kv = caches["past_key_values"]
        if request_kv.get_seq_length() != self.prefix_length:
            logger.warning_once(
                "CUDA Graph decoder expected a prefix cache with logical length "
                "%d, got %d; falling back to eager decoding",
                self.prefix_length,
                request_kv.get_seq_length(),
            )
            return self.decoder._decode_cached(codes, caches, self.prefix_length)

        static_input = self.static_inputs[graph_key]

        if actual_size == padded_size:
            static_input.copy_(codes)
        else:
            static_input.zero_()
            static_input[:, :, :actual_size] = codes
        if caches is not self.static_caches:
            self._copy_caches(caches, self.static_caches)
        self.graphs[graph_key].replay()

        output = self._trim_replay_output(self.static_outputs[graph_key], actual_size, padded_size)
        if clone_graph_output:
            return output.clone()
        return output

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
        return self._decode(codes, caches, clone_graph_output=True)

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

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            if caches is None or codes_chunk.shape[-1] == 4:
                wav_chunk = self.decoder._forward_exact(codes_chunk)
            elif int(caches.get("prefix_frames", self.prefix_length)) != self.prefix_length:
                wav_chunk = self.decoder(codes_chunk, caches)
            elif "ref_wav" not in caches:
                wav_chunk = self._decode_icl_prefix(codes_chunk, caches)
            else:
                codes_chunk = codes_chunk[:, :, self.prefix_length :]
                active_caches = self.static_caches if self.prefix_graph is not None else caches
                wav_chunk = self._decode(codes_chunk, active_caches, clone_graph_output=False)

            # Keep origin/main's concat semantics: Qwen3-Omni can return a chunk
            # that is shorter than the nominal code_len * total_upsample length.
            # Clone each slice because graph outputs are static buffers that later
            # replays may overwrite.
            wavs.append(wav_chunk[..., context_size * total_upsample :].clone())
            start_index = end_index

        if not wavs:
            return self.decoder._forward_exact(codes)
        output = torch.cat(wavs, dim=-1)
        if output.is_cuda and not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(output.device).synchronize()
        return output
