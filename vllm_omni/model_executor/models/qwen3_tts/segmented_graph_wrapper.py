# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

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
        self.num_quantizers = num_quantizers
        self.enabled = enabled
        self._warmed_up = False

        self.static_caches: dict = {}
        self.combined_graphs: dict[int, CUDAGraph] = {}
        self.combined_static_codes: dict[int, torch.Tensor] = {}
        self.combined_static_quantized: dict[int, torch.Tensor] = {}
        self.combined_static_conv: dict[int, torch.Tensor] = {}
        self.combined_static_masks: dict[int, torch.Tensor] = {}
        self.combined_static_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.prefix_graph: CUDAGraph | None = None
        self.prefix_static_input: torch.Tensor | None = None
        self.prefix_static_output: torch.Tensor | None = None
        self.prefix_suffix_quantized: torch.Tensor | None = None
        self.prefix_suffix_conv: torch.Tensor | None = None
        self.prefix_attention_mask: torch.Tensor | None = None
        self.suffix_attention_mask: torch.Tensor | None = None

        self._device = None
        self.prefix_length = 72
        self.suffix_length = [26, 51, 76, 97]

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

        for _, size in capture_shapes:
            try:
                self._capture_combined_suffix(size, self.static_caches, device, dtype)
                logger.info("  Captured combined suffix CUDA Graph for size=%d", size)
            except Exception:
                logger.warning("  Failed to capture combined suffix graph for size=%d", size, exc_info=True)

        self._warmed_up = self.prefix_graph is not None or bool(self.combined_graphs)

    def _run_combined_suffix(
        self,
        codes: torch.Tensor,
        old_quantized: torch.Tensor,
        old_conv: torch.Tensor,
        caches: dict,
        target_frames: int,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 97}
        return self.decoder._decode_suffix_incremental(
            codes,
            old_quantized,
            old_conv,
            caches,
            self.prefix_length,
            previous_frames_by_target[target_frames],
            25,
            attention_mask=attention_mask,
        )

    def _capture_combined_suffix(
        self,
        target_frames: int,
        caches: dict,
        device: torch.device,
        code_dtype: torch.dtype,
    ) -> None:
        previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
        previous_frames = previous_frames_by_target[target_frames]
        model_dtype = next(self.decoder.parameters()).dtype
        static_codes = torch.zeros(1, self.num_quantizers, 25, dtype=code_dtype, device=device)
        static_quantized = torch.zeros(
            1,
            self.decoder.config.codebook_dim,
            min(previous_frames, self.prefix_length),
            dtype=model_dtype,
            device=device,
        )
        static_conv = torch.zeros(
            1,
            min(previous_frames, self.prefix_length - 2),
            self.decoder.config.latent_dim,
            dtype=model_dtype,
            device=device,
        )
        static_mask = torch.ones(1, self.prefix_length + target_frames, dtype=torch.bool, device=device)

        with torch.no_grad():
            _ = self._run_combined_suffix(
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
                    static_codes,
                    static_quantized,
                    static_conv,
                    caches,
                    target_frames,
                    static_mask,
                )

        self.combined_graphs[target_frames] = graph
        self.combined_static_codes[target_frames] = static_codes
        self.combined_static_quantized[target_frames] = static_quantized
        self.combined_static_conv[target_frames] = static_conv
        self.combined_static_masks[target_frames] = static_mask
        self.combined_static_outputs[target_frames] = outputs

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
        suffix_quantized = torch.zeros(
            1,
            config.codebook_dim,
            self.prefix_length,
            dtype=model_dtype,
            device=device,
        )
        suffix_conv = torch.zeros(
            1,
            self.prefix_length - 2,
            config.latent_dim,
            dtype=model_dtype,
            device=device,
        )
        prefix_hidden_mask = torch.ones(1, 1, size, dtype=model_dtype, device=device)
        prefix_attention_mask = torch.ones(1, self.prefix_length, dtype=torch.bool, device=device)
        suffix_attention_mask = torch.ones(1, size, dtype=torch.bool, device=device)

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
                suffix_quantized[:, :, :1].copy_(caches["suffix_quantized"])
                suffix_conv[:, :1, :].copy_(caches["suffix_conv"])

        self.prefix_graph = graph
        self.prefix_static_input = static_input
        self.prefix_static_output = static_output
        self.prefix_suffix_quantized = suffix_quantized
        self.prefix_suffix_conv = suffix_conv
        self.prefix_hidden_mask = prefix_hidden_mask
        self.prefix_attention_mask = prefix_attention_mask
        self.suffix_attention_mask = suffix_attention_mask
        self.static_caches = caches

    def _decode_icl_prefix(self, codes: torch.Tensor, caches: dict) -> torch.Tensor:
        if (
            self.prefix_graph is None
            or self.prefix_static_input is None
            or self.prefix_static_output is None
            or self.prefix_suffix_quantized is None
            or self.prefix_suffix_conv is None
            or self.prefix_hidden_mask is None
            or self.prefix_attention_mask is None
            or self.suffix_attention_mask is None
            or codes.shape[0] != 1
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.decoder._decode_icl_first_chunk(codes, caches, int(caches["prefix_frames"]))

        actual_prefix_frames = int(caches["prefix_frames"])
        suffix_frames = int(codes.shape[-1]) - actual_prefix_frames
        if not 0 < actual_prefix_frames <= self.prefix_length or suffix_frames != 1:
            return self.decoder._decode_icl_first_chunk(codes, caches, actual_prefix_frames)

        prefix_pad_frames = self.prefix_length - actual_prefix_frames
        self.prefix_static_input.zero_()
        self.prefix_static_input[:, :, prefix_pad_frames : self.prefix_length].copy_(codes[:, :, :actual_prefix_frames])
        self.prefix_static_input[:, :, self.prefix_length :].copy_(codes[:, :, actual_prefix_frames:])
        self.prefix_hidden_mask.fill_(1)
        self.prefix_hidden_mask[:, :, :prefix_pad_frames] = 0
        self.prefix_attention_mask.fill_(1)
        self.prefix_attention_mask[:, :prefix_pad_frames] = 0
        self.suffix_attention_mask.fill_(1)
        self.suffix_attention_mask[:, :prefix_pad_frames] = 0

        self.prefix_graph.replay()
        caches.update(self.static_caches)
        caches["prefix_pad_frames"] = prefix_pad_frames
        caches["suffix_quantized"] = self.prefix_suffix_quantized[:, :, :1]
        caches["suffix_conv"] = self.prefix_suffix_conv[:, :1, :]
        self._ensure_suffix_buffers(caches)
        return self.prefix_static_output[..., prefix_pad_frames * self.decoder.total_upsample :]

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
        cached_frames = int(old_quantized.shape[-1])
        rolling = previous_suffix_frames >= 76 and cached_frames == self.prefix_length
        new_frames = suffix_frames - (self.prefix_length if rolling else previous_suffix_frames)
        target_frames_by_previous = {1: 26, 26: 51, 51: 76}
        target_frames = 97 if rolling else target_frames_by_previous.get(previous_suffix_frames)
        if target_frames is None or not 0 < new_frames <= 25 or target_frames not in self.combined_graphs:
            return self._decode_eager(codes, caches), True

        request_kv = caches["past_key_values"]
        if request_kv.get_seq_length() != self.prefix_length:
            logger.warning_once(
                "CUDA Graph decoder expected a prefix cache with logical length "
                "%d, got %d; falling back to eager decoding",
                self.prefix_length,
                request_kv.get_seq_length(),
            )
            return self._decode_eager(codes, caches), True

        new_codes = codes[:, :, -new_frames:]
        static_codes = self.combined_static_codes[target_frames]
        if new_frames == 25:
            static_codes.copy_(new_codes)
        else:
            static_codes.zero_()
            static_codes[:, :, :new_frames].copy_(new_codes)

        self.combined_static_quantized[target_frames].copy_(old_quantized)
        self.combined_static_conv[target_frames].copy_(old_conv)
        static_mask = self.combined_static_masks[target_frames]
        static_mask.fill_(1)
        static_mask[:, : int(caches.get("prefix_pad_frames", 0))] = 0

        if caches is not self.static_caches:
            self._copy_caches(caches, self.static_caches)
        self.combined_graphs[target_frames].replay()

        output, graph_next_quantized, graph_next_conv = self.combined_static_outputs[target_frames]
        if new_frames == 25:
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
