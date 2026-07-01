# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import os
from collections import Counter
from collections.abc import Callable, Sequence

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.cuda_graph_wrapper import BaseCUDAGraphWrapper

logger = init_logger(__name__)


def _normalize_decode_lengths(lengths: Sequence[int], batch_size: int, max_len: int) -> tuple[int, ...]:
    if len(lengths) != batch_size:
        raise ValueError(f"Expected {batch_size} decode lengths, got {len(lengths)}")

    normalized: list[int] = []
    for length in lengths:
        length_int = int(length)
        if length_int < 0 or length_int > max_len:
            raise ValueError(f"Invalid decode length {length_int}; expected 0 <= length <= {max_len}")
        normalized.append(length_int)
    return tuple(normalized)


def _batched_chunked_decode(
    codes: torch.Tensor,
    lengths: Sequence[int],
    *,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    total_upsample: int,
    chunk_size: int = 300,
    left_context_size: int = 25,
    max_batch_size: int = 0,
) -> torch.Tensor:
    """Decode a padded batch by grouping same-round chunks across requests.

    Assumes ``decode_fn`` returns exactly ``chunk_len * total_upsample`` samples per
    chunk (causal-conv trim constant C = 0, e.g. Qwen3-TTS) and raises otherwise. It
    is not used by short-output decoders such as Qwen3-Omni Code2Wav (C = 555), which
    go through the per-request ``chunked_decode`` / ``chunked_decode_streaming`` paths.
    """
    if codes.dim() < 3:
        raise ValueError(f"Expected codes with shape [B, Q, F], got {tuple(codes.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if left_context_size < 0:
        raise ValueError(f"left_context_size must be non-negative, got {left_context_size}")
    if max_batch_size < 0:
        raise ValueError(f"max_batch_size must be non-negative, got {max_batch_size}")

    batch_size = int(codes.shape[0])
    max_input_len = int(codes.shape[-1])
    length_values = _normalize_decode_lengths(lengths, batch_size, max_input_len)
    max_decode_len = max(length_values, default=0)
    if max_decode_len == 0:
        return torch.empty((batch_size, 1, 0), dtype=torch.float32, device=codes.device)

    total_upsample = int(total_upsample)
    wav_out: torch.Tensor | None = None
    num_rounds = (max_decode_len + chunk_size - 1) // chunk_size

    for round_index in range(num_rounds):
        start_index = round_index * chunk_size
        grouped_jobs: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
        for req_index, total_len in enumerate(length_values):
            if start_index >= total_len:
                continue
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            input_start = start_index - context_size
            input_end = end_index
            chunk_len = input_end - input_start
            grouped_jobs.setdefault(chunk_len, []).append(
                (req_index, input_start, input_end, start_index, end_index, context_size)
            )

        for jobs in grouped_jobs.values():
            job_batches = (
                [jobs]
                if max_batch_size <= 0 or len(jobs) <= max_batch_size
                else [jobs[start : start + max_batch_size] for start in range(0, len(jobs), max_batch_size)]
            )
            for job_batch in job_batches:
                chunk_rows = [
                    codes[req_index, :, input_start:input_end]
                    for req_index, input_start, input_end, _, _, _ in job_batch
                ]
                codes_chunk = torch.stack(
                    chunk_rows,
                    dim=0,
                )
                wav_chunk = decode_fn(codes_chunk)
                if wav_chunk.shape[0] != len(job_batch):
                    raise ValueError(
                        f"Decoder returned batch size {wav_chunk.shape[0]} for input batch size {len(job_batch)}"
                    )
                if wav_out is None:
                    wav_out = torch.empty(
                        (batch_size, *wav_chunk.shape[1:-1], max_decode_len * total_upsample),
                        dtype=wav_chunk.dtype,
                        device=wav_chunk.device,
                    )

                for row, (req_index, _, _, chunk_start, chunk_end, context_size) in enumerate(job_batch):
                    src_start = context_size * total_upsample
                    dst_start = chunk_start * total_upsample
                    dst_end = chunk_end * total_upsample
                    src_end = src_start + (dst_end - dst_start)
                    if src_end > wav_chunk.shape[-1]:
                        raise ValueError(
                            f"Decoder returned too-short chunk output: need {src_end}, got {wav_chunk.shape[-1]}"
                        )
                    wav_out[req_index, ..., dst_start:dst_end].copy_(wav_chunk[row, ..., src_start:src_end])

    if wav_out is None:
        return torch.empty((batch_size, 1, 0), dtype=torch.float32, device=codes.device)
    return wav_out


class Qwen3DecoderGraph(BaseCUDAGraphWrapper[tuple[int, int]]):
    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        capture_batch_sizes: list[int] | None = None,
        extra_capture_shapes: list[tuple[int, int]] | None = None,
        num_quantizers: int = 8,
        vllm_config: VllmConfig | None = None,
    ):
        super().__init__(runnable=decoder, vllm_config=vllm_config)
        self.decoder = decoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        self.capture_batch_sizes = sorted(set(capture_batch_sizes or [1]))
        self.extra_capture_shapes = sorted(
            {
                (int(batch_size), int(size))
                for batch_size, size in extra_capture_shapes or []
                if int(batch_size) > 0 and int(size) > 0
            }
        )
        self._bucket_sizes = self.capture_sizes
        self.num_quantizers = num_quantizers

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> list[int]:
        """Compute capture sizes from chunking config for high graph hit rate."""
        sizes: set[int] = set()

        # Streaming exact hits
        if codec_chunk_frames > 0:
            sizes.add(codec_chunk_frames)
            if codec_left_context_frames > 0:
                sizes.add(codec_chunk_frames + codec_left_context_frames)

        # Non-streaming chunked decode: full chunk + last-chunk buckets
        non_stream_max = decode_chunk_size + decode_left_context
        sizes.add(non_stream_max)

        # Power-of-2 buckets covering both streaming IC sizes and non-streaming last-chunk sizes
        for p2 in [2, 4, 8, 16, 32, 64, 128, 256]:
            if p2 <= non_stream_max:
                sizes.add(p2)

        return sorted(sizes)

    def prepare_capture_context(
        self,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ):
        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
                decode_chunk_size=decode_chunk_size,
                decode_left_context=decode_left_context,
            )

        self.capture_batch_sizes = [bs for bs in self.capture_batch_sizes if bs > 0] or [1]
        self._bucket_sizes = sorted(set(self.capture_sizes) | {size for _, size in self.extra_capture_shapes})

    def before_capture(self, keys):
        for key in keys:
            static_args, static_kwargs = self.get_static_call_args(key)
            with torch.no_grad():
                self.runnable(*static_args, **static_kwargs)
        torch.accelerator.synchronize(self.device)

    def get_capture_keys(self) -> list[tuple[int, int]]:
        shapes = {
            (batch_size, bucket_size) for batch_size in self.capture_batch_sizes for bucket_size in self.capture_sizes
        }
        shapes.update(self.extra_capture_shapes)
        return sorted(shapes)

    def get_static_call_args(self, key, *args, **kwargs):
        if key not in self.static_inputs:
            batch_size, bucket_size = key
            self.static_inputs[key] = torch.zeros(
                batch_size,
                self.num_quantizers,
                bucket_size,
                dtype=self.dtype,
                device=self.device,
            )
        return (self.static_inputs[key],), {}

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ):
        self.capture(
            device=device,
            dtype=dtype,
            codec_chunk_frames=codec_chunk_frames,
            codec_left_context_frames=codec_left_context_frames,
            decode_chunk_size=decode_chunk_size,
            decode_left_context=decode_left_context,
        )

    def select_runtime_key(self, codes: torch.Tensor, *args, **kwargs) -> tuple[int, int] | None:
        batch_size = int(codes.shape[0])
        actual_size = int(codes.shape[-1])
        bucket_size = self.find_ge_bucket(self._bucket_sizes, actual_size)
        return (batch_size, bucket_size) if bucket_size is not None else None

    def on_replay_fallback_log(self, reason, key, codes: torch.Tensor, *args, **kwargs) -> None:
        logger.debug(
            "Qwen3 decoder CUDA Graph fallback: reason=%s key=%s batch=%d frames=%d buckets=%s",
            reason,
            key,
            int(codes.shape[0]),
            int(codes.shape[-1]),
            self._bucket_sizes,
        )

    def prepare_runtime_input(self, key: tuple[int, int], codes: torch.Tensor, *args, **kwargs) -> None:
        _, bucket_size = key
        actual_size = int(codes.shape[-1])
        static_input = self.static_inputs[key]
        if actual_size == bucket_size:
            static_input.copy_(codes)
        else:
            static_input.zero_()
            static_input[:, :, :actual_size] = codes

    def postprocess(
        self,
        key: tuple[int, int],
        codes: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Trim a graph/compiled replay output to the eager-equivalent length.

        The captured ``static_output`` already reflects the decoder's TRUE output
        length for ``bucket_size`` frames, which for causal decoders (e.g.
        Qwen3-Omni Code2Wav) is shorter than ``padded_size * total_upsample`` by a
        fixed amount. Each zero-padded frame beyond ``actual_size`` contributes
        ``total_upsample`` trailing samples that are stale buffer content, so trim
        relative to the captured length instead of the nominal length. This is a
        no-op for decoders whose output equals the nominal length (e.g. Qwen3-TTS).
        """
        actual_size = int(codes.shape[-1])
        bucket_size = key[1]
        output = self.trim_captured_minus_padding(
            self.static_outputs[key], actual_size, bucket_size, self.decoder.total_upsample
        )
        return output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.replay(codes)


class Qwen3CompiledDecoderGraph(Qwen3DecoderGraph):
    def __init__(
        self,
        decoder,
        compile_shapes: list[tuple[int, int]] | None = None,
        num_quantizers: int = 8,
        vllm_config: VllmConfig | None = None,
    ):
        super().__init__(
            decoder,
            capture_sizes=[],
            num_quantizers=num_quantizers,
            vllm_config=vllm_config,
        )
        self.compile_shapes = sorted(compile_shapes or [])
        self._bucket_sizes = sorted({size for _, size in self.compile_shapes})
        self.num_warmup = 5
        self.runnable = None

    def prepare_capture_context(self):
        logger.info(
            "Starting torch.compile + CUDA Graph warmup for decoder shapes: %s",
            self.compile_shapes,
        )
        try:
            self.runnable = torch.compile(
                self.decoder.forward,
                mode="default",
                fullgraph=False,
                dynamic=False,
            )
        except Exception:
            logger.warning("Failed to create torch.compile decoder wrapper", exc_info=True)
            self.enabled = False

    def warmup(self, device: torch.device, dtype: torch.dtype = torch.long):
        self.capture(device=device, dtype=dtype)

    def get_capture_keys(self):
        return self.compile_shapes

    def select_runtime_key(self, codes: torch.Tensor, *args, **kwargs) -> tuple[int, int] | None:
        batch_size = int(codes.shape[0])
        actual_size = int(codes.shape[-1])
        exact_key = (batch_size, actual_size)
        if exact_key in self.graphs:
            return exact_key
        bucket_key = super().select_runtime_key(codes, *args, **kwargs)
        if bucket_key in self.graphs:
            return bucket_key
        return None

    def on_replay_fallback_log(self, reason, key, codes: torch.Tensor, *args, **kwargs) -> None:
        logger.debug(
            "Qwen3 compiled decoder CUDA Graph fallback: reason=%s key=%s batch=%d frames=%d buckets=%s",
            reason,
            key,
            int(codes.shape[0]),
            int(codes.shape[-1]),
            self._bucket_sizes,
        )

    def before_capture(self, keys):
        pass

    def run_eager(self, codes: torch.Tensor, *args, **kwargs):
        return None

    def try_decode(self, codes: torch.Tensor) -> torch.Tensor | None:
        return self.replay(codes)


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(decoder, capture_sizes=[25, 50, 100, 200, 300])
        wrapper.warmup(device)

        # During inference:
        output = wrapper.decode(codes)  # Automatically uses CUDA graph if possible
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        capture_batch_sizes: list[int] | None = None,
        extra_capture_shapes: list[tuple[int, int]] | None = None,
        compile_shapes: list[tuple[int, int]] | None = None,
        num_quantizers: int = 8,
        vllm_config: VllmConfig | None = None,
    ):
        self.decoder = decoder
        self.normal_graph = Qwen3DecoderGraph(
            decoder=decoder,
            capture_sizes=capture_sizes,
            capture_batch_sizes=capture_batch_sizes,
            extra_capture_shapes=extra_capture_shapes,
            num_quantizers=num_quantizers,
            vllm_config=vllm_config,
        )
        self.compiled_graph = (
            Qwen3CompiledDecoderGraph(
                decoder=decoder,
                compile_shapes=compile_shapes,
                num_quantizers=num_quantizers,
                vllm_config=vllm_config,
            )
            if compile_shapes
            else None
        )
        self._stats_enabled = os.environ.get("VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._stats_log_every = int(os.environ.get("VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS_LOG_EVERY", "0") or 0)
        self._stats_total = 0
        self._stats_hits = 0
        self._stats_compiled_hits = 0
        self._stats_fallbacks = 0
        self._stats_stream_capture_fallbacks = 0
        self._stats_requests: Counter[tuple[int, int]] = Counter()
        self._stats_hit_shapes: Counter[tuple[int, int, int]] = Counter()
        self._stats_compiled_shapes: Counter[tuple[int, int]] = Counter()
        self._stats_fallback_shapes: Counter[tuple[int, int, int]] = Counter()

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> list[int]:
        return Qwen3DecoderGraph.compute_capture_sizes(
            codec_chunk_frames=codec_chunk_frames,
            codec_left_context_frames=codec_left_context_frames,
            decode_chunk_size=decode_chunk_size,
            decode_left_context=decode_left_context,
        )

    @property
    def enabled(self) -> bool:
        return self.normal_graph.enabled

    @property
    def capture_batch_sizes(self):
        return self.normal_graph.capture_batch_sizes

    @property
    def capture_sizes(self):
        return self.normal_graph.capture_sizes

    @property
    def compile_shapes(self):
        return self.compiled_graph.compile_shapes if self.compiled_graph is not None else []

    @property
    def _warmed_up(self) -> bool:
        return self.normal_graph._warmed_up

    @property
    def extra_capture_shapes(self):
        return self.normal_graph.extra_capture_shapes

    def _record_decode_stats(
        self,
        *,
        hit: bool,
        batch_size: int,
        actual_size: int,
        padded_size: int | None,
        stream_capture: bool = False,
        compiled: bool = False,
    ) -> None:
        if not self._stats_enabled:
            return

        padded_key = int(padded_size) if padded_size is not None else -1
        self._stats_total += 1
        self._stats_requests[(batch_size, actual_size)] += 1
        if hit:
            self._stats_hits += 1
            if compiled:
                self._stats_compiled_hits += 1
                self._stats_compiled_shapes[(batch_size, actual_size, padded_key)] += 1
            else:
                self._stats_hit_shapes[(batch_size, actual_size, padded_key)] += 1
        else:
            self._stats_fallbacks += 1
            self._stats_fallback_shapes[(batch_size, actual_size, padded_key)] += 1
            if stream_capture:
                self._stats_stream_capture_fallbacks += 1

        if self._stats_log_every > 0 and self._stats_total % self._stats_log_every == 0:
            self.log_decode_stats()

    def log_decode_stats(self) -> None:
        if not self._stats_enabled or self._stats_total == 0:
            return
        hit_rate = 100.0 * self._stats_hits / self._stats_total
        logger.info(
            "Code2Wav CUDA Graph stats: total=%d hits=%d fallbacks=%d "
            "compiled_hits=%d stream_capture_fallbacks=%d hit_rate=%.2f%% "
            "top_requests=%s top_compiled=%s top_hits=%s top_fallbacks=%s",
            self._stats_total,
            self._stats_hits,
            self._stats_fallbacks,
            self._stats_compiled_hits,
            self._stats_stream_capture_fallbacks,
            hit_rate,
            self._stats_requests.most_common(12),
            self._stats_compiled_shapes.most_common(12),
            self._stats_hit_shapes.most_common(12),
            self._stats_fallback_shapes.most_common(12),
        )

    def _decode_stats_shape(self, codes: torch.Tensor) -> tuple[int, int, int | None]:
        batch_size = int(codes.shape[0])
        actual_size = int(codes.shape[-1])

        if self.compiled_graph is not None and self.compiled_graph.graphs:
            exact_key = (batch_size, actual_size)
            if exact_key in self.compiled_graph.graphs:
                return batch_size, actual_size, actual_size

            padded_size = self.compiled_graph.find_ge_bucket(self.compiled_graph._bucket_sizes, actual_size)
            if padded_size is not None and (batch_size, padded_size) in self.compiled_graph.graphs:
                return batch_size, actual_size, padded_size

        padded_size = self.normal_graph.find_ge_bucket(self.normal_graph._bucket_sizes, actual_size)
        return batch_size, actual_size, padded_size

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> None:
        self.normal_graph.warmup(
            device=device,
            dtype=dtype,
            codec_chunk_frames=codec_chunk_frames,
            codec_left_context_frames=codec_left_context_frames,
            decode_chunk_size=decode_chunk_size,
            decode_left_context=decode_left_context,
        )
        if self.compiled_graph is not None:
            self.compiled_graph.warmup(device, dtype)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        stream_capture = torch.cuda.is_current_stream_capturing()
        if stream_capture:
            self._record_decode_stats(
                hit=False,
                batch_size=int(codes.shape[0]),
                actual_size=int(codes.shape[-1]),
                padded_size=None,
                stream_capture=True,
            )
            return self.normal_graph.run_eager(codes)

        batch_size, actual_size, padded_size = self._decode_stats_shape(codes)

        if self.compiled_graph is not None:
            compiled_output = self.compiled_graph.try_decode(codes)
            if compiled_output is not None:
                self._record_decode_stats(
                    hit=True,
                    batch_size=batch_size,
                    actual_size=actual_size,
                    padded_size=padded_size,
                    compiled=True,
                )
                return compiled_output

        graph_key = (batch_size, padded_size) if padded_size is not None else None
        graph_hit = (
            self.normal_graph.can_replay(codes) and graph_key is not None and graph_key in self.normal_graph.graphs
        )
        self._record_decode_stats(
            hit=graph_hit,
            batch_size=batch_size,
            actual_size=actual_size,
            padded_size=padded_size,
        )
        return self.normal_graph.decode(codes)

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
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
            wav_chunk = self.decode(codes_chunk)

            # Keep origin/main's concat semantics: Qwen3-Omni can return a chunk
            # that is shorter than the nominal code_len * total_upsample length.
            # decode() already returns a clone, so this slice is safe to keep.
            wavs.append(wav_chunk[..., context_size * total_upsample :])
            start_index = end_index

        if not wavs:
            return self.decoder(codes)
        return torch.cat(wavs, dim=-1)

    def batched_chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        lengths: Sequence[int],
        chunk_size: int = 300,
        left_context_size: int = 25,
        max_batch_size: int = 0,
    ) -> torch.Tensor:
        return _batched_chunked_decode(
            codes,
            lengths,
            decode_fn=self.decode,
            total_upsample=self.decoder.total_upsample,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            max_batch_size=max_batch_size,
        )
