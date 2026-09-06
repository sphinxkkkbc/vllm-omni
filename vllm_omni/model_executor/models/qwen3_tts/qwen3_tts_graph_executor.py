# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Target-backed Qwen3-TTS decode execution owned by ``Code2Wav``.

The decoder deliberately remains unaware of Targets.  This adapter supplies
the Target callbacks to its shared stateful routing API and performs the
small amount of Qwen-specific static batching needed for graph replay.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    VocoderCUDAGraphTarget,
)
from vllm_omni.model_executor.models.qwen3_tts.stateful_chunking import (
    resolve_stateful_chunk_contract,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    _prepare_icl_prefix_batch,
    _resolve_suffix_batch,
)


class Qwen3TTSGraphExecutor:
    """Run Qwen decoder batches through model-owned Targets when available."""

    def __init__(
        self,
        *,
        decoder: Any,
        stateless_target: VocoderCUDAGraphTarget,
        icl_prefix_target: VocoderCUDAGraphTarget,
        xvec_prefix_target: VocoderCUDAGraphTarget,
        suffix_target: VocoderCUDAGraphTarget,
        initial_codec_chunk_frames: int,
        codec_chunk_frames: int,
        codec_chunk_ramp: list[int],
    ) -> None:
        self.decoder = decoder
        self.stateless_target = stateless_target
        self.icl_prefix_target = icl_prefix_target
        self.xvec_prefix_target = xvec_prefix_target
        self.suffix_target = suffix_target
        self.initial_codec_chunk_frames = initial_codec_chunk_frames
        self.codec_chunk_frames = codec_chunk_frames
        self.codec_chunk_ramp = codec_chunk_ramp

    @staticmethod
    def _target_batch_ranges(
        target: VocoderCUDAGraphTarget,
        request_count: int,
        *,
        mode: str | None = None,
        target_frames: int | None = None,
        eager_max_batch_size: int = 0,
    ) -> list[tuple[int, int]]:
        if target.is_graph_bound:
            batches = sorted(
                {
                    int(descriptor.variant.batch_size)
                    for descriptor in target.descriptors
                    if (mode is None or descriptor.variant.mode == mode)
                    and (target_frames is None or descriptor.variant.target_frames == target_frames)
                }
            )
            batch_limit = max(batches, default=0)
        else:
            batch_limit = eager_max_batch_size
        if request_count <= 0 or batch_limit <= 0 or request_count <= batch_limit:
            return [(0, request_count)]
        return [(start, min(start + batch_limit, request_count)) for start in range(0, request_count, batch_limit)]

    def _run_target_batches(
        self,
        target: VocoderCUDAGraphTarget,
        request_count: int,
        execute: Callable[[int, int], list[torch.Tensor] | None],
        *,
        mode: str | None = None,
        target_frames: int | None = None,
        eager_max_batch_size: int = 0,
    ) -> list[torch.Tensor] | None:
        outputs: list[torch.Tensor] = []
        for start, end in self._target_batch_ranges(
            target,
            request_count,
            mode=mode,
            target_frames=target_frames,
            eager_max_batch_size=eager_max_batch_size,
        ):
            batch_outputs = execute(start, end)
            if batch_outputs is None:
                return None
            outputs.extend(batch_outputs)
        return outputs

    def _decode_icl_prefix_target_batch(
        self,
        codes_list: list[torch.Tensor],
        request_caches: list[dict[str, Any]],
        *,
        prefix_length: int,
        initial_chunk_frames: int,
        eager_max_batch_size: int,
    ) -> list[torch.Tensor] | None:
        def once(start: int, end: int) -> list[torch.Tensor] | None:
            group_codes, group_caches = codes_list[start:end], request_caches[start:end]
            prepared = _prepare_icl_prefix_batch(
                group_codes,
                group_caches,
                prefix_length=prefix_length,
                initial_chunk_frames=initial_chunk_frames,
                dtype=self.decoder.dtype,
            )
            if prepared is None:
                return None
            eager_state: dict[str, Any] = {}
            output = self.icl_prefix_target(
                prepared.codes,
                eager_state,
                prefix_length,
                prefix_hidden_mask=prepared.hidden_mask,
                prefix_attention_mask=prepared.prefix_mask,
                suffix_attention_mask=prepared.suffix_mask,
            )
            if isinstance(eager_state.get("past_key_values"), DynamicCache):
                eager_state["past_key_values"] = self.decoder._split_eager_prefix_cache(
                    eager_state["past_key_values"], len(group_caches)
                )
            return self.decoder._finalize_prefix_result(
                output,
                eager_state,
                group_caches,
                initial_chunk_frames=initial_chunk_frames,
                prefix_pads=prepared.prefix_pads,
            )

        return self._run_target_batches(
            self.icl_prefix_target,
            len(codes_list),
            once,
            eager_max_batch_size=eager_max_batch_size,
        )

    def _decode_xvec_prefix_target_batch(
        self,
        codes_list: list[torch.Tensor],
        request_caches: list[dict[str, Any]],
        *,
        initial_chunk_frames: int,
        eager_max_batch_size: int,
    ) -> list[torch.Tensor] | None:
        def once(start: int, end: int) -> list[torch.Tensor] | None:
            group_codes, group_caches = codes_list[start:end], request_caches[start:end]
            if any(int(codes.shape[-1]) != initial_chunk_frames for codes in group_codes):
                return None
            eager_state: dict[str, Any] = {}
            output = self.xvec_prefix_target(torch.cat(group_codes, dim=0), eager_state)
            if isinstance(eager_state.get("past_key_values"), DynamicCache):
                eager_state["past_key_values"] = self.decoder._split_eager_prefix_cache(
                    eager_state["past_key_values"], len(group_caches)
                )
            return self.decoder._finalize_prefix_result(
                output,
                eager_state,
                group_caches,
                initial_chunk_frames=initial_chunk_frames,
                prefix_pads=None,
            )

        return self._run_target_batches(
            self.xvec_prefix_target,
            len(codes_list),
            once,
            eager_max_batch_size=eager_max_batch_size,
        )

    def _decode_suffix_target_batch(
        self,
        mode: str,
        target_frames: int,
        codes_list: list[torch.Tensor],
        request_caches: list[dict[str, Any]],
        new_frames_list: list[int],
        *,
        transitions_by_mode: dict[str, dict[int, int]],
        eager_max_batch_size: int,
    ) -> list[torch.Tensor] | None:
        previous_frames = transitions_by_mode[mode].get(target_frames)
        if previous_frames is None:
            return None

        def once(start: int, end: int) -> list[torch.Tensor] | None:
            group_codes = codes_list[start:end]
            group_caches = request_caches[start:end]
            group_frames = new_frames_list[start:end]
            metadata = _resolve_suffix_batch(
                self.decoder,
                mode,
                target_frames,
                group_codes,
                group_caches,
                group_frames,
                transitions_by_mode=transitions_by_mode,
            )
            if metadata is None:
                return None
            result = self.suffix_target(mode, target_frames, group_codes, group_caches, group_frames)
            if result is None:
                return None
            return self.decoder._finalize_suffix_result(
                result,
                group_caches,
                group_frames,
                target_frames,
                metadata.expected_new_frames,
            )

        return self._run_target_batches(
            self.suffix_target,
            len(codes_list),
            once,
            mode=mode,
            target_frames=target_frames,
            eager_max_batch_size=eager_max_batch_size,
        )

    def batched_chunked_decode(
        self,
        codes: torch.Tensor,
        lengths: list[int],
        *,
        caches: list[dict[str, Any]] | None,
        chunk_size: int,
        left_context_size: int,
        max_batch_size: int,
    ) -> list[torch.Tensor]:
        if caches is not None:
            prefix_length = int(getattr(self.decoder.config, "sliding_window", 0) or 0)
            contract = resolve_stateful_chunk_contract(
                prefix_length=prefix_length,
                initial_codec_chunk_frames=self.initial_codec_chunk_frames,
                codec_chunk_frames=self.codec_chunk_frames,
                codec_chunk_ramp=self.codec_chunk_ramp,
            )
            initial_chunk_frames, transitions = contract.resolved_initial_frames, contract.transitions
            codes_list = [codes[row : row + 1, :, :length] for row, length in enumerate(lengths)]
            outputs = self.decoder.batched_request_decode(
                codes_list,
                caches,
                prefix_length=prefix_length,
                initial_chunk_frames=initial_chunk_frames,
                codec_chunk_frames=self.codec_chunk_frames,
                transitions_by_mode={"icl": transitions, "xvec": transitions},
                decode_icl_prefix_batch=lambda group_codes, group_caches: self._decode_icl_prefix_target_batch(
                    group_codes,
                    group_caches,
                    prefix_length=prefix_length,
                    initial_chunk_frames=initial_chunk_frames,
                    eager_max_batch_size=max_batch_size,
                ),
                decode_xvec_prefix_batch=lambda group_codes, group_caches: self._decode_xvec_prefix_target_batch(
                    group_codes,
                    group_caches,
                    initial_chunk_frames=initial_chunk_frames,
                    eager_max_batch_size=max_batch_size,
                ),
                decode_suffix_batch=lambda mode, target, group_codes, group_caches, new_frames: (
                    self._decode_suffix_target_batch(
                        mode,
                        target,
                        group_codes,
                        group_caches,
                        new_frames,
                        transitions_by_mode={"icl": transitions, "xvec": transitions},
                        eager_max_batch_size=max_batch_size,
                    )
                ),
                decode_fallback=lambda request_codes, cache: self.decoder.chunked_decode(
                    request_codes,
                    caches=cache,
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                ),
            )
            return [output[0] for output in outputs]
        return self._batched_stateless_chunked_decode(
            codes,
            lengths,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            max_batch_size=max_batch_size,
        )

    def _stateless_frame_bucket(self, frames: int) -> int:
        candidates = [
            int(descriptor.variant.frames)
            for descriptor in self.stateless_target.descriptors
            if int(descriptor.variant.frames) >= frames
        ]
        return min(candidates, default=frames)

    def _batched_stateless_chunked_decode(
        self, codes: torch.Tensor, lengths: list[int], *, chunk_size: int, left_context_size: int, max_batch_size: int
    ) -> list[torch.Tensor]:
        batch_size = int(codes.shape[0])
        if len(lengths) != batch_size:
            raise ValueError("codes and lengths must have the same batch size")
        if any(length < 0 or length > codes.shape[-1] for length in lengths):
            raise ValueError(f"Decode lengths must be within [0, {codes.shape[-1]}], got {lengths}")
        if not lengths or max(lengths) == 0:
            return [codes.new_empty((1, 0), dtype=torch.float32) for _ in range(batch_size)]
        if chunk_size <= 0 or left_context_size < 0 or max_batch_size < 0:
            raise ValueError("invalid stateless chunk decode settings")
        outputs: list[torch.Tensor | None] = [
            codes.new_empty((1, 0), dtype=torch.float32) if length == 0 else None for length in lengths
        ]
        for start in range(0, max(lengths), chunk_size):
            groups: dict[int, list[tuple[int, int, int, int, int]]] = {}
            for request_index, request_length in enumerate(lengths):
                if start < request_length:
                    end = min(start + chunk_size, request_length)
                    context = left_context_size if start > left_context_size else start
                    input_start = start - context
                    groups.setdefault(self._stateless_frame_bucket(end - input_start), []).append(
                        (request_index, input_start, end, start, context)
                    )
            for bucket, jobs in groups.items():
                graph_batches = tuple(
                    sorted(
                        {
                            int(descriptor.variant.batch_size)
                            for descriptor in self.stateless_target.descriptors
                            if int(descriptor.variant.frames) == bucket
                        }
                    )
                )
                offset = 0
                while offset < len(jobs):
                    remaining = len(jobs) - offset
                    graph_batch = (
                        max(graph_batches)
                        if graph_batches and remaining > max(graph_batches)
                        else next((size for size in graph_batches if size >= remaining), None)
                    )
                    request_count = min(remaining, graph_batch) if graph_batch is not None else remaining
                    if graph_batch is None and max_batch_size > 0:
                        request_count = min(request_count, max_batch_size)
                    decode_batch = graph_batch if graph_batch is not None else request_count
                    job_batch = jobs[offset : offset + request_count]
                    batched_codes = codes.new_zeros((decode_batch, codes.shape[1], bucket))
                    for row, (request_index, input_start, end, _, _) in enumerate(job_batch):
                        chunk = codes[request_index, :, input_start:end]
                        batched_codes[row, :, : chunk.shape[-1]].copy_(chunk)
                    wav = self.stateless_target(batched_codes)
                    for row, (request_index, _, end, chunk_start, context) in enumerate(job_batch):
                        if outputs[request_index] is None:
                            outputs[request_index] = wav.new_empty(
                                (*wav.shape[1:-1], lengths[request_index] * self.decoder.total_upsample)
                            )
                        src_start = context * self.decoder.total_upsample
                        dst_start = chunk_start * self.decoder.total_upsample
                        dst_end = end * self.decoder.total_upsample
                        outputs[request_index][..., dst_start:dst_end].copy_(
                            wav[row, ..., src_start : src_start + dst_end - dst_start]
                        )
                    offset += request_count
        if any(output is None for output in outputs):
            raise RuntimeError("stateless batched decode did not produce an output for every request")
        return [output for output in outputs if output is not None]
