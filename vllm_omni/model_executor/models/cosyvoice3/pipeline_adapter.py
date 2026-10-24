# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Translate pipeline inputs into CosyVoice3's tensor-level decoding contract.

Request routing and token windows belong here; Flow/HiFT math and cache
contents belong to Code2Wav. Neither pipeline changes the decoder algorithm.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import to_struct
from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_batch_flow_debug
from vllm_omni.model_executor.models.cosyvoice3.utils import unpad_prompt_conditioning
from vllm_omni.model_executor.models.output_templates import OmniOutput

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import (
        CosyVoice3Code2Wav,
        StreamingFlowItem,
        StreamingHiFTState,
    )

logger = init_logger(__name__)


def _normalize_request_conditioning(payload: dict) -> dict:
    """Unwrap singleton conditioning containers at the per-request boundary.

    A processor/prefix-cache passthrough can retain a one-item tensor list.
    The code2wav schema consumes one request, so that wrapper carries no
    batch dimension. A multi-item list here is an unsplit batch and must
    fail instead of silently choosing another request's voice.
    """
    embed = payload.get("embed")
    if not isinstance(embed, Mapping):
        return payload
    normalized = dict(embed)
    for key in ("speech_token", "speech_feat", "embedding", "speech_token_len"):
        value = normalized.get(key)
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"CosyVoice3 per-request {key} contains an unsplit batch of {len(value)} items")
            normalized[key] = value[0]
    return {**payload, "embed": normalized}


class CrossStageAdapter:
    """Interpret connector payloads during the downstream stage's forward."""

    def __init__(self, sample_rate: int, vocab_size: int):
        self.sample_rate = sample_rate
        self.vocab_size = vocab_size
        self._stream_audio_cache_lock = Lock()
        self._stream_vocoder_cache_by_req: dict[str, StreamingHiFTState] = {}

    def finish_requests(self, request_ids: set[str]) -> None:
        with self._stream_audio_cache_lock:
            for request_id in request_ids:
                self._stream_vocoder_cache_by_req.pop(request_id, None)

    @staticmethod
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments."""
        if seq_token_counts is not None:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + int(count))
            total = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], total)] for i in range(len(seq_token_counts))]

        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + int(s))
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]

        return [ids]

    def _sanitize_codec_tokens(self, req_ids: torch.Tensor) -> torch.Tensor:
        """Filter non-code tokens before feeding flow token embedding."""
        vocab_size = self.vocab_size
        valid_mask = (req_ids >= 0) & (req_ids < vocab_size)
        return req_ids[valid_mask]

    def forward(
        self,
        decoder: CosyVoice3Code2Wav,
        input_ids: torch.Tensor,
        runtime_info: list,
        seq_token_counts: list[int] | None = None,
    ) -> OmniOutput:
        from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import decode_streaming_batch

        flat_ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(flat_ids, seq_token_counts)

        num_reqs = max(1, len(request_ids_list))
        debug_batch_flow = cosyvoice3_batch_flow_debug()
        if debug_batch_flow:
            logger.info(
                "CosyVoice3 code2wav debug: forward num_reqs=%d seq_token_counts=%s flat_ids=%d",
                num_reqs,
                seq_token_counts,
                int(flat_ids.numel()),
            )
        sample_rate = torch.tensor(int(self.sample_rate), dtype=torch.int32)
        empty_audio = torch.zeros((0,), dtype=torch.float32, device=input_ids.device)
        audios: list[torch.Tensor] = [empty_audio] * num_reqs
        srs: list[torch.Tensor] = [sample_rate] * num_reqs
        if not isinstance(runtime_info, list):
            runtime_info = []
        streaming_flow_items: list[StreamingFlowItem] = []
        bindings: list[tuple[int, str | None, bool]] = []

        for idx, req_ids in enumerate(request_ids_list):
            raw = runtime_info[idx] if idx < len(runtime_info) and isinstance(runtime_info[idx], dict) else {}
            payload = to_struct(_normalize_request_conditioning(raw))
            meta = payload.meta
            embed = payload.embed

            req_id = meta.req_id[0] if (meta and meta.req_id) else None
            stream_finished = (
                bool(meta.stream_finished.item()) if (meta and meta.stream_finished is not None) else False
            )
            speech_token = embed.speech_token if embed else None
            speech_feat = embed.speech_feat if embed else None
            embedding = embed.embedding if embed else None
            # Drop any right-padding carried from batched talker emission.
            if speech_token is not None and speech_feat is not None:
                speech_token, speech_feat = unpad_prompt_conditioning(
                    speech_token, speech_feat, embed.speech_token_len if embed else None
                )
            if speech_token is None or speech_feat is None or embedding is None:
                if stream_finished and req_id is not None:
                    self.finish_requests({req_id})
                audios[idx] = empty_audio
                if req_ids.numel() > 0 and (
                    (meta and meta.left_context_size is not None) or payload.generated_len is not None
                ):
                    info_keys = ",".join(
                        sorted(f for f in payload.__struct_fields__ if getattr(payload, f) is not None)
                    )
                    logger.warning_once(
                        "CosyVoice3 code2wav missing prompt conditioning for non-empty codec tokens: "
                        "raw_len=%d info_keys=%s",
                        int(req_ids.numel()),
                        info_keys,
                    )
                continue

            token = self._sanitize_codec_tokens(req_ids)
            if token.numel() == 0:
                if stream_finished and req_id is not None:
                    self.finish_requests({req_id})
                audios[idx] = empty_audio
                if req_ids.numel() > 0:
                    logger.warning_once(
                        "CosyVoice3 code2wav received no valid codec tokens after filtering: "
                        "raw_len=%d raw_range=[%d,%d] vocab_size=%d",
                        req_ids.numel(),
                        int(req_ids.min().item()),
                        int(req_ids.max().item()),
                        self.vocab_size,
                    )
                continue

            # `generated_len` is injected for many models by the generic
            # runner, so only explicit chunk-routing fields should switch
            # code2wav into the streaming path.
            uses_streaming_decode = meta and (meta.stream_finished is not None or meta.left_context_size is not None)
            if uses_streaming_decode:
                token_offset = max(0, meta.left_context_size or 0)

                cache_state = None
                if req_id is not None:
                    with self._stream_audio_cache_lock:
                        cache_state = self._stream_vocoder_cache_by_req.get(req_id)

                bindings.append((idx, req_id, stream_finished))
                streaming_flow_items.append(
                    {
                        "token": token.unsqueeze(0),
                        "prompt_token": speech_token[:1],
                        "prompt_feat": speech_feat[:1],
                        "embedding": embedding[:1],
                        "cache_state": cache_state,
                        "token_offset_tokens": token_offset,
                        "finalize": stream_finished,
                    }
                )
                continue
            else:
                token_offset = max(0, meta.talker_prefill_offset or 0) if meta else 0
                tts_speech = decoder.forward(
                    token=token.unsqueeze(0),
                    prompt_token=speech_token[:1],
                    prompt_feat=speech_feat[:1],
                    embedding=embedding[:1],
                    n_timesteps=10,
                    token_offset_tokens=token_offset,
                )

            audio = tts_speech.reshape(-1).to(dtype=torch.float32)

            if stream_finished and req_id is not None:
                self.finish_requests({req_id})
            audios[idx] = audio

        if streaming_flow_items:
            if debug_batch_flow:
                item_shapes = [
                    (
                        tuple(item["token"].shape),
                        tuple(item["prompt_token"].shape),
                        tuple(item["prompt_feat"].shape),
                        bool(item.get("finalize", False)),
                    )
                    for item in streaming_flow_items
                ]
                logger.info(
                    "CosyVoice3 code2wav debug: streaming_items=%d item_shapes=%s",
                    len(streaming_flow_items),
                    item_shapes,
                )
            streaming_results = decode_streaming_batch(decoder, streaming_flow_items)

            for (idx, req_id, stream_finished), (tts_speech, new_cache_state) in zip(bindings, streaming_results):
                if req_id is not None:
                    if new_cache_state is None or stream_finished:
                        self.finish_requests({req_id})
                    else:
                        with self._stream_audio_cache_lock:
                            self._stream_vocoder_cache_by_req[req_id] = new_cache_state

                audio = tts_speech.reshape(-1).to(dtype=torch.float32)
                audios[idx] = audio

        return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audios, "sr": srs})


@dataclass
class StreamingDecodeState:
    """Pipeline-owned accumulation; the decoder cache is opaque to this layer."""

    token: torch.Tensor
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor
    emitted: int = 0
    hop: int = 25
    cache: StreamingHiFTState | None = None
    finished: bool = False


CodecRequestKey = tuple[str, int]


class SingleStageAdapter:
    """Plan codec chunks over a request's immutable, validated token prefix.

    The AR producer owns token writes; the codec thread owns this planner and
    its vocoder caches. Generation keys isolate request-ID reuse. Chunk ramp,
    prompt padding and lookahead match the two-stage connector.
    """

    def __init__(self, vocab_size: int, chunk_size: int = 25, lookahead: int = 3):
        if chunk_size <= 0 or lookahead < 0:
            raise ValueError("CosyVoice3 chunk size must be positive and lookahead nonnegative")
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.lookahead = lookahead
        self.states: dict[CodecRequestKey, StreamingDecodeState] = {}

    def initialize(self, req_id: CodecRequestKey, conditioning: Mapping, token_buffer: torch.Tensor) -> None:
        """Bind an append-only producer buffer; chunk/cache state stays consumer-owned."""
        if req_id in self.states:
            return
        payload = to_struct(_normalize_request_conditioning({"embed": dict(conditioning)}))
        embed = payload.embed
        if embed is None or any(getattr(embed, key) is None for key in ("speech_token", "speech_feat", "embedding")):
            raise ValueError(f"CosyVoice3 single-stage request {req_id} has no prompt conditioning")
        prompt_token, prompt_feat = unpad_prompt_conditioning(
            embed.speech_token, embed.speech_feat, embed.speech_token_len
        )
        state = StreamingDecodeState(
            token=token_buffer,
            prompt_token=prompt_token[:1],
            prompt_feat=prompt_feat[:1],
            embedding=embed.embedding[:1],
            hop=self.chunk_size,
        )
        self.states[req_id] = state

    def prepare(self, req_id: CodecRequestKey, length: int, final: bool) -> StreamingFlowItem | None:
        """Plan the existing chunk/tail path using runner-derived lengths."""
        state = self.states[req_id]
        if state.finished:
            return None
        pad = (-state.prompt_token.shape[1]) % self.chunk_size if state.emitted == 0 else 0
        hop = state.hop + pad
        if not final and length - state.emitted < hop + self.lookahead:
            return None
        if length == 0:
            state.finished = final
            return None
        prefix = length if final else state.emitted + hop + self.lookahead
        item: StreamingFlowItem = {
            "token": state.token[:, :prefix],
            "prompt_token": state.prompt_token,
            "prompt_feat": state.prompt_feat,
            "embedding": state.embedding,
            "cache_state": state.cache,
            "token_offset_tokens": state.emitted,
            "finalize": final,
        }
        if not final:
            state.emitted += hop
            state.hop = min(4 * self.chunk_size, 2 * state.hop)
        state.finished = final
        return item

    def apply_result(self, req_id: CodecRequestKey, cache: StreamingHiFTState | None) -> None:
        state = self.states[req_id]
        state.cache = None if state.finished else cache

    def finish_requests(self, request_ids: Iterable[CodecRequestKey]) -> None:
        for req_id in request_ids:
            self.states.pop(req_id, None)
