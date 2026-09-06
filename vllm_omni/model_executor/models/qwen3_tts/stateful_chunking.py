# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-TTS stateful streaming chunk-shape policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3TTSStatefulChunkContract:
    """Resolved first chunk and graph transition coverage for one decoder."""

    resolved_initial_frames: int
    transitions: dict[int, int]


def resolve_stateful_chunk_contract(
    *,
    prefix_length: int,
    initial_codec_chunk_frames: int,
    codec_chunk_frames: int,
    codec_chunk_ramp: Sequence[int] | None,
) -> Qwen3TTSStatefulChunkContract:
    """Derive the legacy Qwen3-TTS stateful chunk progression once."""

    ramp = tuple(int(size) for size in codec_chunk_ramp or ())
    resolved_initial_frames = ramp[0] if ramp else int(initial_codec_chunk_frames)
    transitions: dict[int, int] = {}
    if prefix_length <= 2 or resolved_initial_frames <= 0 or codec_chunk_frames <= 0:
        return Qwen3TTSStatefulChunkContract(resolved_initial_frames, transitions)
    previous = resolved_initial_frames
    for new_frames in ramp[1:]:
        if previous >= prefix_length:
            break
        target = previous + new_frames
        transitions[target] = previous
        previous = target
    while previous < prefix_length:
        target = previous + int(codec_chunk_frames)
        transitions[target] = previous
        previous = target
    transitions[int(prefix_length) + int(codec_chunk_frames)] = int(prefix_length)
    return Qwen3TTSStatefulChunkContract(resolved_initial_frames, transitions)
