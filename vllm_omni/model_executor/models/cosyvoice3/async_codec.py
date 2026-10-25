# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Append-only GPU token buffers and coalesced per-request codec progress.

The producer never waits for the codec to consume a step notification. A slow
codec catches up from immutable token/validity prefixes instead of a step FIFO.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_omni.platforms import current_omni_platform

from .cosyvoice3_code2wav import iter_decode_streaming_batch
from .pipeline_adapter import CodecRequestKey, SingleStageAdapter

if TYPE_CHECKING:
    from .cosyvoice3_code2wav import CosyVoice3Code2Wav


@dataclass
class SampleEntry:
    """CPU sampling metadata; token values remain in the supplied GPU tensor."""

    request_id: str
    position: int
    length_finished: bool
    conditioning: dict
    generation: int


@dataclass
class TokenBuffer:
    """Append-only storage: AR writes suffixes, codec confirms copied validity bits."""

    tokens: torch.Tensor
    speech: torch.Tensor
    host_speech: torch.Tensor
    conditioning: dict
    last_position: int = -1  # Producer-owned.
    limit_position: int | None = None
    confirmed_end: int = 0  # Consumer-owned, exclusive end of confirmed validity bits.


@dataclass
class TokenProgress:
    """Latest AR prefix and the GPU event that makes that prefix readable."""

    buffer: TokenBuffer
    end: int  # Exclusive end of the published token prefix.
    limit_position: int | None
    producer_done: torch.cuda.Event


@dataclass
class ControlRead:
    """Retain the source and pinned destination until the validity D2H completes."""

    progress: TokenProgress
    start: int
    copy_done: torch.cuda.Event


@dataclass
class DecodeProgress:
    """Confirmed token prefix and optional EOS/length-limit token position."""

    buffer: TokenBuffer
    length: int
    stop_position: int | None


@dataclass
class ReadyAudio:
    request_id: str
    pcm: torch.Tensor | None
    final: bool
    generation: int


@dataclass
class PendingAudio:
    """Retain GPU PCM until its pinned-host copy completes."""

    output: ReadyAudio
    copy_done: torch.cuda.Event
    source: torch.Tensor | None


class AsyncCodec:
    """AR submits prefixes; one codec thread consumes them and publishes CPU PCM.

    register/submit/cancel/release run on the engine thread. Token updates and
    retirements cross a short mailbox lock; reads, decoding and audio copies
    stay on the codec thread. Only completed D2H results enter outputs.
    """

    def __init__(
        self,
        decoder: CosyVoice3Code2Wav,
        adapter: SingleStageAdapter,
        device,
        max_model_len: int,
        pending_audio_threshold: int,
    ):
        self.decoder = decoder
        self.adapter = adapter
        self.device = device
        self.max_model_len = max_model_len
        self.outputs: queue.SimpleQueue[ReadyAudio] = queue.SimpleQueue()
        # Engine-owned request generations and append-only input buffers.
        self.generations: dict[str, int] = {}
        self.next_generation = 0
        # Checked at batch admission; not a strict per-row bound.
        self.pending_audio_threshold = pending_audio_threshold
        self.buffers: dict[CodecRequestKey, TokenBuffer] = {}
        # Cross-thread mailboxes coalesce updates instead of retaining steps.
        self.mailbox: dict[CodecRequestKey, TokenProgress] = {}
        self.retirements: dict[CodecRequestKey, tuple[bool, bool]] = {}
        self.lock = threading.Lock()
        # Codec-owned state; tombstones survive pending control/audio copies.
        self.cancelled: set[CodecRequestKey] = set()
        self.ended: set[CodecRequestKey] = set()
        self.released: set[CodecRequestKey] = set()
        self.reads: dict[CodecRequestKey, ControlRead] = {}
        self.progress: dict[CodecRequestKey, DecodeProgress] = {}
        self.pending: deque[PendingAudio] = deque()
        self.error: BaseException | None = None
        self.started = threading.Event()
        self.tick = threading.Event()
        self.closed = False
        self.thread = threading.Thread(target=self.run, name="cosyvoice3-codec", daemon=True)
        self.thread.start()
        if not self.started.wait(10):
            raise TimeoutError("CosyVoice3 codec thread did not initialize")
        self.check_error()

    def check_error(self) -> None:
        if self.error is not None:
            raise RuntimeError("CosyVoice3 asynchronous codec failed") from self.error

    def submit(
        self,
        entries: Sequence[SampleEntry],
        tokens: torch.Tensor,
        speech: torch.Tensor,
        rows: Sequence[int] | None = None,
    ) -> None:
        self.check_error()
        if self.closed:
            raise RuntimeError("CosyVoice3 codec is closed")
        updates = []
        source_rows = range(len(entries)) if rows is None else rows
        for row, entry in zip(source_rows, entries, strict=True):
            key = (entry.request_id, entry.generation)
            if self.generations.get(entry.request_id) != entry.generation:
                continue
            buffer = self.buffers.get(key)
            if buffer is None:
                if entry.position != 0:
                    raise ValueError("CosyVoice3 token buffer must start at position zero")
                buffer = TokenBuffer(
                    torch.empty((1, self.max_model_len), dtype=torch.long, device=tokens.device),
                    torch.empty(self.max_model_len, dtype=torch.bool, device=tokens.device),
                    torch.empty(self.max_model_len, dtype=torch.bool, pin_memory=True),
                    dict(entry.conditioning),
                )
                self.buffers[key] = buffer
            position = entry.position
            if buffer.limit_position is not None and position > buffer.limit_position:
                # The runner may already have scheduled a step beyond its CPU
                # length limit. It cannot extend the accepted speech prefix.
                continue
            if not buffer.last_position < position < self.max_model_len or position != buffer.last_position + 1:
                raise ValueError("CosyVoice3 token positions must be contiguous and within max_model_len")
            # Only new suffix positions are written. Codec reads frozen prefixes
            # directly; neither graph outputs nor an unbounded step queue are retained.
            buffer.tokens[:, position : position + 1].copy_(tokens[row : row + 1].reshape(1, 1))
            buffer.speech[position : position + 1].copy_(speech[row : row + 1])
            buffer.last_position = position
            if entry.length_finished and buffer.limit_position is None:
                buffer.limit_position = position
            updates.append((key, buffer, position + 1, buffer.limit_position))
        if not updates:
            return
        ready = torch.cuda.Event()
        ready.record()
        with self.lock:
            for key, buffer, end, limit in updates:
                self.mailbox[key] = TokenProgress(buffer, end, limit, ready)
        self.tick.set()

    def register(self, request_id: str) -> int:
        self.next_generation += 1
        self.generations[request_id] = self.next_generation
        return self.next_generation

    def retire(self, request_id: str, *, cancel: bool, release: bool) -> None:
        self.check_error()
        key = (request_id, self.generations[request_id])
        with self.lock:
            cancelled, released = self.retirements.get(key, (False, False))
            self.retirements[key] = (cancelled or cancel, released or release)
        if release:
            self.generations.pop(request_id)
            self.buffers.pop(key, None)
        self.tick.set()

    def cancel(self, request_id: str) -> None:
        self.retire(request_id, cancel=True, release=False)

    def release(self, request_id: str) -> None:
        self.retire(request_id, cancel=False, release=True)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.tick.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError("CosyVoice3 codec did not drain during shutdown")
        self.check_error()

    def consume_retirements(self) -> None:
        with self.lock:
            controls, self.retirements = self.retirements, {}
            for key in controls:
                self.mailbox.pop(key, None)
        for key, (cancelled, released) in controls.items():
            if cancelled:
                self.cancelled.add(key)
            if released:
                self.released.add(key)
            self.progress.pop(key, None)
            self.adapter.finish_requests({key})
        # ControlRead keeps source flags and pinned destination alive until its
        # event completes, even if the request was cancelled in the meantime.

    def publish(self) -> None:
        while self.pending and self.pending[0].copy_done.query():
            audio = self.pending.popleft().output
            if (audio.request_id, audio.generation) not in self.cancelled:
                self.outputs.put(audio)
        pending = {(x.output.request_id, x.output.generation) for x in self.pending}
        for key in self.released - pending - self.reads.keys():
            self.cancelled.discard(key)
            self.ended.discard(key)
            self.released.discard(key)

    def start_reads(self) -> None:
        """Copy only unread validity suffixes, ordered after AR prefix writes."""
        with self.lock:
            snapshots = [(key, value) for key, value in self.mailbox.items() if key not in self.reads]
            for key, _ in snapshots:
                self.mailbox.pop(key)
        with torch.cuda.stream(self.control_stream):
            for key, progress in snapshots:
                if key in self.ended or key in self.cancelled:
                    continue
                buffer = progress.buffer
                start = buffer.confirmed_end
                self.control_stream.wait_event(progress.producer_done)
                buffer.speech.record_stream(self.control_stream)
                buffer.host_speech[start : progress.end].copy_(buffer.speech[start : progress.end], non_blocking=True)
                copied = torch.cuda.Event()
                copied.record(self.control_stream)
                self.reads[key] = ControlRead(progress, start, copied)

    def poll_reads(self) -> None:
        """Confirm EOS on the CPU only after its validity copy is complete."""
        for key, read in list(self.reads.items()):
            if not read.copy_done.query():
                continue
            del self.reads[key]
            if key in self.cancelled or key in self.released or key in self.ended:
                continue
            progress = read.progress
            buffer = progress.buffer
            values = buffer.host_speech[read.start : progress.end].tolist()
            eos = next((read.start + i for i, value in enumerate(values) if not value), None)
            limit = progress.limit_position
            terminal = limit
            if eos is not None:
                terminal = eos if limit is None else min(eos, limit)
            length = progress.end
            if terminal is not None:
                length = terminal + int(values[terminal - read.start])
                self.ended.add(key)
            buffer.confirmed_end = progress.end
            self.progress[key] = DecodeProgress(buffer, length, terminal)

    def decode_ready(self) -> bool:
        """Submit ready chunks and their D2H without waiting for GPU completion."""
        items, bindings, empty_finals = [], [], []
        for key, progress in self.progress.items():
            if key in self.cancelled or key in self.released:
                continue
            buffer = progress.buffer
            buffer.tokens.record_stream(self.stream)
            self.adapter.initialize(key, buffer.conditioning, buffer.tokens)
            state = self.adapter.states[key]
            if state.finished:
                continue
            for tensor in (state.prompt_token, state.prompt_feat, state.embedding):
                tensor.record_stream(self.stream)
            item = self.adapter.prepare(key, progress.length, progress.stop_position)
            if item is not None:
                items.append(item)
                bindings.append((key, item["finalize"]))
            elif state.finished and progress.length == 0:
                empty_finals.append(key)
        if items:
            for index, wav, cache in iter_decode_streaming_batch(self.decoder, items):
                key, final = bindings[index]
                self.adapter.apply_result(key, cache)
                # Snapshot before the next request can reuse graph output storage.
                wav = wav.reshape(-1).float().clone()
                decoded = torch.cuda.Event()
                decoded.record(self.stream)
                with torch.cuda.stream(self.copy_stream):
                    self.copy_stream.wait_event(decoded)
                    host = torch.empty(wav.shape, dtype=wav.dtype, pin_memory=True)
                    host.copy_(wav, non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record(self.copy_stream)
                    self.pending.append(PendingAudio(ReadyAudio(key[0], host, final, key[1]), copied, wav))
                # Publish any completed copies between submissions, without
                # waiting for either HiFT or D2H before queuing the next row.
                self.publish()
        for key in empty_finals:
            copied = torch.cuda.Event()
            copied.record(self.copy_stream)
            self.pending.append(PendingAudio(ReadyAudio(key[0], None, True, key[1]), copied, None))
        return bool(items or empty_finals)

    @torch.inference_mode()
    def run(self) -> None:
        try:
            current_omni_platform.set_device(self.device)
            self.stream = torch.cuda.Stream(device=self.device)
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self.control_stream = torch.cuda.Stream(device=self.device)
            self.started.set()
            with torch.cuda.stream(self.stream):
                while True:
                    self.tick.clear()
                    self.consume_retirements()
                    self.publish()
                    self.start_reads()
                    self.poll_reads()
                    decoded = self.decode_ready() if len(self.pending) < self.pending_audio_threshold else False
                    with self.lock:
                        incoming = bool(self.mailbox or self.retirements)
                    if self.closed and not incoming and not self.reads and not self.pending and not decoded:
                        break
                    if not decoded:
                        self.tick.wait(0.0005)
                self.stream.synchronize()
                self.copy_stream.synchronize()
                self.control_stream.synchronize()
                self.adapter.finish_requests(set(self.adapter.states))
                self.buffers.clear()
                self.progress.clear()
        except BaseException as error:
            self.error = error
        finally:
            self.started.set()
