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

import torch

from vllm_omni.platforms import current_omni_platform

from .cosyvoice3_code2wav import iter_decode_streaming_batch
from .pipeline_adapter import InnerStageAdapter

RequestKey = tuple[str, int]


@dataclass
class SampleEntry:
    request_id: str
    position: int
    length_finished: bool
    conditioning: dict
    generation: int


@dataclass
class TokenBuffer:
    tokens: torch.Tensor
    speech: torch.Tensor
    host_speech: torch.Tensor
    conditioning: dict
    last_position: int = -1  # Producer-owned.
    limit_position: int | None = None
    confirmed: int = 0  # Consumer-owned.


@dataclass
class TokenProgress:
    buffer: TokenBuffer
    end: int
    limit_position: int | None
    ready: torch.cuda.Event


@dataclass
class ControlRead:
    progress: TokenProgress
    start: int
    ready: torch.cuda.Event


@dataclass
class DecodeProgress:
    buffer: TokenBuffer
    length: int
    nonfinal_length: int
    final: bool


@dataclass
class ReadyAudio:
    request_id: str
    pcm: torch.Tensor | None
    final: bool
    generation: int


@dataclass
class PendingAudio:
    output: ReadyAudio
    ready: torch.cuda.Event
    source: torch.Tensor | None


class AsyncCodec:
    def __init__(self, model, adapter: InnerStageAdapter, device, max_model_len: int, capacity: int):
        self.model = model
        self.adapter = adapter
        self.device = device
        self.max_model_len = max_model_len
        self.outputs: queue.SimpleQueue[ReadyAudio] = queue.SimpleQueue()
        self.generations: dict[str, int] = {}
        self._next_generation = 0
        self._capacity = capacity  # Original batch-admission threshold; never producer notifications.
        self._buffers: dict[RequestKey, TokenBuffer] = {}
        self._mailbox: dict[RequestKey, TokenProgress] = {}
        self._retirements: dict[RequestKey, tuple[bool, bool]] = {}
        self._lock = threading.Lock()
        self._cancelled: set[RequestKey] = set()
        self._ended: set[RequestKey] = set()
        self._released: set[RequestKey] = set()
        self._reads: dict[RequestKey, ControlRead] = {}
        self._progress: dict[RequestKey, DecodeProgress] = {}
        self._pending: deque[PendingAudio] = deque()
        self._error: BaseException | None = None
        self._started = threading.Event()
        self._tick = threading.Event()
        self._closed = False
        self.thread = threading.Thread(target=self._run, name="cosyvoice3-codec", daemon=True)
        self.thread.start()
        if not self._started.wait(10):
            raise TimeoutError("CosyVoice3 codec thread did not initialize")
        self.check_error()

    def check_error(self):
        if self._error is not None:
            raise RuntimeError("CosyVoice3 asynchronous codec failed") from self._error

    def submit(self, entries, tokens, speech, rows: Sequence[int] | None = None):
        self.check_error()
        if self._closed:
            raise RuntimeError("CosyVoice3 codec is closed")
        updates = []
        source_rows = range(len(entries)) if rows is None else rows
        for row, entry in zip(source_rows, entries, strict=True):
            key = (entry.request_id, entry.generation)
            if self.generations.get(entry.request_id) != entry.generation:
                continue
            buffer = self._buffers.get(key)
            if buffer is None:
                if entry.position != 0:
                    raise ValueError("CosyVoice3 token buffer must start at position zero")
                buffer = TokenBuffer(
                    torch.empty((1, self.max_model_len), dtype=torch.long, device=tokens.device),
                    torch.empty(self.max_model_len, dtype=torch.bool, device=tokens.device),
                    torch.empty(self.max_model_len, dtype=torch.bool, pin_memory=True),
                    dict(entry.conditioning),
                )
                self._buffers[key] = buffer
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
        with self._lock:
            for key, buffer, end, limit in updates:
                self._mailbox[key] = TokenProgress(buffer, end, limit, ready)
        self._tick.set()

    def register(self, request_id: str):
        self._next_generation += 1
        self.generations[request_id] = self._next_generation
        return self._next_generation

    def _retire(self, request_id: str, *, cancel: bool, release: bool):
        self.check_error()
        key = (request_id, self.generations[request_id])
        with self._lock:
            cancelled, released = self._retirements.get(key, (False, False))
            self._retirements[key] = (cancelled or cancel, released or release)
        if release:
            self.generations.pop(request_id)
            self._buffers.pop(key, None)
        self._tick.set()

    def cancel(self, request_id: str):
        self._retire(request_id, cancel=True, release=False)

    def release(self, request_id: str):
        self._retire(request_id, cancel=False, release=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._tick.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError("CosyVoice3 codec did not drain during shutdown")
        self.check_error()

    def _consume_retirements(self):
        with self._lock:
            controls, self._retirements = self._retirements, {}
            for key in controls:
                self._mailbox.pop(key, None)
        for key, (cancelled, released) in controls.items():
            if cancelled:
                self._cancelled.add(key)
            if released:
                self._released.add(key)
            self._progress.pop(key, None)
            self.adapter.finish_requests({key})
        # ControlRead keeps source flags and pinned destination alive until its
        # event completes, even if the request was cancelled in the meantime.

    def _publish(self):
        while self._pending and self._pending[0].ready.query():
            audio = self._pending.popleft().output
            if (audio.request_id, audio.generation) not in self._cancelled:
                self.outputs.put(audio)
        pending = {(x.output.request_id, x.output.generation) for x in self._pending}
        for key in self._released - pending - self._reads.keys():
            self._cancelled.discard(key)
            self._ended.discard(key)
            self._released.discard(key)

    def _start_reads(self):
        with self._lock:
            snapshots = [(key, value) for key, value in self._mailbox.items() if key not in self._reads]
            for key, _ in snapshots:
                self._mailbox.pop(key)
        with torch.cuda.stream(self.control_stream):
            for key, progress in snapshots:
                if key in self._ended or key in self._cancelled:
                    continue
                buffer = progress.buffer
                start = buffer.confirmed
                self.control_stream.wait_event(progress.ready)
                buffer.speech.record_stream(self.control_stream)
                buffer.host_speech[start : progress.end].copy_(buffer.speech[start : progress.end], non_blocking=True)
                copied = torch.cuda.Event()
                copied.record(self.control_stream)
                self._reads[key] = ControlRead(progress, start, copied)

    def _poll_reads(self):
        for key, read in list(self._reads.items()):
            if not read.ready.query():
                continue
            del self._reads[key]
            if key in self._cancelled or key in self._released or key in self._ended:
                continue
            progress = read.progress
            buffer = progress.buffer
            values = buffer.host_speech[read.start : progress.end].tolist()
            eos = next((read.start + i for i, value in enumerate(values) if not value), None)
            limit = progress.limit_position
            terminal = min(p for p in (eos, limit) if p is not None) if eos is not None or limit is not None else None
            length = progress.end
            nonfinal_length = length
            if terminal is not None:
                length = terminal + int(values[terminal - read.start])
                nonfinal_length = terminal
                self._ended.add(key)
            buffer.confirmed = progress.end
            self._progress[key] = DecodeProgress(buffer, length, nonfinal_length, terminal is not None)

    def _decode_ready(self) -> bool:
        items, bindings, empty_finals = [], [], []
        for key, progress in self._progress.items():
            if key in self._cancelled or key in self._released:
                continue
            buffer = progress.buffer
            buffer.tokens.record_stream(self.stream)
            self.adapter.initialize(key, buffer.conditioning, buffer.tokens)
            state = self.adapter.states[key]
            if state.finished:
                continue
            for tensor in (state.prompt_token, state.prompt_feat, state.embedding):
                tensor.record_stream(self.stream)
            pad = (-state.prompt_token.shape[1]) % self.adapter.chunk_size if state.emitted == 0 else 0
            threshold = state.emitted + state.hop + pad + self.adapter.lookahead
            # Catch up through the original chunk ramp before emitting a tail.
            # Coalescing notifications must not merge all pending chunks into one.
            final = progress.final and progress.nonfinal_length < threshold
            item = self.adapter.prepare(key, progress.length, final)
            if item is not None:
                items.append(item)
                bindings.append((key, final))
            elif final and progress.length == 0:
                empty_finals.append(key)
        if items:
            for index, wav, cache in iter_decode_streaming_batch(self.model.code2wav, items):
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
                    self._pending.append(PendingAudio(ReadyAudio(key[0], host, final, key[1]), copied, wav))
                # Publish any completed copies between submissions, without
                # waiting for either HiFT or D2H before queuing the next row.
                self._publish()
        for key in empty_finals:
            copied = torch.cuda.Event()
            copied.record(self.copy_stream)
            self._pending.append(PendingAudio(ReadyAudio(key[0], None, True, key[1]), copied, None))
        return bool(items or empty_finals)

    @torch.inference_mode()
    def _run(self):
        try:
            current_omni_platform.set_device(self.device)
            self.stream = torch.cuda.Stream(device=self.device)
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self.control_stream = torch.cuda.Stream(device=self.device)
            self._started.set()
            with torch.cuda.stream(self.stream):
                while True:
                    self._tick.clear()
                    self._consume_retirements()
                    self._publish()
                    self._start_reads()
                    self._poll_reads()
                    decoded = self._decode_ready() if len(self._pending) < self._capacity else False
                    with self._lock:
                        incoming = bool(self._mailbox or self._retirements)
                    if self._closed and not incoming and not self._reads and not self._pending and not decoded:
                        break
                    if not decoded:
                        self._tick.wait(0.0005)
                self.stream.synchronize()
                self.copy_stream.synchronize()
                self.control_stream.synchronize()
                self.adapter.finish_requests(set(self.adapter.states))
                self._buffers.clear()
                self._progress.clear()
        except BaseException as error:
            self._error = error
        finally:
            self._started.set()
