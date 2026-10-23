# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Engine-owned routing for host-ready codec output; no GPU waits here."""

from __future__ import annotations

import queue

import torch
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs


class CosyVoice3AsyncScheduler(OmniARAsyncScheduler):
    def __init__(self, *args, **kwargs):
        self.codec = None
        self._audio_routes: dict[str, int] = {}
        self._audio_generations: dict[str, int] = {}
        self._abort_outputs = []
        self._audio_terminals: dict[str, OmniEngineCoreOutput] = {}
        self._audio_done: set[str] = set()
        self._audio_cancelled: set[str] = set()
        super().__init__(*args, **kwargs)

    def schedule(self, throttle_prefills=False):
        if self.codec is None:
            raise RuntimeError("CosyVoice3 async codec requires an in-process UniProc executor")
        self.codec.check_error()
        # Finished AR requests keep their codec capacity until final PCM is
        # consumed. Preserve capacity for active streams instead of admitting
        # an unbounded succession of detached tails.
        original = self.max_num_active_reqs
        running = {r.request_id for r in self.running}
        detached = len(self._audio_routes.keys() - running)
        self.max_num_active_reqs = max(0, original - detached)
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self.max_num_active_reqs = original
        for request in output.scheduled_new_reqs:
            req_id = request.req_id
            self._audio_generations[req_id] = self.codec.register(req_id)
            self._audio_routes[req_id] = self.requests[req_id].client_index
            self._audio_cancelled.discard(req_id)
        # Restarting a partially emitted waveform on KV preemption is unsafe.
        if output.preempted_req_ids:
            self.finish_requests(output.preempted_req_ids, RequestStatus.FINISHED_ERROR)
        return output

    def _retire_audio(self, req_id):
        self._audio_routes.pop(req_id, None)
        self._audio_generations.pop(req_id, None)
        self._audio_terminals.pop(req_id, None)
        self._audio_done.discard(req_id)
        self._audio_cancelled.discard(req_id)
        self.codec.release(req_id)

    def finish_requests(self, request_ids, finished_status):
        if request_ids is not None and not isinstance(request_ids, str):
            request_ids = list(request_ids)
        ids = (
            set(self._audio_routes)
            if request_ids is None
            else ({request_ids} if isinstance(request_ids, str) else set(request_ids))
        )
        for req_id in ids & self._audio_routes.keys():
            self.codec.cancel(req_id)
            self._audio_cancelled.add(req_id)
            if req_id in self._audio_terminals:
                output = self._audio_terminals[req_id]
                output.finish_reason = (
                    FinishReason.ABORT if finished_status == RequestStatus.FINISHED_ABORTED else FinishReason.ERROR
                )
                self._abort_outputs.append((self._audio_routes[req_id], output))
                self._retire_audio(req_id)
        return super().finish_requests(request_ids, finished_status)

    def get_num_unfinished_requests(self):
        return super().get_num_unfinished_requests() + len(self._audio_terminals)

    def add_request(self, request):
        if request.request_id in self._audio_routes:
            raise ValueError("Request ID is still draining codec audio")
        return super().add_request(request)

    def has_requests(self):
        return bool(self._audio_routes or self._abort_outputs) or super().has_requests()

    def has_unfinished_requests(self):
        return bool(self._audio_routes or self._abort_outputs) or super().has_unfinished_requests()

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        self.codec.check_error()
        # Retain normal stop metadata/tokens, while freeing AR/KV through the
        # unchanged base algorithm. Only the client terminal is deferred.
        for client_output in result.values():
            kept = []
            for output in client_output.outputs:
                req_id = output.request_id
                if req_id in self._audio_routes and output.finish_reason is not None:
                    if output.finish_reason in (FinishReason.ABORT, FinishReason.ERROR):
                        self.codec.cancel(req_id)
                        self._audio_cancelled.add(req_id)
                        self._retire_audio(req_id)
                        kept.append(output)
                    else:
                        self._audio_terminals[req_id] = output
                else:
                    kept.append(output)
            client_output.outputs = kept
            if client_output.finished_requests:
                client_output.finished_requests.difference_update(self._audio_routes)
        for client, output in self._abort_outputs:
            batch = result.setdefault(client, OmniEngineCoreOutputs(outputs=[]))
            batch.outputs.append(output)
            if batch.finished_requests is None:
                batch.finished_requests = set()
            batch.finished_requests.add(output.request_id)
        self._abort_outputs.clear()
        ready: dict[str, list[torch.Tensor]] = {}
        while True:
            try:
                audio = self.codec.outputs.get_nowait()
            except queue.Empty:
                break
            req_id = audio.request_id
            if self._audio_generations.get(req_id) != audio.generation or req_id in self._audio_cancelled:
                continue
            if audio.pcm is not None and audio.pcm.numel():
                ready.setdefault(req_id, []).append(audio.pcm)
            if audio.final:
                self._audio_done.add(req_id)
        for req_id in ready.keys() | (self._audio_done & self._audio_terminals.keys()):
            client = self._audio_routes[req_id]
            batch = result.setdefault(client, OmniEngineCoreOutputs(outputs=[]))
            final = req_id in self._audio_done and req_id in self._audio_terminals
            output = (
                self._audio_terminals[req_id] if final else OmniEngineCoreOutput(request_id=req_id, new_token_ids=[])
            )
            pieces = ready.get(req_id)
            if pieces:
                output.multimodal_output = {
                    "audio": pieces[0] if len(pieces) == 1 else torch.cat(pieces),
                    "sr": torch.tensor(self.audio_sample_rate, dtype=torch.int32),
                }
            existing = next((o for o in batch.outputs if o.request_id == req_id), None)
            if existing is not None:
                output.new_token_ids = existing.new_token_ids + output.new_token_ids
                batch.outputs.remove(existing)
            batch.outputs.append(output)
            if final:
                if batch.finished_requests is None:
                    batch.finished_requests = set()
                batch.finished_requests.add(req_id)
                self._retire_audio(req_id)
        return result
