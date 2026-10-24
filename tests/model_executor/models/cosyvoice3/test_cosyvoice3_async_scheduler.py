# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scheduler routing tests for deferred completion, cancellation and stale audio.

Dependency shims isolate the actual model-local code from CUDA-only vLLM imports.
Run with --noconftest on hosts without a vLLM installation.
"""

import importlib.util
import queue
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/cosyvoice3"


def _module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def modules(monkeypatch):
    class Reason(Enum):
        STOP = 0
        LENGTH = 1
        ABORT = 2
        ERROR = 3

    class Status(Enum):
        FINISHED_ABORTED = 0
        FINISHED_ERROR = 1

    class Output:
        def __init__(self, request_id, new_token_ids, finish_reason=None, **kw):
            self.request_id = request_id
            self.new_token_ids = new_token_ids
            self.finish_reason = finish_reason
            self.multimodal_output = None
            self.__dict__.update(kw)

    class Outputs:
        def __init__(self, outputs):
            self.outputs = outputs
            self.finished_requests = None

    class Base:
        def __init__(self):
            self.running = []
            self.requests = {}
            self.max_num_active_reqs = 2
            self.next_schedule = SimpleNamespace(scheduled_new_reqs=[], preempted_req_ids=set())

        def update_from_output(self, _, result):
            return result

        def schedule(self, _):
            self.observed_limit = self.max_num_active_reqs
            return self.next_schedule

        def has_requests(self):
            return False

        def has_unfinished_requests(self):
            return False

        def finish_requests(self, ids, status):
            return []

    _module(monkeypatch, "vllm.v1.engine", FinishReason=Reason)
    _module(monkeypatch, "vllm.v1.request", RequestStatus=Status)
    _module(monkeypatch, "vllm_omni.core.sched.omni_ar_scheduler", OmniARAsyncScheduler=Base)
    _module(monkeypatch, "vllm_omni.engine", OmniEngineCoreOutput=Output, OmniEngineCoreOutputs=Outputs)
    _module(monkeypatch, "vllm_omni.platforms", current_omni_platform=SimpleNamespace())
    pkg = "_cosy_codec_protocol"
    _module(monkeypatch, pkg, __path__=[str(ROOT)])
    _module(monkeypatch, pkg + ".pipeline_adapter", SingleStageAdapter=object, CodecRequestKey=tuple[str, int])
    _module(
        monkeypatch,
        pkg + ".cosyvoice3_code2wav",
        StreamingHiFTState=dict,
        iter_decode_streaming_batch=lambda *a: iter(()),
    )
    loaded = []
    for name in ("async_codec", "async_scheduler"):
        full = pkg + "." + name
        spec = importlib.util.spec_from_file_location(full, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return SimpleNamespace(
        codec=loaded[0],
        scheduler=loaded[1],
        Reason=Reason,
        Status=Status,
        Output=Output,
        Outputs=Outputs,
    )


def _scheduler(m):
    s = m.scheduler.CosyVoice3AsyncScheduler()

    class Codec:
        def __init__(self):
            self.outputs = queue.SimpleQueue()
            self.generations = {"r": 1}
            self.released, self.cancelled = [], []

        def check_error(self):
            pass

        def register(self, req_id):
            self.generations[req_id] = self.generations.get(req_id, 0) + 1
            return self.generations[req_id]

        def release(self, req_id):
            self.released.append(req_id)

        def cancel(self, req_id):
            self.cancelled.append(req_id)

    s.codec = Codec()
    s.audio_sample_rate = 24000
    s.audio_routes["r"] = 7
    s.audio_generations["r"] = 1
    return s


def _terminal(m, s):
    out = m.Output("r", [6561], m.Reason.STOP)
    batch = m.Outputs([out])
    batch.finished_requests = {"r"}
    result = s.update_from_output(None, {7: batch})
    assert not result[7].outputs
    assert not result[7].finished_requests
    return out


def test_eos_frees_ar_but_defers_client_finish_until_host_audio(modules):
    m = modules
    s = _scheduler(m)
    terminal = _terminal(m, s)
    assert s.has_requests()  # Engine must continue without an AR batch.
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.tensor([1.0, 2.0]), True, 1))
    out = s.update_from_output(None, {})[7]
    assert out.outputs == [terminal]
    assert terminal.multimodal_output["audio"].tolist() == [1.0, 2.0]
    assert out.finished_requests == {"r"}
    assert s.codec.released == ["r"]
    assert not s.has_requests()


def test_audio_can_finish_before_engine_consumes_eos(modules):
    m = modules
    s = _scheduler(m)
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.ones(2), True, 1))
    first = s.update_from_output(None, {})[7].outputs[0]
    assert first.finish_reason is None
    assert not s.codec.released
    terminal = m.Output("r", [6561], m.Reason.STOP)
    result = s.update_from_output(None, {7: m.Outputs([terminal])})
    assert result[7].outputs[0].finish_reason == m.Reason.STOP
    assert s.codec.released == ["r"]


def test_multiple_ready_chunks_coalesce_without_duplicate_request_outputs(modules):
    m = modules
    s = _scheduler(m)
    for n in (1.0, 2.0):
        s.codec.outputs.put(m.codec.ReadyAudio("r", torch.tensor([n]), False, 1))
    tokens = m.Output("r", [12])
    batch = s.update_from_output(None, {7: m.Outputs([tokens])})[7]
    assert len(batch.outputs) == 1
    assert batch.outputs[0].new_token_ids == [12]
    assert batch.outputs[0].multimodal_output["audio"].tolist() == [1.0, 2.0]


def test_stale_audio_cannot_reach_reused_request_id(modules):
    m = modules
    s = _scheduler(m)
    s.audio_generations["r"] = 2
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.ones(1), True, 1))
    assert s.update_from_output(None, {}) == {}
    assert not s.audio_done


def test_cancel_detached_tail_does_not_hang_or_emit_pcm(modules):
    m = modules
    s = _scheduler(m)
    _terminal(m, s)
    s.finish_requests(["r"], m.Status.FINISHED_ABORTED)
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.ones(1), True, 1))
    output = s.update_from_output(None, {})[7].outputs[0]
    assert output.finish_reason == m.Reason.ABORT
    assert output.multimodal_output is None
    assert not s.has_requests()


def test_detached_tail_reserves_admission_capacity(modules):
    s = _scheduler(modules)
    s.schedule()
    assert s.observed_limit == 1
    assert s.max_num_active_reqs == 2
