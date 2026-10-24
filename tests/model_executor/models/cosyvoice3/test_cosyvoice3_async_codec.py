# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU protocol tests for completion ordering, cancellation and producer isolation.

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

import numpy as np
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

        def on_requests_finished(self, ids):
            pass

    _module(monkeypatch, "vllm.v1.engine", FinishReason=Reason)
    _module(monkeypatch, "vllm.v1.request", RequestStatus=Status)
    _module(monkeypatch, "vllm_omni.core.sched.omni_ar_scheduler", OmniARAsyncScheduler=Base)
    _module(monkeypatch, "vllm_omni.engine", OmniEngineCoreOutput=Output, OmniEngineCoreOutputs=Outputs)
    _module(monkeypatch, "vllm_omni.platforms", current_omni_platform=SimpleNamespace())
    _module(monkeypatch, "vllm_omni.worker_v2.model_states.omni_model_state", OmniModelState=Base)
    pkg = "_cosy_codec_protocol"
    _module(monkeypatch, pkg, __path__=[str(ROOT)])
    _module(monkeypatch, pkg + ".pipeline_adapter", InnerStageAdapter=object)
    _module(
        monkeypatch,
        pkg + ".cosyvoice3_code2wav",
        StreamingHiFTState=dict,
        iter_decode_streaming_batch=lambda *a: iter(()),
    )
    _module(monkeypatch, "vllm_omni.model_executor.models.cosyvoice3.pipeline_adapter", InnerStageAdapter=object)
    loaded = []
    for name in ("async_codec", "async_scheduler", "model_state"):
        full = pkg + "." + name
        spec = importlib.util.spec_from_file_location(full, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        # model_state uses the public import path for its producer types.
        monkeypatch.setitem(sys.modules, "vllm_omni.model_executor.models.cosyvoice3." + name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return SimpleNamespace(
        codec=loaded[0],
        scheduler=loaded[1],
        state=loaded[2],
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
    s._audio_routes["r"] = 7
    s._audio_generations["r"] = 1
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
    s._audio_generations["r"] = 2
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.ones(1), True, 1))
    assert s.update_from_output(None, {}) == {}
    assert not s._audio_done


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


def test_worker_publishes_only_after_d2h_completion(modules):
    m = modules
    worker = object.__new__(m.codec.AsyncCodec)
    worker.outputs = queue.SimpleQueue()
    worker._cancelled = set()
    worker._released = set()
    worker._reads = {}
    from collections import deque

    ready = SimpleNamespace(value=False)
    event = SimpleNamespace(query=lambda: ready.value)
    audio = m.codec.ReadyAudio("r", torch.ones(1), True, 1)
    worker._pending = deque([m.codec.PendingAudio(audio, event, torch.ones(1))])
    worker._publish()
    assert worker.outputs.empty()
    assert len(worker._pending) == 1  # GPU source remains owned until copy completion.
    ready.value = True
    worker._publish()
    assert worker.outputs.get_nowait() is audio


def test_producer_snapshots_metadata_without_cpu_readback(modules, monkeypatch):
    m = modules
    state = object.__new__(m.state.CosyVoice3ModelState)
    state.adapter = SimpleNamespace(vocab_size=6561)
    state._max_model_len = 32
    state._req_states = SimpleNamespace(max_seq_len=np.array([32]), prompt_len=SimpleNamespace(np=np.array([4])))
    payload = {"embed": {}}
    state.intermediate_buffer = SimpleNamespace(gather=lambda batch: [payload])
    submitted = []
    state._codec = SimpleNamespace(generations={"r": 3}, submit=lambda *args: submitted.append(args))
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r"],
        idx_mapping_np=np.array([0]),
        num_scheduled_tokens=np.array([1]),
        num_computed_prefill_tokens_np=np.array([4]),
        prefill_len_np=np.array([4]),
        is_prefilling_np=np.array([False]),
        num_computed_tokens_np=np.array([4]),
    )

    def unexpected(*args, **kw):
        pytest.fail("Runner must not read sampled GPU control data on the CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", unexpected)
    outputs = {"embed": {"embedding": [torch.ones(1, 192)]}}
    state.run_eager_mtp(batch, torch.zeros(1, 8), torch.tensor([[6561]]), outputs)
    batch.req_ids[0] = "reused"
    batch.num_computed_tokens_np[0] = 99
    entries, tokens, speech, rows = submitted[0]
    assert entries[0].request_id == "r"
    assert entries[0].position == 1
    assert entries[0].generation == 3
    assert rows == [0]
    assert tokens.tolist() == [6561]
    assert speech.tolist() == [False]
    assert outputs == {}  # AR future has no pending PCM dependency.


def test_producer_passes_cpu_row_mapping_without_tensor_advanced_indexing(modules, monkeypatch):
    m = modules
    state = object.__new__(m.state.CosyVoice3ModelState)
    state.adapter = SimpleNamespace(vocab_size=100)
    state._max_model_len = 32
    state._req_states = SimpleNamespace(max_seq_len=np.full(5, 32), prompt_len=SimpleNamespace(np=np.full(5, 4)))
    state.intermediate_buffer = SimpleNamespace(gather=lambda batch: [{} for _ in range(5)])
    submitted = []
    state._codec = SimpleNamespace(generations={"b": 2, "a": 1}, submit=lambda *args: submitted.append(args))
    batch = SimpleNamespace(
        num_reqs=5,
        req_ids=["prefill", "b", "_warmup_0", "a", "retired"],
        idx_mapping_np=np.arange(5),
        num_scheduled_tokens=np.ones(5, dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([1, 4, 4, 4, 4]),
        prefill_len_np=np.full(5, 4),
        is_prefilling_np=np.array([True, False, False, False, False]),
        num_computed_tokens_np=np.array([1, 3, 3, 3, 3]),
    )
    original = torch.Tensor.__getitem__

    def basic_index_only(tensor, index):
        parts = index if isinstance(index, tuple) else (index,)
        if any(isinstance(part, (list, np.ndarray, torch.Tensor)) for part in parts):
            pytest.fail("CPU row metadata must not be used for CUDA advanced indexing")
        return original(tensor, index)

    monkeypatch.setattr(torch.Tensor, "__getitem__", basic_index_only)
    sampled = torch.tensor([[99], [12], [77], [100], [55]])
    outputs = {"audio": []}
    state.run_eager_mtp(batch, torch.zeros(5, 8), sampled, outputs)
    entries, tokens, speech, rows = submitted[0]
    assert rows == [1, 3]
    assert [(entry.request_id, entry.generation) for entry in entries] == [("b", 2), ("a", 1)]
    assert tokens.untyped_storage().data_ptr() == sampled.untyped_storage().data_ptr()
    assert tokens.tolist() == [99, 12, 77, 100, 55]
    assert speech.tolist() == [True, True, True, False, True]
    assert outputs == {}


@pytest.fixture
def per_request_worker(modules, monkeypatch):
    from collections import deque
    from contextlib import nullcontext

    class Event:
        def __init__(self):
            self.done = False

        def query(self):
            return self.done

        def record(self, stream):
            pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    empty = torch.empty

    def host_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", host_empty)
    worker = object.__new__(modules.codec.AsyncCodec)
    worker._pending = deque()
    worker._cancelled, worker._released, worker._ended = set(), set(), set()
    worker._reads = {}
    worker.outputs = queue.SimpleQueue()
    worker.stream = object()
    worker.copy_stream = SimpleNamespace(wait_event=lambda event: None)
    worker.applied = []
    worker._progress, states = {}, {}
    for index, key in enumerate([("a", 1), ("b", 2)]):
        worker._progress[key] = modules.codec.DecodeProgress(
            SimpleNamespace(tokens=torch.ones(1, 1, dtype=torch.long), conditioning={}), 1, int(index == 0), index == 1
        )
        states[key] = SimpleNamespace(
            prompt_token=torch.empty(1, 0),
            prompt_feat=torch.empty(1, 0, 80),
            embedding=torch.ones(1, 192),
            finished=False,
            emitted=0,
            hop=1,
        )

    def prepare(key, length, final):
        states[key].finished = final
        return {"key": key}

    worker.adapter = SimpleNamespace(
        states=states,
        chunk_size=1,
        lookahead=0,
        initialize=lambda *args: None,
        prepare=prepare,
        apply_result=lambda *args: worker.applied.append(args),
    )
    worker.model = SimpleNamespace(code2wav=object())
    return worker


def test_d2h_queued_per_request_without_waiting_for_previous_hift(per_request_worker, modules, monkeypatch):
    worker = per_request_worker
    shared = torch.ones(1)

    def decode(model, items):
        assert [item["key"][0] for item in items] == ["a", "b"]
        yield 0, shared, {"row": 0}
        # A already has a D2H before B submits, even though A's copy is not done.
        assert len(worker._pending) == 1
        assert not worker._pending[0].ready.done
        shared.fill_(2)
        yield 1, shared, {"row": 1}

    monkeypatch.setattr(modules.codec, "iter_decode_streaming_batch", decode)
    assert worker._decode_ready()
    assert worker.outputs.empty()
    assert [entry.output.request_id for entry in worker._pending] == ["a", "b"]
    assert [entry.source.tolist() for entry in worker._pending] == [[1], [2]]
    for entry in worker._pending:
        entry.ready.done = True
    worker._publish()
    first, last = worker.outputs.get_nowait(), worker.outputs.get_nowait()
    assert first.pcm.tolist() == [1] and not first.final
    assert last.pcm.tolist() == [2] and last.final


def test_completed_audio_published_between_hift_submissions(per_request_worker, modules, monkeypatch):
    worker = per_request_worker

    def decode(model, items):
        yield 0, torch.ones(1), None
        worker._pending[0].ready.done = True
        yield 1, torch.full((1,), 2.0), None
        # Publication is polled between rows; B's unfinished D2H cannot hold A.
        assert worker.outputs.get_nowait().request_id == "a"
        assert len(worker._pending) == 1
        assert not worker._pending[0].ready.done

    monkeypatch.setattr(modules.codec, "iter_decode_streaming_batch", decode)
    assert worker._decode_ready()
    assert worker._pending[0].output.request_id == "b"
