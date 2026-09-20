# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise AuK graph eviction through the live Speech API.

The worker hook observes real captures and replays, and the negative case owns
the freed workspace allocation in that same process. No CUDA pointers cross RPC.
"""

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from openai import APIStatusError

from tests.helpers.client import OnlineOmniClient
from tests.helpers.fixtures.runtime import omni_fixture_lock
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, iter_omni_server
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("VLLM_OMNI_AUK_MODEL_DIR")
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.tts,
    pytest.mark.skipif(not MODEL, reason="set VLLM_OMNI_AUK_MODEL_DIR to an assembled AuK checkpoint"),
]


def _install_graph_probe():
    """Observe worker graphs and optionally occupy the evicted workspace."""
    import json
    import os

    import torch

    from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper

    original_call = AuKCUDAGraphWrapper.__call__

    def emit(event, **fields):
        with open(os.environ["AUK_GRAPH_EVENTS"], "a") as stream:
            stream.write(json.dumps({"event": event, **fields}) + "\n")

    def occupy_workspace(wrapper):
        address = wrapper._test_workspace_address
        size = int(torch._C._cuda_getCublasWorkspaceSize())
        block = next(
            block
            for segment in torch.cuda.memory._snapshot()["segments"]
            if segment.get("segment_pool_id") == wrapper._pool_handle
            for block in segment["blocks"]
            if block["address"] <= address < block["address"] + block["size"]
        )
        if block["state"] != "inactive":
            raise RuntimeError("AuK regression: evicted workspace was not released")
        graph = torch.cuda.CUDAGraph()
        prefix = None
        with torch.cuda.graph(graph, pool=wrapper._pool_handle):
            if address > block["address"]:
                prefix = torch.empty(address - block["address"], dtype=torch.uint8, device="cuda")
            allocation = torch.empty(size, dtype=torch.uint8, device="cuda")
            allocation.fill_(0xA5)
        if allocation.data_ptr() != address:
            emit("occupation_missed", expected_address=address, actual_address=allocation.data_ptr())
            raise RuntimeError("AuK regression: failed to occupy the exact evicted workspace address")
        graph.replay()
        torch.accelerator.synchronize()
        wrapper._test_occupation = (graph, prefix, allocation, allocation.clone())
        emit("occupied", address=address, size=size)

    def call(self, **kwargs):
        if not hasattr(self, "_test_workspace_address"):
            self._test_workspace_address = None
            self._test_occupation = None
            emit("installed", max_graphs=self.max_graphs)
        # The test arms occupation only after the eviction request succeeds.
        # This also keeps multi-step Flash requests outside the error boundary.
        if (
            os.environ["AUK_GRAPH_MODE"] == "occupy"
            and self._test_occupation is None
            and os.path.exists(os.environ["AUK_GRAPH_EVENTS"] + ".occupy")
        ):
            occupy_workspace(self)
        before = tuple(self._cache)
        result = original_call(self, **kwargs)
        torch.accelerator.synchronize()
        after = tuple(self._cache)
        captured = [key for key in after if key not in before]
        evicted = [key for key in before if key not in after]
        emit("capture" if captured else "replay", key=after[-1])
        if os.environ["AUK_GRAPH_MODE"] == "occupy":
            if self._test_workspace_address is None:
                size = int(torch._C._cuda_getCublasWorkspaceSize())
                candidates = [
                    block["address"]
                    for segment in torch.cuda.memory._snapshot()["segments"]
                    if segment.get("segment_pool_id") == self._pool_handle
                    for block in segment["blocks"]
                    if block["state"] == "active_allocated" and block.get("requested_size") == size
                ]
                if len(candidates) != 1:
                    raise RuntimeError(f"AuK regression: expected one capture workspace, got {candidates}")
                self._test_workspace_address = candidates[0]
            if self._test_occupation is not None:
                _, _, allocation, expected = self._test_occupation
                if not torch.equal(allocation, expected):
                    changed_bytes = int(torch.count_nonzero(allocation != expected).item())
                    emit("corruption", address=allocation.data_ptr(), key=after[-1], changed_bytes=changed_bytes)
                    raise RuntimeError("AuK CUDA graph replay corrupted the occupied workspace")
        for key in evicted:
            emit("evict", key=key)
        return result

    AuKCUDAGraphWrapper.__call__ = call


@pytest.fixture
def graph_server(request, tmp_path, run_level):
    # Copy only the worker helper into Python's subprocess startup module;
    # importing the entire test module there would also initialize the harness.
    hook = tmp_path / "worker_hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(inspect.getsource(_install_graph_probe) + "\n_install_graph_probe()\n")
    deploy = yaml.safe_load(Path(get_deploy_config_path("auk.yaml")).read_text())
    stage = next(stage for stage in deploy["stages"] if stage["stage_id"] == 1)
    stage.setdefault("model_config", {})["max_dit_graphs"] = 3
    deploy_path = tmp_path / "auk.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy))
    events = tmp_path / "events.jsonl"
    params = OmniServerParams(
        model=MODEL or ".",
        stage_config_path=str(deploy_path),
        server_args=["--trust-remote-code", "--disable-log-stats"],
        env_dict={
            "PYTHONPATH": os.pathsep.join(
                (str(hook), str(Path(__file__).resolve().parents[3]), os.getenv("PYTHONPATH", ""))
            ),
            "AUK_GRAPH_EVENTS": str(events),
            "AUK_GRAPH_MODE": request.param,
            "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
        },
    )
    server_request = SimpleNamespace(param=params, node=request.node)
    for server in iter_omni_server(server_request, run_level, omni_fixture_lock):
        client = OnlineOmniClient(
            host=server.host, port=server.port, api_key="EMPTY", run_level=run_level, log_stats=server.log_stats
        )
        yield server, client, events


def _request(server, duration):
    return {
        "model": server.model,
        "input": "",
        "instructions": 'Generate speech based on the following description: "A clear, natural voice.". '
        'The content to speak is: "Hello, this is an AuK graph regression test.".',
        "voice": "default",
        "duration_seconds": duration,
        "seed": 7,
        "extra_params": {"t_grid": [0.0, 1.0]},
        "response_format": "wav",
        "timeout": 300,
    }


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("graph_server", ["normal"], indirect=True)
def test_auk_retained_graph_replays_after_eviction_and_new_capture(graph_server):
    server, client, path = graph_server
    # Capture A/B/C/D (evict A), capture E, then replay retained C.
    # 50 latent frames/second: A through E occupy five distinct buckets.
    for duration in (1.0, 2.0, 3.0, 4.0, 6.0, 3.0):
        client.send_audio_speech_request(_request(server, duration))
    events = _events(path)
    assert any(event["event"] == "installed" and event["max_graphs"] == 3 for event in events)
    captures = [event["key"][0] for event in events if event["event"] == "capture"]
    assert captures == [64, 128, 192, 256, 320]
    eviction = next(i for i, event in enumerate(events) if event["event"] == "evict")
    assert events[eviction]["key"][0] == 64
    capture_e = next(i for i, event in enumerate(events) if event["event"] == "capture" and event["key"][0] == 320)
    assert capture_e > eviction
    assert events[-1]["event"] == "replay" and events[-1]["key"][0] == 192


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("graph_server", ["occupy"], indirect=True)
def test_auk_replay_detects_occupied_workspace_corruption(graph_server):
    server, client, path = graph_server
    # Do not retry a failed request: replay mutates the graph/workspace state.
    client.client.max_retries = 0
    # A/B/C capture successfully; D captures and evicts A. All steps of these
    # requests must succeed, including the fixed multi-step Flash schedule.
    for duration in (1.0, 2.0, 3.0, 4.0):
        client.send_audio_speech_request(_request(server, duration))
    events = _events(path)
    assert [event["key"][0] for event in events if event["event"] == "capture"] == [64, 128, 192, 256]
    assert [event["key"][0] for event in events if event["event"] == "evict"] == [64]
    assert not any(event["event"] in {"occupied", "corruption"} for event in events)
    # Request 5: capture an unrelated operation at A's freed address, then
    # replay retained B. Only this request is allowed to raise the target error.
    Path(str(path) + ".occupy").touch()
    boundary = len(events)
    with pytest.raises(APIStatusError, match="AuK CUDA graph replay corrupted the occupied workspace"):
        try:
            client.send_audio_speech_request(_request(server, 2.0))
        except APIStatusError as error:
            if "AuK regression: failed to occupy the exact evicted workspace address" in str(error):
                request_events = _events(path)[boundary:]
                missed = next((event for event in request_events if event["event"] == "occupation_missed"), None)
                if missed is not None:
                    pytest.skip(
                        "Allocator did not reuse the workspace address: "
                        f"expected {missed['expected_address']:#x}, got {missed['actual_address']:#x}"
                    )
            raise
    events = _events(path)[boundary:]
    occupied = next(event for event in events if event["event"] == "occupied")
    corruption = next(event for event in events if event["event"] == "corruption")
    assert corruption["address"] == occupied["address"]
    assert corruption["key"][0] == 128
    assert corruption["changed_bytes"] > 0
    assert events.index(corruption) > events.index(occupied)
    assert any(event["event"] == "replay" and event["key"][0] == 128 for event in events)
    assert not any(event["event"] == "capture" for event in events)
