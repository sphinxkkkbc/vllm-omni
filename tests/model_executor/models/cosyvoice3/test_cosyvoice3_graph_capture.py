# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Codec captures must not prohibit independent AR-thread CUDA allocations."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.graph_capture import capture_codec_graph

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def streams(monkeypatch):
    calls = []
    caller = SimpleNamespace(wait_stream=lambda stream: calls.append("caller_waits_capture"))
    capture = SimpleNamespace(wait_stream=lambda stream: calls.append("capture_waits_caller"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    return capture, calls


@pytest.mark.parametrize("pool", [None, (11, 22)])
def test_capture_uses_private_stream_without_global_preamble(monkeypatch, pool, streams):
    calls = []
    graph = SimpleNamespace(
        capture_begin=lambda **kw: calls.append(("begin", kw)),
        capture_end=lambda: calls.append(("end", None)),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Codec capture must not enter the global graph wrapper or synchronize the device")

    monkeypatch.setattr(torch.cuda, "graph", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)
    with capture_codec_graph(graph, pool, stream=streams[0]):
        calls.append(("body", None))
    assert streams[1] == ["capture_waits_caller", "caller_waits_capture"]
    assert calls == [
        ("begin", {"pool": pool, "capture_error_mode": "thread_local"}),
        ("body", None),
        ("end", None),
    ]


def test_capture_ends_before_propagating_body_error(streams):
    ended = []
    graph = SimpleNamespace(capture_begin=lambda **kw: None, capture_end=lambda: ended.append(True))
    with pytest.raises(ValueError, match="capture body failed"):
        with capture_codec_graph(graph, stream=streams[0]):
            raise ValueError("capture body failed")
    assert ended == [True]


def test_failed_capture_begin_does_not_attempt_end(streams):
    ended = []

    def begin(**kwargs):
        raise RuntimeError("begin failed")

    graph = SimpleNamespace(capture_begin=begin, capture_end=lambda: ended.append(True))
    with pytest.raises(RuntimeError, match="begin failed"):
        with capture_codec_graph(graph, stream=streams[0]):
            pytest.fail("body must not run")
    assert ended == []
