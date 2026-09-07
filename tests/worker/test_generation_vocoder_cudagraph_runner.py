# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_omni.worker import gpu_generation_model_runner as generation_runner_module
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _GraphModel:
    supports_vocoder_cudagraph = True

    def get_vocoder_cudagraph_targets(self):
        return ()


class _FakeManager:
    instances: list["_FakeManager"] = []

    def __init__(self, *, vllm_config, device) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.prepared_model = None
        self.cleared = False
        self.instances.append(self)

    def prepare(self, model) -> None:
        self.prepared_model = model

    def clear(self) -> None:
        self.cleared = True


def _runner(*, enforce_eager: bool, mode: CUDAGraphMode):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model_config = SimpleNamespace(enforce_eager=enforce_eager)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=mode)
    runner.vllm_config = SimpleNamespace()
    runner.device = torch.device("cpu")
    runner.model = _GraphModel()
    runner.vocoder_cudagraph_manager = None
    return runner


def test_load_model_prepares_manager_from_unwrapped_model(monkeypatch) -> None:
    _FakeManager.instances.clear()
    monkeypatch.setattr(OmniGPUModelRunner, "load_model", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(generation_runner_module, "VocoderCUDAGraphManager", _FakeManager)
    runner = _runner(enforce_eager=False, mode=CUDAGraphMode.FULL)

    GPUGenerationModelRunner.load_model(runner)

    assert len(_FakeManager.instances) == 1
    manager = _FakeManager.instances[0]
    assert manager.prepared_model is runner.model
    assert runner.vocoder_cudagraph_manager is manager


def test_enforce_eager_skips_manager_and_shutdown_restores_targets(monkeypatch) -> None:
    _FakeManager.instances.clear()
    monkeypatch.setattr(OmniGPUModelRunner, "load_model", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(generation_runner_module, "VocoderCUDAGraphManager", _FakeManager)
    runner = _runner(enforce_eager=True, mode=CUDAGraphMode.NONE)

    GPUGenerationModelRunner.load_model(runner)
    assert runner.vocoder_cudagraph_manager is None

    manager = _FakeManager(vllm_config=runner.vllm_config, device=runner.device)
    runner.vocoder_cudagraph_manager = manager
    shutdown_called = []
    monkeypatch.setattr(OmniGPUModelRunner, "shutdown", lambda self: shutdown_called.append(True))
    GPUGenerationModelRunner.shutdown(runner)

    assert manager.cleared
    assert runner.vocoder_cudagraph_manager is None
    assert shutdown_called == [True]
