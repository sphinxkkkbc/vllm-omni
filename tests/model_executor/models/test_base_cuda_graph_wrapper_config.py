# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for shared CUDA Graph wrapper configuration policy."""

from types import SimpleNamespace
from typing import cast

from vllm.config import VllmConfig

from vllm_omni.model_executor.cuda_graph_wrapper import BaseCUDAGraphWrapper


class ConfigOnlyCUDAGraphWrapper(BaseCUDAGraphWrapper[tuple[int]]):
    def get_capture_keys(self) -> list[tuple[int]]:
        return []

    def get_static_call_args(self, key: tuple[int], *args, **kwargs):
        return (), {}

    def prepare_runtime_input(self, key, *args, **kwargs) -> None:
        pass

    def select_runtime_key(self, *args, **kwargs) -> tuple[int] | None:
        return None


def _vllm_config(*, enforce_eager: bool):
    return cast(VllmConfig, SimpleNamespace(model_config=SimpleNamespace(enforce_eager=enforce_eager)))


def test_enabled_defaults_to_constructor_flag_without_vllm_config():
    assert ConfigOnlyCUDAGraphWrapper(enabled=True).enabled is True
    assert ConfigOnlyCUDAGraphWrapper(enabled=False).enabled is False


def test_enabled_combines_constructor_flag_and_vllm_enforce_eager():
    assert ConfigOnlyCUDAGraphWrapper(enabled=True, vllm_config=_vllm_config(enforce_eager=False)).enabled is True
    assert ConfigOnlyCUDAGraphWrapper(enabled=True, vllm_config=_vllm_config(enforce_eager=True)).enabled is False
    assert ConfigOnlyCUDAGraphWrapper(enabled=False, vllm_config=_vllm_config(enforce_eager=False)).enabled is False
