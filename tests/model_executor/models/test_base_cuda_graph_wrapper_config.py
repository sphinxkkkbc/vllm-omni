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


def test_enabled_defaults_to_true_without_vllm_config():
    assert ConfigOnlyCUDAGraphWrapper().enabled is True


def test_enabled_uses_vllm_enforce_eager():
    assert ConfigOnlyCUDAGraphWrapper(vllm_config=_vllm_config(enforce_eager=False)).enabled is True
    assert ConfigOnlyCUDAGraphWrapper(vllm_config=_vllm_config(enforce_eager=True)).enabled is False


def test_adapter_can_override_vllm_enable_policy():
    class AdapterPolicyWrapper(ConfigOnlyCUDAGraphWrapper):
        def is_enabled(self, vllm_config: VllmConfig | None) -> bool:
            return False

    assert AdapterPolicyWrapper(vllm_config=_vllm_config(enforce_eager=False)).enabled is False
