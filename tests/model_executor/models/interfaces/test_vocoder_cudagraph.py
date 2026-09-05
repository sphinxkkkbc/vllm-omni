# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    BaseVocoderCUDAGraphRoutine,
    SupportsVocoderCUDAGraph,
    VocoderCUDAGraphDescriptor,
    VocoderCUDAGraphTarget,
    VocoderRuntimeKey,
    VocoderRuntimeResolution,
    supports_vocoder_cudagraph,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Routine(BaseVocoderCUDAGraphRoutine):
    target_id = "test.decode"

    @property
    def runnable(self):
        return self.eager_call

    def eager_call(self, value):
        return value + 1

    def validate_runtime_inputs(self, args, kwargs):
        del args, kwargs

    def resolve_runtime(self, args, kwargs, available):
        del args, kwargs, available
        return VocoderRuntimeResolution(VocoderRuntimeKey(0), None)

    def make_lazy_descriptor(self, runtime_key):
        del runtime_key
        return None

    def allocate_buffers(self, descriptor, device):
        del descriptor, device
        return None

    def prepare_for_capture(self, buffers):
        del buffers

    def forward_for_capture(self, buffers):
        del buffers
        return None

    def copy_runtime_inputs(self, args, kwargs, buffers):
        del args, kwargs, buffers

    def output_after_replay(self, args, kwargs, buffers, captured_output):
        del args, kwargs, buffers
        return captured_output

    def reset_after_capture(self, buffers):
        del buffers


def test_descriptor_is_hashable_and_target_namespace_is_local() -> None:
    first = VocoderCUDAGraphDescriptor((1, 25))
    second = VocoderCUDAGraphDescriptor((1, 25))

    assert first == second
    assert hash(first) == hash(second)
    assert first.variant == (1, 25)


def test_target_binds_same_callable_without_exposing_capture_lifecycle() -> None:
    target = VocoderCUDAGraphTarget(
        "test.decode",
        _Routine(),
        [VocoderCUDAGraphDescriptor(1)],
    )

    assert target(2) == 3
    assert not hasattr(target, "replay")
    assert not hasattr(target, "capture")


def test_capability_discovery_requires_both_declaration_and_provider() -> None:
    class Model(SupportsVocoderCUDAGraph):
        supports_vocoder_cudagraph = True

        def get_vocoder_cudagraph_targets(self):
            return ()

    assert supports_vocoder_cudagraph(Model())
    assert not supports_vocoder_cudagraph(object())


def test_runtime_key_is_not_a_capture_descriptor() -> None:
    runtime_key = VocoderRuntimeKey((3, 47))
    descriptor = VocoderCUDAGraphDescriptor((4, 50))

    assert runtime_key.variant != descriptor.variant
