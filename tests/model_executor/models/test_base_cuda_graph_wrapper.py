# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared CUDA Graph wrapper lifecycle."""

import pytest
import torch

from vllm_omni.model_executor.cuda_graph_wrapper import BaseCUDAGraphWrapper

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

DEVICE = torch.device("cuda:0")


class ToyCUDAGraphWrapper(BaseCUDAGraphWrapper[tuple[int]]):
    """Tiny adapter that exercises the base capture/replay state machine."""

    def __init__(
        self,
        *,
        capture_sizes: list[int] | None = None,
        capture_mode: str = "pre-capture",
        enabled: bool = True,
        fn=None,
    ):
        super().__init__(
            fn=fn or self._default_fn,
            enabled=enabled,
            capture_mode=capture_mode,
        )
        self.capture_sizes = capture_sizes or [4]
        self.events: list[str] = []
        self.prepare_context_calls = 0
        self.before_capture_calls = 0
        self.prepare_runtime_calls = 0
        self.eager_calls = 0

    @staticmethod
    def _default_fn(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 1.0

    def prepare_capture_context(self, **context_kwargs) -> None:
        self.prepare_context_calls += 1
        self.events.append("prepare_context")
        if "capture_sizes" in context_kwargs:
            self.capture_sizes = list(context_kwargs["capture_sizes"])

    def get_capture_keys(self) -> list[tuple[int]]:
        self.events.append("get_capture_keys")
        return [(size,) for size in self.capture_sizes]

    def before_capture(self, keys) -> None:
        self.before_capture_calls += 1
        self.events.append(f"before_capture:{len(list(keys))}")

    def get_static_call_args(self, key: tuple[int], *args, **kwargs):
        self.events.append(f"get_static:{key[0]}")
        if key not in self.static_inputs:
            self.static_inputs[key] = torch.zeros(key[0], device=self.device)
        return (self.static_inputs[key],), {}

    def select_runtime_key(self, x: torch.Tensor) -> tuple[int] | None:
        self.events.append("select_key")
        actual_size = int(x.numel())
        for size in self.capture_sizes:
            if actual_size <= size:
                return (size,)
        return None

    def prepare_runtime_input(self, key: tuple[int], x: torch.Tensor) -> None:
        self.prepare_runtime_calls += 1
        self.events.append(f"prepare_runtime:{key[0]}")
        static_input = self.static_inputs[key]
        actual_size = int(x.numel())
        static_input.zero_()
        static_input[:actual_size].copy_(x.reshape(-1))

    def postprocess(self, key: tuple[int], x: torch.Tensor):
        self.events.append(f"postprocess:{key[0]}")
        return self.static_outputs[key][: x.numel()]

    def run_eager(self, *args, **kwargs):
        self.eager_calls += 1
        self.events.append("run_eager")
        return super().run_eager(*args, **kwargs)


class NoFnCUDAGraphWrapper(ToyCUDAGraphWrapper):
    def __init__(self):
        super().__init__(fn=lambda x: x)
        self.fn = None


def _input(size: int) -> torch.Tensor:
    return torch.arange(size, device=DEVICE, dtype=torch.float32)


def _assert_replay_output_is_stable_clone(
    wrapper: BaseCUDAGraphWrapper,
    first_input: torch.Tensor,
    overwrite_input: torch.Tensor,
):
    out = wrapper.replay(first_input)
    expected = out.clone()
    key = wrapper.select_runtime_key(first_input)

    assert key is not None
    assert out is not wrapper.static_outputs[key]

    _ = wrapper.replay(overwrite_input)
    torch.testing.assert_close(out, expected)


def test_pre_capture_lifecycle_and_replay_hit():
    wrapper = ToyCUDAGraphWrapper(capture_sizes=[4, 8])

    wrapper.capture(device=DEVICE)

    assert wrapper.prepare_context_calls == 1
    assert wrapper.before_capture_calls == 1
    assert set(wrapper.graphs) == {(4,), (8,)}
    assert wrapper._warmed_up
    assert wrapper.events[:3] == [
        "prepare_context",
        "get_capture_keys",
        "before_capture:2",
    ]

    x = _input(3)
    out = wrapper.replay(x)

    torch.testing.assert_close(out, x * 2.0 + 1.0)
    assert wrapper.prepare_runtime_calls == 1
    assert wrapper.eager_calls == 0
    assert "prepare_runtime:4" in wrapper.events
    assert "postprocess:4" in wrapper.events


def test_default_run_eager_is_used_for_no_key_fallback():
    wrapper = ToyCUDAGraphWrapper(capture_sizes=[4])
    wrapper.capture(device=DEVICE)

    x = _input(6)
    out = wrapper.replay(x)

    torch.testing.assert_close(out, x * 2.0 + 1.0)
    assert wrapper.eager_calls == 1
    assert wrapper.prepare_runtime_calls == 0


def test_disabled_wrapper_falls_back_to_default_eager():
    wrapper = ToyCUDAGraphWrapper(capture_sizes=[4], enabled=False)

    x = _input(4)
    out = wrapper.replay(x)

    torch.testing.assert_close(out, x * 2.0 + 1.0)
    assert wrapper.eager_calls == 1
    assert not wrapper.graphs


def test_outer_stream_capture_falls_back_to_eager(monkeypatch):
    wrapper = ToyCUDAGraphWrapper(capture_sizes=[4])
    wrapper.capture(device=DEVICE)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    x = _input(4)
    out = wrapper.replay(x)

    torch.testing.assert_close(out, x * 2.0 + 1.0)
    assert wrapper.eager_calls == 1


def test_lazy_mode_skips_pre_capture_and_captures_on_first_replay():
    wrapper = ToyCUDAGraphWrapper(capture_sizes=[4], capture_mode="lazy")

    wrapper.capture(device=DEVICE)

    assert not wrapper.graphs
    assert not wrapper._warmed_up
    assert wrapper.prepare_context_calls == 0

    x = _input(3)
    out = wrapper.replay(x)

    torch.testing.assert_close(out, x * 2.0 + 1.0)
    assert set(wrapper.graphs) == {(4,)}
    assert wrapper.eager_calls == 0
    assert wrapper.prepare_runtime_calls == 1


def test_hybrid_mode_lazy_captures_runtime_miss():
    wrapper = ToyCUDAGraphWrapper(capture_sizes=[4], capture_mode="hybrid")
    wrapper.capture(device=DEVICE)

    wrapper.capture_sizes.append(8)
    x = _input(6)
    out = wrapper.replay(x)

    torch.testing.assert_close(out, x * 2.0 + 1.0)
    assert set(wrapper.graphs) == {(4,), (8,)}
    assert wrapper.eager_calls == 0
    assert wrapper.prepare_runtime_calls == 1


def test_pre_capture_missing_graph_falls_back_without_lazy_capture():
    wrapper = ToyCUDAGraphWrapper(capture_sizes=[4])
    wrapper.capture(device=DEVICE)

    wrapper.capture_sizes.append(8)
    x = _input(6)
    out = wrapper.replay(x)

    torch.testing.assert_close(out, x * 2.0 + 1.0)
    assert (8,) not in wrapper.graphs
    assert wrapper.eager_calls == 1


def test_invalid_capture_mode_raises():
    with pytest.raises(ValueError, match="Unsupported CUDA graph capture mode"):
        ToyCUDAGraphWrapper(capture_mode="invalid")


def test_fn_none_requires_adapter_overrides():
    wrapper = NoFnCUDAGraphWrapper()

    with pytest.raises(NotImplementedError, match="_warmup_for_key"):
        wrapper._warmup_for_key((4,))

    wrapper.capture(device=DEVICE)
    assert not wrapper.graphs
    with pytest.raises(NotImplementedError, match="run_eager"):
        wrapper.replay(_input(2))


def test_default_replay_clones_static_output():
    class DefaultPostprocessWrapper(ToyCUDAGraphWrapper):
        def postprocess(self, key: tuple[int], *args, **kwargs):
            return BaseCUDAGraphWrapper.postprocess(self, key, *args, **kwargs)

    wrapper = DefaultPostprocessWrapper(capture_sizes=[4])
    wrapper.capture(device=DEVICE)

    out = wrapper.replay(_input(4))

    assert out is not wrapper.static_outputs[(4,)]
    torch.testing.assert_close(out, wrapper.static_outputs[(4,)])


def test_replay_can_return_static_output_handle_when_clone_disabled():
    class DefaultPostprocessWrapper(ToyCUDAGraphWrapper):
        def postprocess(self, key: tuple[int], *args, **kwargs):
            return BaseCUDAGraphWrapper.postprocess(self, key, *args, **kwargs)

    wrapper = DefaultPostprocessWrapper(capture_sizes=[4])
    wrapper.capture(device=DEVICE)

    out = wrapper.replay(_input(4), clone_graph_output=False)

    assert out is wrapper.static_outputs[(4,)]


def test_default_replay_output_survives_later_replay():
    wrapper = ToyCUDAGraphWrapper(capture_sizes=[4])
    wrapper.capture(device=DEVICE)

    _assert_replay_output_is_stable_clone(
        wrapper,
        _input(4),
        _input(4) + 10.0,
    )
