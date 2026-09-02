# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-facing contracts for runner-owned vocoder CUDA Graphs.

This module intentionally contains no worker lifecycle implementation. Model
packages depend on these declarations; the worker-side manager consumes them.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable, Sequence, Set
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Literal, Protocol, TypeVar, runtime_checkable

import torch

VariantT = TypeVar("VariantT", bound=Hashable)


@dataclass(frozen=True)
class VocoderCUDAGraphDescriptor(Generic[VariantT]):
    """One immutable, Target-scoped capture specialization."""

    variant: VariantT


@dataclass(frozen=True)
class VocoderRuntimeKey(Generic[VariantT]):
    """Actual runtime input identity before capture-bucket selection."""

    variant: VariantT


@dataclass(frozen=True)
class VocoderRuntimeResolution:
    """A valid runtime invocation mapped to an available Descriptor, if any."""

    runtime_key: VocoderRuntimeKey
    descriptor: VocoderCUDAGraphDescriptor | None


@dataclass
class VocoderCUDAGraphEntry:
    """Manager-owned graph resources for one Target-local Descriptor."""

    descriptor: VocoderCUDAGraphDescriptor
    graph: torch.cuda.CUDAGraph
    buffers: object
    captured_output: object
    # Manager-internal guard for this entry's mutable static replay buffers.
    # It is deliberately not part of equality/identity and is not exposed to
    # model code through VocoderGraphHandle.
    replay_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class VocoderGraphCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class VocoderCUDAGraphRoutine(Protocol):
    """Model-specific interpreter for one exact eager runnable."""

    target_id: str

    @property
    def runnable(self) -> Callable[..., Any]: ...

    def eager_call(self, *args: Any, **kwargs: Any) -> Any: ...

    def validate_runtime_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None: ...

    def resolve_runtime(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        available: Set[VocoderCUDAGraphDescriptor],
    ) -> VocoderRuntimeResolution: ...

    def make_lazy_descriptor(
        self,
        runtime_key: VocoderRuntimeKey,
    ) -> VocoderCUDAGraphDescriptor | None: ...

    def allocate_buffers(
        self,
        descriptor: VocoderCUDAGraphDescriptor,
        device: torch.device,
    ) -> object: ...

    def prepare_for_capture(self, buffers: object) -> None: ...

    def forward_for_capture(self, buffers: object) -> object: ...

    def copy_runtime_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
    ) -> None: ...

    def output_after_replay(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
        captured_output: object,
    ) -> object: ...

    def reset_after_capture(self, buffers: object) -> None: ...


class BaseVocoderCUDAGraphRoutine:
    """Optional no-op defaults that do not interpret model-specific data."""

    def validate_runtime_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del args, kwargs

    def make_lazy_descriptor(
        self,
        runtime_key: VocoderRuntimeKey,
    ) -> VocoderCUDAGraphDescriptor | None:
        return VocoderCUDAGraphDescriptor(runtime_key.variant)

    def prepare_for_capture(self, buffers: object) -> None:
        del buffers

    def reset_after_capture(self, buffers: object) -> None:
        del buffers


class VocoderCUDAGraphTarget:
    """Resolved planning declaration and stable model-owned call site."""

    def __init__(
        self,
        target_id: str,
        routine: VocoderCUDAGraphRoutine,
        descriptors: Sequence[VocoderCUDAGraphDescriptor],
        clone_output: bool = True,
        *,
        supported_config_keys: Set[str] = frozenset(),
    ) -> None:
        if routine.target_id != target_id:
            raise ValueError("Target and Routine target_id must match")
        self.target_id = target_id
        self.routine = routine
        self.descriptors = tuple(descriptors)
        self.clone_output = bool(clone_output)
        self.supported_config_keys = frozenset(supported_config_keys)
        self._delegate: Callable[..., Any] = routine.eager_call
        self._bound_handle: VocoderGraphCallable | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate(*args, **kwargs)

    def _bind_handle(self, handle: VocoderGraphCallable) -> None:
        if self._bound_handle is not None:
            raise RuntimeError(f"Target already bound: {self.target_id}")
        self._bound_handle = handle
        self._delegate = handle

    def _restore_eager(self) -> None:
        self._delegate = self.routine.eager_call
        self._bound_handle = None


@runtime_checkable
class SupportsVocoderCUDAGraph(Protocol):
    supports_vocoder_cudagraph: ClassVar[Literal[True]]

    def get_vocoder_cudagraph_targets(self) -> Sequence[VocoderCUDAGraphTarget]: ...


def supports_vocoder_cudagraph(model: object) -> bool:
    return bool(getattr(model, "supports_vocoder_cudagraph", False)) and callable(
        getattr(model, "get_vocoder_cudagraph_targets", None)
    )
