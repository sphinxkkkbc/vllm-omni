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


class VocoderGraphHandle:
    """Opaque runtime endpoint for one bound vocoder CUDA Graph Target.

    The Handle type is shared by the model-facing Target and worker-side
    Manager, but Handle instances are manager-created and manager-owned. It
    deliberately hides graph entries, descriptors, replay buffers, and
    dispatch policy from model code.
    """

    __slots__ = ("_call",)

    def __init__(self, call: Callable[..., Any]) -> None:
        self._call = call

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)


class VocoderCUDAGraphRoutine(Protocol):
    """Model-specific adapter for graph-shape resolution and static-buffer handling.

    A Routine owns graph-shape resolution and static-buffer adaptation. Request/model
    semantic state transitions are generally kept in the model execution path to
    preserve ownership boundaries.

    Runtime ``args`` and ``kwargs`` provide context for descriptor selection,
    replay-input preparation, and logical output materialization.
    """

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
    ) -> object:
        """Allocate descriptor-owned static buffers used by capture and replay.

        The returned object is owned by the graph entry. Request-local semantic
        state should generally remain model-owned rather than being transferred
        into the graph-buffer lifecycle.
        """
        ...

    def prepare_for_capture(self, buffers: object) -> None:
        """Prepare graph-owned buffers for warmup/capture.

        This hook is intended for initializing or normalizing static graph state
        required by the captured callable. Runtime invocation-specific semantics
        are preferably kept outside the capture lifecycle.
        """
        ...

    def forward_for_capture(self, buffers: object) -> object: ...

    def copy_runtime_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
    ) -> None:
        """Adapt one runtime invocation into descriptor-owned replay buffers.

        ``args`` and ``kwargs`` provide the runtime context needed to populate or
        normalize static graph buffers. Request/model semantic state transitions
        are preferably kept in the model execution path rather than coupled to
        replay-buffer preparation.
        """
        ...

    def output_after_replay(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
        captured_output: object,
    ) -> object:
        """Materialize a runtime-visible result from captured graph output.

        ``args`` and ``kwargs`` provide runtime context such as the logical batch
        size or output extent. Implementations may use them to select, slice, or
        clone graph-owned outputs. Request/model semantic state commit is generally
        better kept in the model execution path to preserve ownership boundaries.
        """
        ...


class BaseVocoderCUDAGraphRoutine:
    """Model-specific adapter between runtime calls and static CUDA Graph buffers.

    A Routine owns graph-shape resolution and static-buffer adaptation.
    Request/model semantic state transitions are generally kept in the model
    execution path to preserve ownership boundaries. Runtime ``args`` and
    ``kwargs`` provide context for descriptor selection, replay-input
    preparation, and logical output materialization.
    """

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


class VocoderCUDAGraphTarget:
    """Resolved planning declaration and stable model-owned call site.

    Before Manager binding, the delegate is ``Routine.eager_call``. After
    binding, it is the Manager-created ``VocoderGraphHandle``. Restoring or
    clearing the Target returns the delegate to ``Routine.eager_call``.
    """

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
        # The Target owns the stable call site, not the runtime Handle
        # lifecycle. The Manager constructs and binds the Handle after capture.
        self._bound_handle: VocoderGraphHandle | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate(*args, **kwargs)

    def _bind_handle(self, handle: VocoderGraphHandle) -> None:
        if not isinstance(handle, VocoderGraphHandle):
            raise TypeError("VocoderCUDAGraphTarget requires a VocoderGraphHandle")
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
