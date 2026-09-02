# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Runner-owned lifecycle for model-declared vocoder CUDA Graph Targets."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import torch
from tqdm import tqdm
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    SupportsVocoderCUDAGraph,
    VocoderCUDAGraphDescriptor,
    VocoderCUDAGraphEntry,
    VocoderCUDAGraphTarget,
    VocoderRuntimeResolution,
)
from vllm_omni.worker.vocoder_cudagraph_handle import VocoderGraphHandle

logger = logging.getLogger(__name__)

_FRAMEWORK_CONFIG_KEYS = frozenset(
    {
        "max_memory_bytes",
        "log_stats",
        "stats_max_runtime_keys",
        "targets",
    }
)
_TARGET_POLICY_KEYS = frozenset(
    {
        "enabled",
        "enable_lazy_capture",
        "max_extra_graphs",
    }
)


def clone_tensor_tree(value: object) -> object:
    """Clone tensor leaves after model-specific graph output processing."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        cloned = [clone_tensor_tree(item) for item in value]
        if hasattr(value, "_fields"):
            constructor = cast(Callable[..., object], type(value))
            return constructor(*cloned)
        return tuple(cloned)
    if isinstance(value, list):
        return [clone_tensor_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tensor_tree(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class VocoderTargetRuntimeConfig:
    enabled: bool = True
    enable_lazy_capture: bool = False
    max_extra_graphs: int = 0


@dataclass
class ManagedTarget:
    target: VocoderCUDAGraphTarget
    entries: OrderedDict[VocoderCUDAGraphDescriptor, VocoderCUDAGraphEntry]
    enable_lazy_capture: bool
    failed_descriptors: set[VocoderCUDAGraphDescriptor] = field(default_factory=set)
    max_graphs: int | None = None


class _NoOpRecorder:
    __slots__ = ()

    def record_graph_hit(self, resolution: VocoderRuntimeResolution) -> None:
        del resolution

    def record_fallback(self, resolution: VocoderRuntimeResolution) -> None:
        del resolution

    def record_replay_error(self, resolution: VocoderRuntimeResolution) -> None:
        del resolution


class _TargetRecorder:
    __slots__ = ("_sink", "_target_id")

    def __init__(self, sink: VocoderGraphStatsSink, target_id: str) -> None:
        self._sink = sink
        self._target_id = target_id

    def record_graph_hit(self, resolution: VocoderRuntimeResolution) -> None:
        self._sink.record("hit", self._target_id, resolution)

    def record_fallback(self, resolution: VocoderRuntimeResolution) -> None:
        self._sink.record("fallback", self._target_id, resolution)

    def record_replay_error(self, resolution: VocoderRuntimeResolution) -> None:
        self._sink.record("replay_error", self._target_id, resolution)


class VocoderGraphStatsSink:
    """Manager-read, Recorder-write bounded runtime counters."""

    def __init__(self, *, enabled: bool, max_runtime_keys: int) -> None:
        self.enabled = enabled
        self.max_runtime_keys = max_runtime_keys
        self._lock = threading.Lock()
        self._calls: Counter[str] = Counter()
        self._outcomes: Counter[tuple[str, str]] = Counter()
        self._descriptors: Counter[tuple[str, object]] = Counter()
        self._runtime_keys: OrderedDict[tuple[str, object], int] = OrderedDict()

    def recorder_for(self, target_id: str) -> _TargetRecorder | _NoOpRecorder:
        if not self.enabled:
            return _NoOpRecorder()
        return _TargetRecorder(self, target_id)

    def record(
        self,
        outcome: str,
        target_id: str,
        resolution: VocoderRuntimeResolution,
    ) -> None:
        with self._lock:
            self._calls[target_id] += 1
            self._outcomes[(target_id, outcome)] += 1
            if resolution.descriptor is not None:
                self._descriptors[(target_id, resolution.descriptor.variant)] += 1
            key = (target_id, resolution.runtime_key.variant)
            self._runtime_keys[key] = self._runtime_keys.get(key, 0) + 1
            self._runtime_keys.move_to_end(key)
            while len(self._runtime_keys) > self.max_runtime_keys:
                self._runtime_keys.popitem(last=False)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "calls": dict(self._calls),
                "outcomes": dict(self._outcomes),
                "descriptors": dict(self._descriptors),
                "runtime_keys": dict(self._runtime_keys),
            }


class VocoderCUDAGraphManager:
    """Consumes resolved Targets and owns capture/bind/restore lifecycle."""

    def __init__(self, *, vllm_config: VllmConfig, device: torch.device) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.targets: tuple[VocoderCUDAGraphTarget, ...] = ()
        self.managed_targets: dict[str, ManagedTarget] = {}
        self._target_configs: dict[str, VocoderTargetRuntimeConfig] = {}
        self._capture_lock = threading.RLock()
        self._runtime_capture_stream: torch.cuda.Stream | None = None
        self._prepared = False
        self._capture_finished = False
        self._captured_memory_bytes = 0

        raw_config = getattr(vllm_config.model_config, "vocoder_cudagraph_config", None)
        if raw_config is None:
            raw_config = {}
        if not isinstance(raw_config, Mapping):
            raise TypeError("vocoder_cudagraph must be a mapping")
        self.config = dict(raw_config)
        self.max_memory_bytes = self._optional_nonnegative_int(
            self.config.get("max_memory_bytes"),
            "vocoder_cudagraph.max_memory_bytes",
        )
        log_stats = self._bool_value(self.config.get("log_stats", False), "vocoder_cudagraph.log_stats")
        stats_max_runtime_keys = self._nonnegative_int(
            self.config.get("stats_max_runtime_keys", 128),
            "vocoder_cudagraph.stats_max_runtime_keys",
        )
        self.stats_sink = VocoderGraphStatsSink(
            enabled=log_stats,
            max_runtime_keys=stats_max_runtime_keys,
        )

    @staticmethod
    def _bool_value(value: object, path: str) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be a bool")
        return value

    @staticmethod
    def _nonnegative_int(value: object, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{path} must be a non-negative integer")
        return value

    @classmethod
    def _optional_nonnegative_int(cls, value: object, path: str) -> int | None:
        return None if value is None else cls._nonnegative_int(value, path)

    @staticmethod
    def _model_shared_config_keys(model: object) -> frozenset[str]:
        declaration: object = getattr(model, "vocoder_cudagraph_shared_config_keys", frozenset())
        if callable(declaration):
            declaration = declaration()
        return frozenset(cast(Sequence[str], declaration))

    def _memory_allocated(self) -> int:
        if self.device.type == "cpu":
            return 0
        return int(torch.accelerator.memory_allocated(self.device))

    def prepare(self, model: SupportsVocoderCUDAGraph) -> None:
        if self._prepared:
            raise RuntimeError("VocoderCUDAGraphManager.prepare() called more than once")

        targets = tuple(model.get_vocoder_cudagraph_targets())
        target_by_id: dict[str, VocoderCUDAGraphTarget] = {}
        for target in targets:
            if not isinstance(target, VocoderCUDAGraphTarget):
                raise TypeError("get_vocoder_cudagraph_targets() must return VocoderCUDAGraphTarget objects")
            if not target.target_id:
                raise ValueError("Vocoder CUDA Graph target_id must not be empty")
            if target.target_id in target_by_id:
                raise ValueError(f"Duplicate vocoder CUDA Graph target_id: {target.target_id}")
            if target.routine.target_id != target.target_id:
                raise ValueError(f"Target/Routine target_id mismatch for {target.target_id}")
            try:
                if len(set(target.descriptors)) != len(target.descriptors):
                    raise ValueError(f"Duplicate Descriptor in Target {target.target_id}")
            except TypeError as exc:
                raise TypeError(f"Descriptors for Target {target.target_id} must be hashable") from exc
            target_by_id[target.target_id] = target
            if not target.descriptors:
                logger.info(
                    "Vocoder CUDA Graph Target %s is known but has no startup "
                    "Descriptors for the resolved model configuration",
                    target.target_id,
                )

        unknown_shared = set(self.config) - _FRAMEWORK_CONFIG_KEYS - self._model_shared_config_keys(model)
        if unknown_shared:
            names = ", ".join(sorted(unknown_shared))
            raise ValueError(f"Unknown vocoder_cudagraph config key(s): {names}")

        raw_target_configs = self.config.get("targets", {})
        if not isinstance(raw_target_configs, Mapping):
            raise TypeError("vocoder_cudagraph.targets must be a mapping")
        unknown_target_ids = set(raw_target_configs) - set(target_by_id)
        if unknown_target_ids:
            names = ", ".join(sorted(str(name) for name in unknown_target_ids))
            raise ValueError(f"Unknown vocoder CUDA Graph target override(s): {names}")

        for target_id, target in target_by_id.items():
            raw_target = raw_target_configs.get(target_id, {})
            if not isinstance(raw_target, Mapping):
                raise TypeError(f"vocoder_cudagraph.targets.{target_id} must be a mapping")
            unknown_keys = set(raw_target) - _TARGET_POLICY_KEYS - set(target.supported_config_keys)
            if unknown_keys:
                names = ", ".join(sorted(unknown_keys))
                raise ValueError(f"Unknown config key(s) for vocoder Target {target_id}: {names}")
            enabled = self._bool_value(
                raw_target.get("enabled", True),
                f"vocoder_cudagraph.targets.{target_id}.enabled",
            )
            lazy = self._bool_value(
                raw_target.get("enable_lazy_capture", False),
                f"vocoder_cudagraph.targets.{target_id}.enable_lazy_capture",
            )
            max_extra = self._nonnegative_int(
                raw_target.get("max_extra_graphs", 0),
                f"vocoder_cudagraph.targets.{target_id}.max_extra_graphs",
            )
            self._target_configs[target_id] = VocoderTargetRuntimeConfig(
                enabled=enabled,
                enable_lazy_capture=lazy,
                max_extra_graphs=max_extra,
            )

        self.targets = targets
        self._prepared = True
        logger.info(
            "Prepared runner-owned vocoder CUDA Graph Targets: %s",
            [target.target_id for target in targets],
        )

    def capture_entry(
        self,
        target: VocoderCUDAGraphTarget,
        descriptor: VocoderCUDAGraphDescriptor,
    ) -> VocoderCUDAGraphEntry | None:
        routine = target.routine
        buffers: object | None = None
        try:
            buffers = routine.allocate_buffers(descriptor, self.device)
            routine.prepare_for_capture(buffers)
            num_warmups = max(
                1,
                int(getattr(self.vllm_config.compilation_config, "cudagraph_num_of_warmups", 0)),
            )
            for _ in range(num_warmups):
                # Stateful routines may need to rebuild their temporary cache
                # object between warmups. This hook is intentionally repeated
                # before the final graph capture as well.
                routine.prepare_for_capture(buffers)
                routine.forward_for_capture(buffers)
                routine.reset_after_capture(buffers)
            routine.prepare_for_capture(buffers)
            torch.cuda.current_stream(self.device).synchronize()
            graph = torch.cuda.CUDAGraph()
            with (
                torch.inference_mode(),
                torch.cuda.graph(
                    graph,
                    pool=current_platform.get_global_graph_pool(),
                ),
            ):
                captured_output = routine.forward_for_capture(buffers)
            return VocoderCUDAGraphEntry(
                descriptor=descriptor,
                graph=graph,
                buffers=buffers,
                captured_output=captured_output,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.warning(
                "Failed to capture vocoder CUDA Graph Target %s Descriptor %r; this Descriptor will remain eager",
                target.target_id,
                descriptor,
                exc_info=True,
            )
            return None
        finally:
            if buffers is not None:
                try:
                    routine.reset_after_capture(buffers)
                except Exception:
                    logger.exception(
                        "Failed to reset capture state for vocoder Target %s",
                        target.target_id,
                    )
                    raise

    def _capture_and_register(
        self,
        managed: ManagedTarget,
        descriptor: VocoderCUDAGraphDescriptor,
    ) -> VocoderCUDAGraphEntry | None:
        existing = managed.entries.get(descriptor)
        if existing is not None:
            managed.entries.move_to_end(descriptor)
            return existing
        if descriptor in managed.failed_descriptors:
            return None

        allocated_before = self._memory_allocated()
        entry = self.capture_entry(managed.target, descriptor)
        if entry is None:
            managed.failed_descriptors.add(descriptor)
            return None
        allocated_after = self._memory_allocated()
        captured_bytes = max(0, allocated_after - allocated_before)
        if self.max_memory_bytes is not None and self._captured_memory_bytes + captured_bytes > self.max_memory_bytes:
            logger.warning(
                "Vocoder CUDA Graph memory budget rejected Target %s Descriptor %r",
                managed.target.target_id,
                descriptor,
            )
            return None

        self._captured_memory_bytes += captured_bytes
        managed.entries[descriptor] = entry
        managed.entries.move_to_end(descriptor)
        if managed.max_graphs is not None:
            self._evict_lru_if_needed(managed)
        return entry

    def _evict_lru_if_needed(self, managed: ManagedTarget) -> None:
        assert managed.max_graphs is not None
        while len(managed.entries) > managed.max_graphs:
            managed.entries.popitem(last=False)

    def _touch_entry(self, managed: ManagedTarget, descriptor: VocoderCUDAGraphDescriptor) -> None:
        with self._capture_lock:
            if descriptor in managed.entries:
                managed.entries.move_to_end(descriptor)

    @contextmanager
    def _runtime_capture_scope(self):
        if self._runtime_capture_stream is None:
            self._runtime_capture_stream = torch.cuda.Stream(device=self.device)
        caller = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(caller)
        self._runtime_capture_stream.wait_event(ready)
        unlock_workspace()
        set_cudagraph_capturing_enabled(True)
        try:
            with torch.cuda.stream(self._runtime_capture_stream):
                yield
                complete = torch.cuda.Event()
                complete.record(self._runtime_capture_stream)
            caller.wait_event(complete)
        finally:
            set_cudagraph_capturing_enabled(False)
            lock_workspace()

    def _runtime_capture_and_register(
        self,
        managed: ManagedTarget,
        descriptor: VocoderCUDAGraphDescriptor,
    ) -> VocoderCUDAGraphEntry | None:
        if not managed.enable_lazy_capture:
            return None
        if descriptor in managed.failed_descriptors:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        with self._capture_lock:
            existing = managed.entries.get(descriptor)
            if existing is not None:
                managed.entries.move_to_end(descriptor)
                return existing
            if descriptor in managed.failed_descriptors:
                return None
            if managed.max_graphs == 0:
                return None
            with self._runtime_capture_scope():
                return self._capture_and_register(managed, descriptor)

    def _make_runtime_miss_handler(
        self,
        managed: ManagedTarget,
    ) -> Callable[[VocoderRuntimeResolution], VocoderCUDAGraphEntry | None]:
        if not managed.enable_lazy_capture:
            return lambda resolution: None

        def on_runtime_miss(resolution: VocoderRuntimeResolution) -> VocoderCUDAGraphEntry | None:
            descriptor = resolution.descriptor
            if descriptor is None:
                descriptor = managed.target.routine.make_lazy_descriptor(resolution.runtime_key)
            if descriptor is None:
                return None
            return self._runtime_capture_and_register(managed, descriptor)

        return on_runtime_miss

    def _build_runtime_callable(
        self,
        managed: ManagedTarget,
        recorder: _TargetRecorder | _NoOpRecorder,
    ) -> Callable[..., Any]:
        target = managed.target
        entries = MappingProxyType(managed.entries)
        routine = target.routine
        on_runtime_miss = self._make_runtime_miss_handler(managed)
        clone_output = target.clone_output

        def runtime_callable(*args: Any, **kwargs: Any) -> Any:
            routine.validate_runtime_inputs(args, kwargs)
            resolution = routine.resolve_runtime(args, kwargs, entries.keys())
            entry = entries.get(resolution.descriptor) if resolution.descriptor is not None else None
            if entry is None:
                entry = on_runtime_miss(resolution)
            if entry is None:
                recorder.record_fallback(resolution)
                return routine.eager_call(*args, **kwargs)

            descriptor = entry.descriptor
            graph_resolution = (
                resolution
                if resolution.descriptor == descriptor
                else VocoderRuntimeResolution(
                    runtime_key=resolution.runtime_key,
                    descriptor=descriptor,
                )
            )
            with entry.replay_lock:
                try:
                    routine.copy_runtime_inputs(args, kwargs, entry.buffers)
                    entry.graph.replay()
                    output = routine.output_after_replay(args, kwargs, entry.buffers, entry.captured_output)
                    if clone_output:
                        output = clone_tensor_tree(output)
                except Exception:
                    recorder.record_replay_error(graph_resolution)
                    raise
                if managed.enable_lazy_capture:
                    self._touch_entry(managed, descriptor)
                recorder.record_graph_hit(graph_resolution)
            return output

        return runtime_callable

    def capture_and_bind(self) -> int:
        if not self._prepared:
            raise RuntimeError("VocoderCUDAGraphManager must be prepared before capture")
        if self._capture_finished:
            raise RuntimeError("Vocoder CUDA Graph capture has already completed")
        self._capture_finished = True

        capture_start = time.perf_counter()
        memory_before = self._memory_allocated()
        prepared: dict[str, ManagedTarget] = {}

        # Phase 1: every Target remains eager until all startup captures finish.
        selected = [target for target in self.targets if self._target_configs[target.target_id].enabled]
        for target in selected:
            target_config = self._target_configs[target.target_id]
            if target._bound_handle is not None:
                raise RuntimeError(f"Target already bound before capture: {target.target_id}")
            managed = ManagedTarget(
                target=target,
                entries=OrderedDict(),
                enable_lazy_capture=target_config.enable_lazy_capture,
            )
            progress = (
                tqdm(
                    total=len(target.descriptors),
                    desc=f"Capture {target.target_id}",
                    unit="graph",
                    leave=True,
                )
                if target.descriptors
                else None
            )
            try:
                for descriptor in target.descriptors:
                    try:
                        self._capture_and_register(managed, descriptor)
                    finally:
                        assert progress is not None
                        progress.update(1)
            finally:
                if progress is not None:
                    progress.close()
            logger.info(
                "Vocoder CUDA Graph Target %s captured %d/%d startup Descriptors",
                target.target_id,
                len(managed.entries),
                len(target.descriptors),
            )
            managed.max_graphs = len(managed.entries) + target_config.max_extra_graphs
            if not managed.entries and not managed.enable_lazy_capture:
                continue
            prepared[target.target_id] = managed

        # Phase 2: assembly/binding failure is isolated to one Target.
        active: dict[str, ManagedTarget] = {}
        for target_id, managed in prepared.items():
            target = managed.target
            try:
                runtime_callable = self._build_runtime_callable(
                    managed,
                    self.stats_sink.recorder_for(target_id),
                )
                target._bind_handle(VocoderGraphHandle(runtime_callable))
            except Exception:
                target._restore_eager()
                managed.entries.clear()
                logger.exception(
                    "Failed to assemble/bind vocoder CUDA Graph Target %s; leaving it eager",
                    target_id,
                )
                continue
            active[target_id] = managed

        self.managed_targets = active
        memory_after = self._memory_allocated()
        captured_memory = max(0, memory_after - memory_before)
        logger.info(
            "Vocoder CUDA Graph capture finished in %.2fs, bound=%s, memory=%.2f MiB",
            time.perf_counter() - capture_start,
            list(active),
            captured_memory / (1 << 20),
        )
        return captured_memory

    def clear(self) -> None:
        for managed in self.managed_targets.values():
            managed.target._restore_eager()
            managed.entries.clear()
            managed.failed_descriptors.clear()
        self.managed_targets.clear()
        self._runtime_capture_stream = None
        if self.stats_sink.enabled:
            logger.info("Vocoder CUDA Graph runtime stats: %s", self.stats_sink.snapshot())
