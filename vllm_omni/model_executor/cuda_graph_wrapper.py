from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from threading import Lock
from typing import Any, Generic, TypeVar

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

KeyT = TypeVar("KeyT")
CaptureMode = str

_CAPTURE_MODE_PRECAPTURE = "pre-capture"
_CAPTURE_MODE_LAZY = "lazy"
_CAPTURE_MODE_HYBRID = "hybrid"
_VALID_CAPTURE_MODES = {
    _CAPTURE_MODE_PRECAPTURE,
    _CAPTURE_MODE_LAZY,
    _CAPTURE_MODE_HYBRID,
}


class BaseCUDAGraphWrapper(ABC, Generic[KeyT]):
    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        enabled: bool = True,
        device: torch.device | str = "cuda",
        capture_mode: CaptureMode = _CAPTURE_MODE_PRECAPTURE,
        num_warmup: int = 1,
        graph_pool=None,
    ):
        self.fn = fn
        self.enabled = enabled
        self.graphs: dict[KeyT, CUDAGraph] = {}
        self._warmed_up = False
        self.capture_mode = self._normalize_capture_mode(capture_mode)
        self.device = torch.device(device)
        self.dtype: torch.dtype | None = None
        self.num_warmup = int(num_warmup)
        self.graph_pool = graph_pool or current_platform.get_global_graph_pool()
        self.static_inputs: dict[KeyT, Any] = {}
        self.static_outputs: dict[KeyT, Any] = {}
        # Used only by lazy/hybrid replay misses to avoid duplicate first-hit capture.
        self._lazy_capture_lock = Lock()

    @staticmethod
    def _normalize_capture_mode(capture_mode: CaptureMode) -> CaptureMode:
        mode = capture_mode.lower()
        if mode not in _VALID_CAPTURE_MODES:
            raise ValueError(
                f"Unsupported CUDA graph capture mode {capture_mode!r}. Expected one of {sorted(_VALID_CAPTURE_MODES)}."
            )
        return mode

    def prepare_capture_context(self, **context_kwargs) -> None:
        pass

    @abstractmethod
    def get_capture_keys(self) -> list[KeyT]:
        return []

    @abstractmethod
    def get_static_call_args(self, key: KeyT, *args, **kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Get or initialize the static arguments for warmup and capture.

        This hook owns adapter-specific static input/state initialization for
        ``key``. Returned objects must stay address-stable for the same key
        after capture because CUDA Graph replay reads and writes the captured
        memory addresses. Runtime ``args``/``kwargs`` are only needed by lazy
        capture paths whose static buffers depend on the first request. The
        pre-capture path calls this hook with ``key`` only.

        Returns:
            A pair of ``(args, kwargs)`` used to call ``self.fn(*args, **kwargs)``.
        """
        ...

    def _warmup_for_key(self, key: KeyT, *runtime_args, **runtime_kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.fn is None:
            raise NotImplementedError("_warmup_for_key must be overridden when fn is not provided")
        # ``runtime_args``/``runtime_kwargs`` are intentionally forwarded only
        # for lazy capture, where the first request may be needed to initialize
        # static buffers. Pre-capture calls this method with ``key`` only.
        static_args, static_kwargs = self.get_static_call_args(key, *runtime_args, **runtime_kwargs)
        for _ in range(self.num_warmup):
            with torch.no_grad():
                self.fn(*static_args, **static_kwargs)

        return static_args, static_kwargs

    def _capture_for_key(self, key: KeyT, static_args: tuple[Any, ...], static_kwargs: dict[str, Any]) -> None:
        if self.fn is None:
            raise NotImplementedError("_capture_for_key must be overridden when fn is not provided")

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=self.graph_pool):
                output = self.fn(*static_args, **static_kwargs)

        self.graphs[key] = graph
        self.save_capture_output(key, output)

    def _get_cuda_memory_stats(self) -> tuple[int, int, int] | None:
        if self.device.type != "cuda":
            return None
        try:
            return (
                int(torch.cuda.memory_allocated(self.device)),
                int(torch.cuda.memory_reserved(self.device)),
                int(torch.cuda.max_memory_reserved(self.device)),
            )
        except Exception:
            return None

    @staticmethod
    def _capture_start_log(key: KeyT) -> None:
        pass

    @staticmethod
    def _capture_success_log(key: KeyT) -> None:
        pass

    @staticmethod
    def _capture_complete_log(keys, captured, failed, memory_delta) -> None:
        before, after = memory_delta
        if before is None or after is None:
            return

        def gib(value: int) -> float:
            return value / 1024**3

        alloc_before, reserved_before, max_reserved_before = before
        alloc_after, reserved_after, max_reserved_after = after
        logger.info(
            f"Captured {captured} out of {len(keys)} keys, ({failed} failed). "
            f" (cuda_mem allocated {gib(alloc_before):.2f}->{gib(alloc_after):.2f} GiB, "
            f"reserved {gib(reserved_before):.2f}->{gib(reserved_after):.2f} GiB, "
            f"max_reserved {gib(max_reserved_before):.2f}->{gib(max_reserved_after):.2f} GiB)"
        )

    def _replay_hit_log(self, key, *args, **kwargs) -> None:
        pass

    def _replay_fallback_log(self, reason, key, *args, **kwargs) -> None:
        logger.debug(
            "Failed to replay CUDA graph for key %s, reason: %s",
            key,
            reason,
        )

    def run_eager(self, *args, **kwargs):
        if self.fn is None:
            raise NotImplementedError("run_eager must be overridden when fn is not provided")
        return self.fn(*args, **kwargs)

    def can_replay(self, *args, **kwargs) -> bool:
        return (
            self.enabled
            and self.device.type == "cuda"
            and (
                self._warmed_up or self.capture_mode == _CAPTURE_MODE_LAZY or self.capture_mode == _CAPTURE_MODE_HYBRID
            )
        )

    @abstractmethod
    def prepare_runtime_input(self, key, *args, **kwargs) -> None:
        """Copy/pad the runtime request into the already captured static inputs."""
        raise NotImplementedError

    @abstractmethod
    def select_runtime_key(self, *args, **kwargs) -> KeyT | None: ...

    def save_capture_output(self, key: KeyT, output: torch.Tensor | Any) -> None:
        self.static_outputs[key] = output

    @staticmethod
    def trim_nominal(output, actual_size, scale):
        return output[..., : actual_size * scale]

    @staticmethod
    def trim_captured_minus_padding(output, actual_size, bucket_size, scale):
        drop = (bucket_size - actual_size) * scale
        return output[..., : max(0, output.shape[-1] - drop)]

    def postprocess(self, key, *args, **kwargs):
        """
        Built-in trim helpers cover:
        - nominal: trim_nominal(...)
        - captured_minus_padding: trim_captured_minus_padding(...)

        Per-row batch and custom outputs should override this method.
        """
        return self.static_outputs[key]

    def _lazy_capture_key(self, key, *args, **kwargs):
        with self._lazy_capture_lock:
            if key in self.graphs:
                return
            self._capture_start_log(key)
            try:
                static_args, static_kwargs = self._warmup_for_key(key, *args, **kwargs)
                torch.accelerator.synchronize(self.device)
                self._capture_for_key(key, static_args, static_kwargs)
                self._capture_success_log(key)
            except Exception:
                logger.warning("Failed to capture CUDA graph for key %s", key, exc_info=True)
                raise

    def clone_output(self, output):
        if torch.is_tensor(output):
            return output.clone()
        if isinstance(output, tuple) and hasattr(output, "_fields"):
            return type(output)(*(self.clone_output(x) for x in output))
        if isinstance(output, tuple):
            return tuple(self.clone_output(x) for x in output)
        if isinstance(output, list):
            return [self.clone_output(x) for x in output]
        if isinstance(output, dict):
            return {k: self.clone_output(v) for k, v in output.items()}
        return output

    def before_capture(self, keys: Iterable[KeyT]) -> None:
        pass

    def capture(
        self,
        keys: Iterable[KeyT] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **context_kwargs,
    ) -> None:
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype

        if self.capture_mode == _CAPTURE_MODE_LAZY:
            return
        if not self.enabled or self._warmed_up:
            return
        if self.device.type != "cuda":
            self.enabled = False
            return

        self.prepare_capture_context(**context_kwargs)

        keys = list(self.get_capture_keys() if keys is None else keys)
        memory_before = self._get_cuda_memory_stats()
        captured = 0
        failed = 0

        self.before_capture(keys)

        for key in keys:
            try:
                self._capture_start_log(key)
                static_args, static_kwargs = self._warmup_for_key(key)
                torch.accelerator.synchronize(self.device)
                self._capture_for_key(key, static_args, static_kwargs)
                self._capture_success_log(key)
                captured += 1
            except Exception:
                failed += 1
                logger.warning("Failed to capture CUDA graph for key %s", key, exc_info=True)

        memory_after = self._get_cuda_memory_stats()
        memory_delta = (memory_before, memory_after)
        self._capture_complete_log(keys, captured, failed, memory_delta)

        self._warmed_up = True

    def replay(self, *args, clone_graph_output: bool = True, **kwargs):
        if not self.can_replay(*args, **kwargs):
            self._replay_fallback_log("not_replayable", None, *args, **kwargs)
            return self.run_eager(*args, **kwargs)

        if torch.cuda.is_current_stream_capturing():
            self._replay_fallback_log("outer_stream_capture", None, *args, **kwargs)
            return self.run_eager(*args, **kwargs)

        key = self.select_runtime_key(*args, **kwargs)

        if key is None:
            self._replay_fallback_log("no_key", None, *args, **kwargs)
            return self.run_eager(*args, **kwargs)

        if key not in self.graphs:
            if not (self.capture_mode == _CAPTURE_MODE_LAZY or self.capture_mode == _CAPTURE_MODE_HYBRID):
                self._replay_fallback_log("missing_graph", key, *args, **kwargs)
                return self.run_eager(*args, **kwargs)

            try:
                self._lazy_capture_key(key, *args, **kwargs)
            except Exception:
                self._replay_fallback_log("lazy_capture_failed", key, *args, **kwargs)
                return self.run_eager(*args, **kwargs)

        if key not in self.graphs:
            self._replay_fallback_log("missing_graph_after_lazy_capture", key, *args, **kwargs)
            return self.run_eager(*args, **kwargs)

        self.prepare_runtime_input(key, *args, **kwargs)
        self.graphs[key].replay()
        self._replay_hit_log(key, *args, **kwargs)
        output = self.postprocess(key, *args, **kwargs)
        return self.clone_output(output) if clone_graph_output else output

    def __call__(self, *args, **kwargs):
        return self.replay(*args, **kwargs)
