from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

KeyT = TypeVar("KeyT")
CaptureMode = str
StaticCallArgs = Any

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
        self.num_warmup = int(num_warmup)
        self.graph_pool = graph_pool or current_platform.get_global_graph_pool()
        self.static_inputs: dict[KeyT, Any] = {}
        self.static_outputs: dict[KeyT, Any] = {}

    @staticmethod
    def _normalize_capture_mode(capture_mode: CaptureMode) -> CaptureMode:
        mode = capture_mode.lower()
        if mode not in _VALID_CAPTURE_MODES:
            raise ValueError(
                f"Unsupported CUDA graph capture mode {capture_mode!r}. Expected one of {sorted(_VALID_CAPTURE_MODES)}."
            )
        return mode

    def before_capture(self) -> None:
        pass

    @abstractmethod
    def get_capture_keys(self) -> list[KeyT]:
        return []

    @abstractmethod
    def get_static_call_args(self, key: KeyT, *args, **kwargs) -> StaticCallArgs:
        """Get or initialize the static arguments for warmup and capture.

        The common qwen3-style path returns one cached static input tensor.
        Multi-input or kwargs based captures can return ``((arg0, arg1), kwargs)``.
        The returned objects must be stable for the same key after capture because
        CUDA Graph replay writes to the captured memory addresses.
        """
        ...

    def _normalize_static_call_args(self, static_call_args: StaticCallArgs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Normalize adapter-owned static buffers into fn(*args, **kwargs).

        The common one-input case may return a single tensor/buffer directly.
        Multi-input or kwargs based captures can return ``((arg0, arg1), kwargs)``.
        """
        if (
            isinstance(static_call_args, tuple)
            and len(static_call_args) == 2
            and isinstance(static_call_args[0], tuple)
            and isinstance(static_call_args[1], dict)
        ):
            return static_call_args
        return (static_call_args,), {}

    def _warmup_for_key(self, key: KeyT, *args, **kwargs) -> None:
        if self.fn is None:
            raise NotImplementedError("eager_warmup_one must be overridden when fn is not provided")
        args, kwargs = self._normalize_static_call_args(self.get_static_call_args(key, *args, **kwargs))
        for _ in range(self.num_warmup):
            with torch.no_grad():
                self.fn(*args, **kwargs)

    def _capture_for_key(self, key: KeyT, *args, **kwargs) -> None:
        if self.fn is None:
            raise NotImplementedError("_capture_for_key must be overridden when fn is not provided")

        args, kwargs = self._normalize_static_call_args(self.get_static_call_args(key, *args, **kwargs))
        graph = CUDAGraph()

        with torch.no_grad():
            with torch.cuda.graph(graph, pool=self.graph_pool):
                output = self.fn(*args, **kwargs)

        self.graphs[key] = graph
        self.save_capture_output(key, output)

    def on_capture_error(self, key: KeyT, exc: Exception) -> None:
        logger.warning("Failed to capture CUDA graph for key %s", key, exc_info=exc)

    @abstractmethod
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

    def prepare_runtime_input(self, key, *args, **kwargs) -> None:
        """Copy/pad the runtime request into the already captured static inputs."""
        pass

    @abstractmethod
    def select_runtime_key(self, *args, **kwargs) -> KeyT | None: ...

    def save_capture_output(self, key: KeyT, output: torch.Tensor | Any) -> None:
        self.static_outputs[key] = output

    def postprocess(self, key, *args, **kwargs) -> Any:
        """Read and adapt captured output handles after graph replay."""
        return self.static_outputs[key]

    def _lazy_capture_key(self, key, *args, **kwargs):
        # If first-hit capture can run concurrently, subclasses should guard
        # this method with a lock to avoid duplicate capture for the same key.
        self._warmup_for_key(key, *args, **kwargs)
        torch.accelerator.synchronize(self.device)
        self._capture_for_key(key, *args, **kwargs)

    def capture(self, keys: Iterable[KeyT] | None = None) -> None:
        if self.capture_mode == _CAPTURE_MODE_LAZY:
            return
        if not self.enabled or self._warmed_up:
            return
        if self.device.type != "cuda":
            self.enabled = False
            return

        self.before_capture()

        keys = list(self.get_capture_keys() if keys is None else keys)

        for key in keys:
            self._warmup_for_key(key)
            torch.accelerator.synchronize(self.device)

            try:
                self._capture_for_key(key)
            except Exception as exc:
                self.on_capture_error(key, exc)

        self._warmed_up = True

    def replay(self, *args, **kwargs):
        if not self.can_replay(*args, **kwargs):
            return self.run_eager(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            return self.run_eager(*args, **kwargs)

        key = self.select_runtime_key(*args, **kwargs)

        if key is None:
            return self.run_eager(*args, **kwargs)

        if key not in self.graphs:
            if not (self.capture_mode == _CAPTURE_MODE_LAZY or self.capture_mode == _CAPTURE_MODE_HYBRID):
                return self.run_eager(*args, **kwargs)
            try:
                self._lazy_capture_key(key, *args, **kwargs)
            except Exception as exc:
                self.on_capture_error(key, exc)
                return self.run_eager(*args, **kwargs)

        if key not in self.graphs:
            return self.run_eager(*args, **kwargs)

        self.prepare_runtime_input(key, *args, **kwargs)
        self.graphs[key].replay()
        return self.postprocess(key, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.replay(*args, **kwargs)
