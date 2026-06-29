import bisect
import enum
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Generic, TypeVar

import torch
from torch.cuda import CUDAGraph
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

KeyT = TypeVar("KeyT")


class CaptureMode(enum.Enum):
    PRE_CAPTURE = "pre-capture"
    LAZY = "lazy"
    HYBRID = "hybrid"


@dataclass
class CUDAGraphOptions:
    clone_output: bool = True
    enable_log: bool = True


class BaseCUDAGraphWrapper(ABC, Generic[KeyT]):
    def __init__(
        self,
        runnable: Callable[..., Any] | None = None,
        device: torch.device | str = "cuda",
        capture_mode: CaptureMode | str = CaptureMode.PRE_CAPTURE,
        num_warmup: int = 1,
        graph_pool=None,
        enabled: bool | None = None,
        vllm_config: VllmConfig | None = None,
        cudagraph_options: CUDAGraphOptions | None = None,
    ):
        self.runnable = runnable
        self.enabled = True
        if vllm_config is not None:
            self.enabled = self.enabled and not vllm_config.model_config.enforce_eager

        if enabled is not None:
            self.enabled = self.enabled and bool(enabled)

        self.cudagraph_options = cudagraph_options or CUDAGraphOptions()
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
    def _normalize_capture_mode(capture_mode: CaptureMode | str) -> CaptureMode:
        if isinstance(capture_mode, CaptureMode):
            return capture_mode
        try:
            return CaptureMode(capture_mode.lower())
        except ValueError as exc:
            raise ValueError(
                f"Unsupported CUDA graph capture mode {capture_mode!r}. "
                f"Expected one of {[mode.value for mode in CaptureMode]}."
            ) from exc

    @staticmethod
    def find_ge_bucket(buckets: Sequence[int], actual: int) -> int | None:
        idx = bisect.bisect_left(buckets, actual)
        if idx < len(buckets):
            return buckets[idx]
        return None

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
            A pair of ``(args, kwargs)`` used to call
            ``self.runnable(*args, **kwargs)``.
        """
        ...

    def _warmup(self, key: KeyT, *runtime_args, **runtime_kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.runnable is None:
            raise NotImplementedError("_warmup must be overridden when runnable is not provided")
        # ``runtime_args``/``runtime_kwargs`` are intentionally forwarded only
        # for lazy capture, where the first request may be needed to initialize
        # static buffers. Pre-capture calls this method with ``key`` only.
        static_args, static_kwargs = self.get_static_call_args(key, *runtime_args, **runtime_kwargs)
        for _ in range(self.num_warmup):
            with torch.no_grad():
                self.runnable(*static_args, **static_kwargs)

        return static_args, static_kwargs

    def _capture(self, key: KeyT, static_args: tuple[Any, ...], static_kwargs: dict[str, Any]) -> None:
        if self.runnable is None:
            raise NotImplementedError("_capture must be overridden when runnable is not provided")

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=self.graph_pool):
                output = self.runnable(*static_args, **static_kwargs)

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

    def _log_capture_complete(
        self,
        capture_start_s: float,
        memory_before: tuple[int, int, int] | None,
        memory_after: tuple[int, int, int] | None,
        *,
        captured: int | None = None,
        total: int | None = None,
        failed: int | None = None,
    ) -> None:
        if not self.cudagraph_options.enable_log:
            return

        elapsed_ms = (time.perf_counter() - capture_start_s) * 1000.0
        memory_suffix = ""
        if memory_before is not None and memory_after is not None:
            alloc_before, reserved_before, max_reserved_before = [v / 1024**3 for v in memory_before]
            alloc_after, reserved_after, max_reserved_after = [v / 1024**3 for v in memory_after]
            memory_suffix = (
                f" (cuda_mem allocated {alloc_before:.2f}->{alloc_after:.2f} GiB, "
                f"reserved {reserved_before:.2f}->{reserved_after:.2f} GiB, "
                f"max_reserved {max_reserved_before:.2f}->{max_reserved_after:.2f} GiB)"
            )
        if captured is not None and total is not None and failed is not None:
            logger.info(
                "CUDA Graph capture complete: %d/%d captured, %d failed in %.1f ms%s",
                captured,
                total,
                failed,
                elapsed_ms,
                memory_suffix,
            )
        else:
            # logged in lazy mode
            logger.info("CUDA Graph capture complete in %.1f ms%s", elapsed_ms, memory_suffix)

    @staticmethod
    def on_capture_begin_log(key: KeyT) -> None:
        pass

    @staticmethod
    def on_capture_success_log(key: KeyT, elapsed_ms: float | None = None) -> None:
        if elapsed_ms is None:
            logger.info("CUDA Graph ready for key=%s", key)
        else:
            logger.info("CUDA Graph ready for key=%s in %.1f ms", key, elapsed_ms)

    @staticmethod
    def on_replay_hit_log(key, *args, **kwargs) -> None:
        pass

    @staticmethod
    def on_replay_fallback_log(reason, key, *args, **kwargs) -> None:
        pass

    def run_eager(self, *args, **kwargs):
        if self.runnable is None:
            raise NotImplementedError("run_eager must be overridden when runnable is not provided")
        return self.runnable(*args, **kwargs)

    def can_replay(self, *args, **kwargs) -> bool:
        return (
            self.enabled
            and self.device.type == "cuda"
            and (self._warmed_up or self.capture_mode == CaptureMode.LAZY or self.capture_mode == CaptureMode.HYBRID)
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
            capture_start_s = time.perf_counter()
            memory_before = self._get_cuda_memory_stats()
            try:
                if self.cudagraph_options.enable_log:
                    self.on_capture_begin_log(key)
                    key_start_s = time.perf_counter()

                static_args, static_kwargs = self._warmup(key, *args, **kwargs)
                torch.accelerator.synchronize(self.device)
                self._capture(key, static_args, static_kwargs)

                if self.cudagraph_options.enable_log:
                    self.on_capture_success_log(key, (time.perf_counter() - key_start_s) * 1000.0)

                self._log_capture_complete(capture_start_s, memory_before, self._get_cuda_memory_stats())
            except Exception:
                if self.cudagraph_options.enable_log:
                    logger.warning("Failed to lazy capture CUDA graph for key %s", key, exc_info=True)
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

        if self.capture_mode == CaptureMode.LAZY:
            return
        if not self.enabled or self._warmed_up:
            return
        if self.device.type != "cuda":
            self.enabled = False
            return

        self.prepare_capture_context(**context_kwargs)

        keys = list(self.get_capture_keys() if keys is None else keys)
        capture_start_s = time.perf_counter()
        memory_before = self._get_cuda_memory_stats()
        captured = 0
        failed = 0

        self.before_capture(keys)

        for key in keys:
            try:
                if self.cudagraph_options.enable_log:
                    self.on_capture_begin_log(key)
                    key_start_s = time.perf_counter()

                static_args, static_kwargs = self._warmup(key)
                torch.accelerator.synchronize(self.device)
                self._capture(key, static_args, static_kwargs)

                if self.cudagraph_options.enable_log:
                    self.on_capture_success_log(key, (time.perf_counter() - key_start_s) * 1000.0)

                captured += 1
            except Exception:
                failed += 1
                if self.cudagraph_options.enable_log:
                    logger.warning("Failed to capture CUDA graph for key %s", key, exc_info=True)

        memory_after = self._get_cuda_memory_stats()
        self._log_capture_complete(
            capture_start_s,
            memory_before,
            memory_after,
            captured=captured,
            total=len(keys),
            failed=failed,
        )

        self._warmed_up = True

    def replay(self, *args, **kwargs):
        if not self.can_replay(*args, **kwargs):
            if self.cudagraph_options.enable_log:
                self.on_replay_fallback_log("not_replayable", None, *args, **kwargs)
            return self.run_eager(*args, **kwargs)

        if torch.cuda.is_current_stream_capturing():
            if self.cudagraph_options.enable_log:
                self.on_replay_fallback_log("outer_stream_capture", None, *args, **kwargs)
            return self.run_eager(*args, **kwargs)

        key = self.select_runtime_key(*args, **kwargs)

        if key is None:
            if self.cudagraph_options.enable_log:
                self.on_replay_fallback_log("no_key", None, *args, **kwargs)
            return self.run_eager(*args, **kwargs)

        if key not in self.graphs:
            if not (self.capture_mode == CaptureMode.LAZY or self.capture_mode == CaptureMode.HYBRID):
                if self.cudagraph_options.enable_log:
                    self.on_replay_fallback_log("missing_graph", key, *args, **kwargs)
                return self.run_eager(*args, **kwargs)

            try:
                self._lazy_capture_key(key, *args, **kwargs)
            except Exception:
                if self.cudagraph_options.enable_log:
                    self.on_replay_fallback_log("lazy_capture_failed", key, *args, **kwargs)
                return self.run_eager(*args, **kwargs)

        if key not in self.graphs:
            if self.cudagraph_options.enable_log:
                self.on_replay_fallback_log("missing_graph_after_lazy_capture", key, *args, **kwargs)
            return self.run_eager(*args, **kwargs)

        self.prepare_runtime_input(key, *args, **kwargs)
        self.graphs[key].replay()

        if self.cudagraph_options.enable_log:
            self.on_replay_hit_log(key, *args, **kwargs)

        output = self.postprocess(key, *args, **kwargs)

        return self.clone_output(output) if self.cudagraph_options.clone_output else output

    def __call__(self, *args, **kwargs):
        return self.replay(*args, **kwargs)
