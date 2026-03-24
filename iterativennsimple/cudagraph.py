from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch


class CUDAGraphFunctionCache:
    """Lazily capture and replay CUDA graphs for a tensor-only callable."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        enabled: bool = True,
        num_warmup_iters: int = 3,
        name: str | None = None,
    ) -> None:
        self.fn = fn
        self.enabled = enabled
        self.num_warmup_iters = num_warmup_iters
        self.name = name or getattr(fn, "__name__", fn.__class__.__name__)
        self._cache: dict[tuple[Any, ...], Callable[..., Any]] = {}
        self._disabled_reason: str | None = None

    @property
    def graph_count(self) -> int:
        return len(self._cache)

    @property
    def is_active(self) -> bool:
        return self.enabled and self._disabled_reason is None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _can_use_cuda_graphs(self, args: tuple[Any, ...]) -> bool:
        if not self.enabled:
            return False
        if self._disabled_reason is not None:
            return False
        if not torch.cuda.is_available():
            return False
        if not args or not all(torch.is_tensor(arg) for arg in args):
            return False
        return all(arg.is_cuda for arg in args)

    @staticmethod
    def _signature(args: tuple[torch.Tensor, ...]) -> tuple[Any, ...]:
        return tuple(
            (
                tuple(arg.shape),
                arg.dtype,
                arg.device.type,
                arg.device.index,
                arg.requires_grad,
            )
            for arg in args
        )

    @staticmethod
    def _clone_sample_arg(arg: torch.Tensor) -> torch.Tensor:
        sample = arg.detach().clone()
        if arg.requires_grad:
            sample.requires_grad_(True)
        return sample

    def _capture(self, sample_args: tuple[torch.Tensor, ...], num_warmup_iters: int) -> Callable[..., Any]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return torch.cuda.make_graphed_callables(
            self.fn,
            sample_args,
            num_warmup_iters=num_warmup_iters,
        )

    @staticmethod
    def _cleanup_cuda_failure() -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __call__(self, *args: torch.Tensor) -> Any:
        if not self._can_use_cuda_graphs(args):
            return self.fn(*args)

        key = self._signature(args)
        graphed = self._cache.get(key)
        if graphed is None:
            sample_args = tuple(self._clone_sample_arg(arg) for arg in args)
            try:
                graphed = self._capture(sample_args, self.num_warmup_iters)
            except Exception as exc:  # pragma: no cover - depends on CUDA runtime support
                should_retry = (
                    self.num_warmup_iters > 0
                    and isinstance(exc, RuntimeError)
                    and "out of memory" in str(exc).lower()
                )
                if should_retry:
                    try:
                        graphed = self._capture(sample_args, 0)
                    except Exception as retry_exc:  # pragma: no cover - depends on CUDA runtime support
                        self._cleanup_cuda_failure()
                        self._disabled_reason = f"{type(retry_exc).__name__}: {retry_exc}"
                        warnings.warn(
                            f"Disabling CUDA graphs for {self.name}: {self._disabled_reason}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        return self.fn(*args)
                else:
                    self._cleanup_cuda_failure()
                    self._disabled_reason = f"{type(exc).__name__}: {exc}"
                    warnings.warn(
                        f"Disabling CUDA graphs for {self.name}: {self._disabled_reason}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return self.fn(*args)
            self._cache[key] = graphed

        return graphed(*args)
