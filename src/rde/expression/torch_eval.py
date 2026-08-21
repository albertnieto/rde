"""Torch-accelerated expression evaluation (explicit opt-in: torch / torch_mps only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.expression.ast import Expr


def _torch():
    import torch

    return torch


def numpy_env_to_torch(env: dict[str, np.ndarray], *, device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    dev = torch.device(device)
    return {k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in env.items()}


class _TorchExprBackend:
    def __init__(self, torch, device) -> None:
        self._torch = torch
        self._device = device

    def env_len(self, env: dict[str, Any]) -> int:
        return len(next(iter(env.values()))) if env else 1

    def const(self, value: float, n: int) -> Any:
        return self._torch.full((n,), float(value), dtype=self._torch.float32, device=self._device)

    def var(self, key: str, env: dict[str, Any]) -> Any:
        if key not in env:
            raise KeyError(f"Unknown variable: {key!r}")
        return env[key]

    def neg(self, x: Any) -> Any:
        return -x

    def abs(self, x: Any) -> Any:
        return self._torch.abs(x)

    def sqrt(self, x: Any) -> Any:
        return self._torch.sqrt(self._torch.clamp(x, min=0.0))

    def log(self, x: Any) -> Any:
        return self._torch.log(self._torch.clamp(x, min=1e-30))

    def exp(self, x: Any) -> Any:
        return self._torch.exp(self._torch.clamp(x, min=-700, max=700))

    def add(self, a: Any, b: Any) -> Any:
        return a + b

    def sub(self, a: Any, b: Any) -> Any:
        return a - b

    def mul(self, a: Any, b: Any) -> Any:
        return a * b

    def div(self, num: Any, denom: Any) -> Any:
        safe = self._torch.where(
            self._torch.abs(denom) < 1e-12,
            self._torch.full_like(denom, float("nan")),
            denom,
        )
        return num / safe

    def pow(self, base: Any, expn: Any) -> Any:
        return self._torch.pow(base, expn)


def evaluate_expr_torch(expr: Expr, env: dict[str, Any]) -> Any:
    """Evaluate expression tree with torch tensors in env."""
    from rde.expression.dispatch import evaluate_expr_with_backend

    torch = _torch()
    if env:
        device = next(iter(env.values())).device
    else:
        device = torch.device("cpu")
    return evaluate_expr_with_backend(expr, env, _TorchExprBackend(torch, device))


def torch_device_for_backend(backend: str, *, strict: bool = False) -> str:
    key = backend.lower()
    # Generic GPU aliases (mps/gpu/metal) are resolved to MLX upstream — never torch.
    if key == "torch_mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if strict:
                from rde.backends.resolve import GpuBackendError

                raise GpuBackendError("torch_mps requested but PyTorch MPS is unavailable")
        except ImportError as exc:
            if strict:
                from rde.backends.resolve import GpuBackendError

                raise GpuBackendError("torch_mps requested but torch is not installed") from exc
    if strict and key == "torch_mps":
        from rde.backends.resolve import GpuBackendError

        raise GpuBackendError(f"torch_mps requested but PyTorch MPS/CUDA is unavailable")
    return "cpu"


def evaluate_to_numpy(expr: Expr, env: dict[str, np.ndarray], *, backend: str = "numpy") -> np.ndarray:
    if backend == "numpy":
        from rde.expression.evaluate import evaluate_expr

        return evaluate_expr(expr, env)
    from rde.backends.torch_mps_backend import _as_numpy
    from rde.expression.torch_eval import evaluate_expr_torch, numpy_env_to_torch, torch_device_for_backend

    device = torch_device_for_backend(backend)
    tenv = numpy_env_to_torch(env, device=device)
    out = evaluate_expr_torch(expr, tenv)
    return _as_numpy(out)
