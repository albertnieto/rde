"""Apple MLX backend for batched real-valued enumeration (Apple Silicon GPU)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rde.backends.base import FullInstanceBatch


def _mlx_available() -> bool:
    try:
        import mlx.core as mx  # noqa: F401

        return True
    except ImportError:
        return False


def _assignment_matrix(mx, n: int):
    """Build (2^n, n) assignment matrix; MLX GPU uses int64 + float32."""
    nums = mx.arange(2**n, dtype=mx.int64)
    bits = mx.arange(n, dtype=mx.int64)
    return ((nums[:, None] >> bits[None, :]) & 1).astype(mx.float32)


class MlxBackend:
    name = "mlx"

    def __init__(self) -> None:
        if not _mlx_available():
            raise ImportError(
                "MLX backend requires `pip install mlx` on Apple Silicon macOS."
            )
        import mlx.core as mx

        self._mx = mx
        self._assignment_cache: dict[int, Any] = {}

    def _assignments(self, n: int) -> Any:
        cached = self._assignment_cache.get(n)
        if cached is None:
            if len(self._assignment_cache) >= 2:
                self._assignment_cache.pop(next(iter(self._assignment_cache)))
            cached = _assignment_matrix(self._mx, n)
            self._assignment_cache[n] = cached
        return cached

    def all_costs(self, Q: np.ndarray) -> np.ndarray:
        costs, *_ = self._costs_and_weights_gpu(Q)
        mx = self._mx
        mx.eval(costs)
        return np.asarray(costs, dtype=float)

    def _costs_and_weights_gpu(self, Q: np.ndarray):
        """Return lazy MLX arrays: costs, w_amp, w_prob, qm, a, b (scalars on host)."""
        mx = self._mx
        q = np.asarray(Q, dtype=np.float32)
        n = q.shape[0]
        qm = float(np.max(np.abs(q)))
        if qm == 0:
            size = 2**n
            ones = mx.ones((size,), dtype=mx.float32)
            return ones, ones, ones, 0.0, 0.0, math.pi / 2
        x = self._assignments(n)
        q_m = mx.array(q)
        costs = mx.sum(x * (x @ q_m), axis=1)
        a = math.pi / (2 * n**2 * qm)
        b = math.pi / 2
        theta = (a * costs + b) / 2.0
        w_amp = mx.cos(theta)
        w_prob = w_amp * w_amp
        return costs, w_amp, w_prob, qm, a, b

    def bound_normalized_weights(
        self, Q: np.ndarray, *, costs: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        if costs is not None:
            mx = self._mx
            n = Q.shape[0]
            qm = float(np.max(np.abs(Q)))
            if qm == 0:
                ones = np.ones(2**n)
                return ones, ones, 0.0, 0.0, math.pi / 2
            a = math.pi / (2 * n**2 * qm)
            b = math.pi / 2
            costs_m = mx.array(np.asarray(costs, dtype=np.float32))
            theta = (a * costs_m + b) / 2.0
            w_amp = mx.cos(theta)
            mx.eval(w_amp)
            w_np = np.asarray(w_amp, dtype=float)
            return w_np, w_np**2, qm, a, b
        costs_m, w_amp, w_prob, qm, a, b = self._costs_and_weights_gpu(Q)
        mx = self._mx
        mx.eval(costs_m, w_amp, w_prob)
        return (
            np.asarray(w_amp, dtype=float),
            np.asarray(w_prob, dtype=float),
            qm,
            a,
            b,
        )

    def trajectory_states(self, w_amp: np.ndarray, indices: list[int]) -> dict[int, np.ndarray]:
        if not indices:
            return {}
        mx = self._mx
        w = mx.array(np.asarray(w_amp, dtype=np.float32))
        rs = mx.array(np.asarray(indices, dtype=np.float32))
        powered = w[None, :] ** rs[:, None]
        norms = mx.sqrt(mx.sum(powered * powered, axis=1, keepdims=True))
        norms = mx.where(norms > 0, norms, mx.ones_like(norms))
        powered = powered / norms
        mx.eval(powered)
        arr = np.asarray(powered, dtype=float)
        return {int(r): arr[i] for i, r in enumerate(indices)}

    def costs_weights_and_trajectory(
        self, Q: np.ndarray, indices: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, dict[int, np.ndarray]]:
        """Single GPU sync: costs, w_amp, w_prob, scalars, and all trajectory states."""
        costs, w_amp, w_prob, qm, a, b = self._costs_and_weights_gpu(Q)
        mx = self._mx
        states: dict[int, np.ndarray] = {}
        if indices:
            rs = mx.array(np.asarray(indices, dtype=np.float32))
            powered = w_amp[None, :] ** rs[:, None]
            norms = mx.sqrt(mx.sum(powered * powered, axis=1, keepdims=True))
            norms = mx.where(norms > 0, norms, mx.ones_like(norms))
            powered = powered / norms
            mx.eval(costs, w_amp, w_prob, powered)
            powered_np = np.asarray(powered, dtype=float)
            for i, r in enumerate(indices):
                states[int(r)] = powered_np[i]
        else:
            mx.eval(costs, w_amp, w_prob)
        return (
            np.asarray(costs, dtype=float),
            np.asarray(w_amp, dtype=float),
            np.asarray(w_prob, dtype=float),
            qm,
            a,
            b,
            states,
        )

    def full_instance_batch(self, Qs: np.ndarray, indices: list[int]) -> FullInstanceBatch:
        """Fused multi-Q GPU path with a single mx.eval sync."""
        mx = self._mx
        q_host = np.asarray(Qs, dtype=np.float32)
        if q_host.ndim == 2:
            q_host = q_host[None, ...]
        batch_size, n, _ = q_host.shape
        qm_host = np.max(np.abs(q_host), axis=(1, 2))
        a_host = np.full(batch_size, math.pi / 2, dtype=np.float32)
        b_host = np.full(batch_size, math.pi / 2, dtype=np.float32)
        nonzero = qm_host > 0
        if np.any(nonzero):
            a_host[nonzero] = (math.pi / (2 * n**2 * qm_host[nonzero])).astype(np.float32)

        q_m = mx.array(q_host)
        x = self._assignments(n)
        xq = mx.einsum("sn,bnm->bsm", mx.array(x), q_m)
        costs = mx.sum(x[None, :, :] * xq, axis=2)
        theta = (mx.array(a_host)[:, None] * costs + mx.array(b_host)[:, None]) / 2.0
        w_amp = mx.cos(theta)
        ones_row = mx.ones((1, 2**n), dtype=mx.float32)
        w_amp = mx.where(mx.array(qm_host > 0)[:, None], w_amp, ones_row)
        w_prob = w_amp * w_amp
        states: dict[int, np.ndarray] = {}
        to_eval: list[Any] = [costs, w_amp, w_prob]
        powered_m = None
        if indices:
            rs = mx.array(np.asarray(indices, dtype=np.float32))
            powered_m = w_amp[:, None, :] ** rs[None, :, None]
            norms = mx.sqrt(mx.sum(powered_m * powered_m, axis=2, keepdims=True))
            norms = mx.where(norms > 0, norms, mx.ones_like(norms))
            powered_m = powered_m / norms
            to_eval.append(powered_m)
        mx.eval(*to_eval)
        if powered_m is not None:
            powered_np = np.asarray(powered_m, dtype=float)
            for i, r in enumerate(indices):
                states[int(r)] = powered_np[:, i, :]
        return FullInstanceBatch(
            costs=np.asarray(costs, dtype=float),
            w_amp=np.asarray(w_amp, dtype=float),
            w_prob=np.asarray(w_prob, dtype=float),
            qm=qm_host.astype(float),
            a=a_host.astype(float),
            b=b_host.astype(float),
            states=states,
        )
