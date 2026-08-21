"""NumPy CPU backend (default)."""

from __future__ import annotations

import math

import numpy as np

from rde.backends.base import FullInstanceBatch, normalize_trajectory_states, normalize_trajectory_states_batch


def _all_assignments_matrix(n: int) -> np.ndarray:
    nums = np.arange(2**n, dtype=np.uint64)
    bits = np.arange(n, dtype=np.uint64)
    return ((nums[:, None] >> bits[None, :]) & 1).astype(float)


def _all_costs_vectorized(
    Q: np.ndarray,
    assignments: np.ndarray | None = None,
) -> np.ndarray:
    q = np.asarray(Q, dtype=float)
    x = assignments if assignments is not None else _all_assignments_matrix(q.shape[0])
    return np.sum(x * (x @ q), axis=1)


class NumpyBackend:
    name = "numpy"

    def __init__(self) -> None:
        self._assignment_cache: dict[int, np.ndarray] = {}

    def _assignments(self, n: int) -> np.ndarray:
        cached = self._assignment_cache.get(n)
        if cached is None:
            if len(self._assignment_cache) >= 2:
                self._assignment_cache.pop(next(iter(self._assignment_cache)))
            cached = _all_assignments_matrix(n)
            cached.flags.writeable = False
            self._assignment_cache[n] = cached
        return cached

    def all_costs(self, Q: np.ndarray) -> np.ndarray:
        q = np.asarray(Q, dtype=float)
        return _all_costs_vectorized(q, self._assignments(q.shape[0]))

    def bound_normalized_weights(
        self, Q: np.ndarray, *, costs: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        n = Q.shape[0]
        qm = float(np.max(np.abs(Q)))
        if qm == 0:
            ones = np.ones(2**n)
            return ones, ones, 0.0, 0.0, math.pi / 2
        a = math.pi / (2 * n**2 * qm)
        b = math.pi / 2
        if costs is None:
            costs = self.all_costs(Q)
        theta = (a * costs + b) / 2.0
        w_amp = np.cos(theta)
        return w_amp, w_amp**2, qm, a, b

    def trajectory_states(self, w_amp: np.ndarray, indices: list[int]) -> dict[int, np.ndarray]:
        return normalize_trajectory_states(w_amp, indices)

    def full_instance_batch(self, Qs: np.ndarray, indices: list[int]) -> FullInstanceBatch:
        q_arr = np.asarray(Qs, dtype=float)
        if q_arr.ndim == 2:
            q_arr = q_arr[None, ...]
        batch_size, n, _ = q_arr.shape
        x = self._assignments(n)
        xq = np.einsum("sn,bnm->bsm", x, q_arr)
        costs = np.sum(x[None, :, :] * xq, axis=2)
        qm = np.max(np.abs(q_arr), axis=(1, 2))
        a = np.full(batch_size, math.pi / 2, dtype=float)
        b = np.full(batch_size, math.pi / 2, dtype=float)
        nonzero = qm > 0
        if np.any(nonzero):
            a[nonzero] = math.pi / (2 * n**2 * qm[nonzero])
        w_amp = np.ones_like(costs)
        if np.any(nonzero):
            theta = (a[nonzero, None] * costs[nonzero] + b[nonzero, None]) / 2.0
            w_amp[nonzero] = np.cos(theta)
        w_prob = w_amp**2
        states = normalize_trajectory_states_batch(w_amp, indices)
        return FullInstanceBatch(
            costs=costs,
            w_amp=w_amp,
            w_prob=w_prob,
            qm=qm,
            a=a,
            b=b,
            states=states,
        )
