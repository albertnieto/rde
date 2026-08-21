"""Compute backend protocol for trajectory enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ComputeBackend(Protocol):
    """Backend for O(2^N) QUBO cost / amplitude enumeration."""

    name: str

    def all_costs(self, Q: np.ndarray) -> np.ndarray:
        ...

    def bound_normalized_weights(
        self, Q: np.ndarray, *, costs: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        ...

    def trajectory_states(self, w_amp: np.ndarray, indices: list[int]) -> dict[int, np.ndarray]:
        ...


@dataclass(frozen=True)
class FullInstanceBatch:
    """Fused batch output for B QUBOs of fixed size n.

    Shapes: costs/w_amp/w_prob are (B, 2^n); qm/a/b are (B,); states[r] is (B, 2^n).
    """

    costs: np.ndarray
    w_amp: np.ndarray
    w_prob: np.ndarray
    qm: np.ndarray
    a: np.ndarray
    b: np.ndarray
    states: dict[int, np.ndarray]


def normalize_trajectory_states(w_amp: np.ndarray, indices: list[int]) -> dict[int, np.ndarray]:
    """Batch-normalize |D_r> coefficients for several r values."""
    w = np.asarray(w_amp, dtype=float)
    idx = np.asarray(indices, dtype=int)
    if np.any(idx <= 0):
        raise ValueError("r must be >= 1")
    powers = w[None, :] ** idx[:, None]
    norms = np.linalg.norm(powers, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    normalized = powers / norms
    return {int(r): normalized[i] for i, r in enumerate(indices)}


def normalize_trajectory_states_batch(
    w_amp: np.ndarray, indices: list[int]
) -> dict[int, np.ndarray]:
    """Batch-normalize trajectory states for B instances at once.

    w_amp: (B, 2^n); returns states[r]: (B, 2^n).
    """
    if not indices:
        return {}
    w = np.asarray(w_amp, dtype=float)
    if w.ndim != 2:
        raise ValueError(f"w_amp must be 2-D (B, 2^n), got shape {w.shape}")
    idx = np.asarray(indices, dtype=float)
    powered = w[None, :, :] ** idx[:, None, None]
    norms = np.linalg.norm(powered, axis=2, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    normalized = powered / norms
    return {int(r): normalized[i] for i, r in enumerate(indices)}


def full_instance_batch_fallback(
    backend: ComputeBackend, Qs: np.ndarray, indices: list[int]
) -> FullInstanceBatch:
    """Scalar API loop for backends without a native batch implementation."""
    q_arr = np.asarray(Qs, dtype=float)
    if q_arr.ndim == 2:
        q_arr = q_arr[None, ...]
    batch_size = q_arr.shape[0]
    costs_list: list[np.ndarray] = []
    w_amp_list: list[np.ndarray] = []
    w_prob_list: list[np.ndarray] = []
    qm_list: list[float] = []
    a_list: list[float] = []
    b_list: list[float] = []
    states_by_r: dict[int, list[np.ndarray]] = {int(r): [] for r in indices}
    for i in range(batch_size):
        q = q_arr[i]
        costs = backend.all_costs(q)
        w_amp, w_prob, qm, a, b = backend.bound_normalized_weights(q, costs=costs)
        costs_list.append(costs)
        w_amp_list.append(w_amp)
        w_prob_list.append(w_prob)
        qm_list.append(qm)
        a_list.append(a)
        b_list.append(b)
        if indices:
            traj = backend.trajectory_states(w_amp, indices)
            for r in indices:
                states_by_r[int(r)].append(traj[int(r)])
    states = {r: np.stack(states_by_r[r], axis=0) for r in indices} if indices else {}
    return FullInstanceBatch(
        costs=np.stack(costs_list, axis=0),
        w_amp=np.stack(w_amp_list, axis=0),
        w_prob=np.stack(w_prob_list, axis=0),
        qm=np.asarray(qm_list, dtype=float),
        a=np.asarray(a_list, dtype=float),
        b=np.asarray(b_list, dtype=float),
        states=states,
    )
