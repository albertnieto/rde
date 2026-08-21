"""r-dynamics step predictor: descriptors at r → target at r+1 (ship 4, G3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.discovery.checkpoint import save_latent_checkpoint
from rde.discovery.datasets import load_state_pair_batch
from rde.discovery.impute import impute_matrix
from rde.discovery.latent import ridge_predict_columns


@dataclass
class DynamicsStepResult:
    train_r_squared: float
    n_pairs: int
    feature_columns: list[str]


@dataclass
class StateDynamicsResult:
    """Linear update operator Ψ_r → Ψ_{r+1} from exact state vectors."""

    train_r_squared: float
    n_pairs: int
    state_dim: int
    transition_rank: int
    low_rank: bool
    method: str = "ridge_state"
    checkpoint_name: str = "state_dynamics"


def _pair_rows(rows: list[dict[str, Any]], target: str, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty((0, len(feature_columns))), np.empty(0)

    columns = list(feature_columns)
    if target not in columns:
        columns.append(target)
    table = FeatureTable.from_rows(rows, columns=columns)
    instance_ids = np.asarray([str(row.get("instance_id", "")) for row in rows], dtype=str)
    family_indices = np.asarray(
        [int(row["family_index"]) if row.get("family_index") is not None else -1 for row in rows],
        dtype=int,
    )
    eligible = (instance_ids != "") & (family_indices >= 0)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        return np.empty((0, len(feature_columns))), np.empty(0)
    order = np.lexsort(
        (
            family_indices[eligible_indices],
            instance_ids[eligible_indices],
        )
    )
    sorted_ids = instance_ids[eligible_indices][order]
    sorted_r = family_indices[eligible_indices][order]
    consecutive = (
        (sorted_ids[:-1] == sorted_ids[1:])
        & (sorted_r[1:] == sorted_r[:-1] + 1)
    )
    pair_positions = np.flatnonzero(consecutive)
    if pair_positions.size == 0:
        return np.empty((0, len(feature_columns))), np.empty(0)

    target_values = table.column(target)
    source_indices = eligible_indices[order[pair_positions]]
    target_indices = eligible_indices[order[pair_positions + 1]]
    valid = np.isfinite(target_values[target_indices])
    if not np.any(valid):
        return np.empty((0, len(feature_columns))), np.empty(0)
    source_indices = source_indices[valid]
    target_indices = target_indices[valid]
    X = table.data[source_indices, : len(feature_columns)]
    y = target_values[target_indices]
    return X, y


def predict_dynamics_step(
    rows: list[dict[str, Any]],
    target: str,
    *,
    feature_columns: list[str] | None = None,
) -> DynamicsStepResult:
    """Ridge: features at slice r predict target at slice r+1 (within instance)."""
    if not rows:
        return DynamicsStepResult(float("nan"), 0, [])
    cols = feature_columns or [
        c
        for c in sorted({k for row in rows for k in row if isinstance(row.get(k), (int, float))})
        if c not in {target, "size", "seed", "family_index"}
        and not str(c).startswith("metric.")
    ][:16]
    X, y = _pair_rows(rows, target, cols)
    if X.shape[0] < X.shape[1] + 2:
        return DynamicsStepResult(float("nan"), int(X.shape[0]), cols)
    _, r2 = ridge_predict_columns(X, y)
    return DynamicsStepResult(train_r_squared=r2, n_pairs=int(X.shape[0]), feature_columns=cols)


def _effective_rank(matrix: np.ndarray, *, tol: float = 1e-6) -> int:
    if matrix.size == 0:
        return 0
    s = np.linalg.svd(matrix, compute_uv=False)
    if s.size == 0:
        return 0
    threshold = tol * float(s[0])
    return int(np.sum(s > threshold))


def predict_state_dynamics(
    run_id: str,
    store_root: str | Path,
    *,
    checkpoint: bool = True,
    low_rank_ratio: float = 0.25,
    compute_backend: str | None = None,
) -> StateDynamicsResult:
    """Ridge map on consecutive Ψ_r state vectors; report effective rank of transition."""
    batch = load_state_pair_batch(run_id, store_root)
    n_pairs = int(batch.prev_states.shape[0])
    state_dim = int(batch.prev_states.shape[1]) if n_pairs else 0
    if n_pairs < 2 or state_dim == 0:
        return StateDynamicsResult(float("nan"), n_pairs, state_dim, 0, False)

    backend = (compute_backend or "numpy").lower()
    if backend not in {"numpy", "mlx"}:
        raise ValueError(f"Unsupported state dynamics backend: {compute_backend!r}")
    prev = impute_matrix(batch.prev_states)
    nxt = impute_matrix(batch.next_states)

    alpha = 1e-2
    method = "ridge_state"
    transition: np.ndarray | None = None
    dual_coefficients: np.ndarray | None = None
    prev_n_sample: np.ndarray
    if backend == "mlx":
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError("MLX state dynamics requires the mlx package") from exc

        prev_m = mx.array(prev, dtype=mx.float32)
        nxt_m = mx.array(nxt, dtype=mx.float32)
        prev_norms_m = mx.sqrt(mx.sum(prev_m * prev_m, axis=1, keepdims=True))
        prev_norms_m = mx.where(prev_norms_m > 0, prev_norms_m, mx.ones_like(prev_norms_m))
        prev_n_m = prev_m / prev_norms_m
        nxt_norms_m = mx.sqrt(mx.sum(nxt_m * nxt_m, axis=1, keepdims=True))
        nxt_norms_m = mx.where(nxt_norms_m > 0, nxt_norms_m, mx.ones_like(nxt_norms_m))
        nxt_n_m = nxt_m / nxt_norms_m

        device_outputs: list[Any] = [prev_n_m[: min(8, n_pairs)]]
        if n_pairs < state_dim:
            # Dual ridge avoids constructing the state_dim x state_dim Gram
            # matrix: W = P.T @ (P @ P.T + αI)^-1 @ N.
            dual_gram_m = prev_n_m @ prev_n_m.T + alpha * mx.eye(n_pairs)
            dual_coefficients_m = mx.linalg.solve(
                dual_gram_m,
                nxt_n_m,
                # Dense solve is CPU-stream-only in the current MLX Metal
                # release; normalization and state-space matmuls stay on GPU.
                stream=mx.cpu,
            )
            pred_m = nxt_n_m - alpha * dual_coefficients_m
            rank_matrix_m = dual_coefficients_m
            method = "dual_ridge_state"
            device_outputs.extend([pred_m, rank_matrix_m])
            transition_m = None
            if checkpoint and state_dim <= 4096:
                transition_m = prev_n_m.T @ dual_coefficients_m
                device_outputs.append(transition_m)
        else:
            gram_m = prev_n_m.T @ prev_n_m + alpha * mx.eye(state_dim)
            transition_m = mx.linalg.solve(
                gram_m,
                prev_n_m.T @ nxt_n_m,
                stream=mx.cpu,
            )
            pred_m = prev_n_m @ transition_m
            rank_matrix_m = transition_m
            device_outputs.extend([pred_m, rank_matrix_m])

        centered_nxt_m = nxt_n_m - mx.mean(nxt_n_m, axis=0, keepdims=True)
        ss_tot_m = mx.sum(centered_nxt_m * centered_nxt_m)
        ss_res_m = mx.sum((nxt_n_m - pred_m) ** 2)
        r2_m = mx.array(1.0) - ss_res_m / mx.maximum(ss_tot_m, mx.array(1e-12))
        device_outputs.append(r2_m)
        mx.eval(*device_outputs)
        prev_n_sample = np.asarray(device_outputs[0], dtype=float)
        rank_matrix = np.asarray(rank_matrix_m, dtype=float)
        if n_pairs < state_dim:
            dual_coefficients = rank_matrix
            if checkpoint and state_dim <= 4096:
                transition = np.asarray(transition_m, dtype=float)
        else:
            transition = rank_matrix
        r2 = float(np.asarray(r2_m)) if float(np.asarray(ss_tot_m)) > 0.0 else float("nan")
    else:
        norms = np.linalg.norm(prev, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        prev_n = prev / norms
        nxt_norms = np.linalg.norm(nxt, axis=1, keepdims=True)
        nxt_norms = np.where(nxt_norms > 0, nxt_norms, 1.0)
        nxt_n = nxt / nxt_norms
        prev_n_sample = prev_n[: min(8, n_pairs)]
        if n_pairs < state_dim:
            # Dual ridge avoids constructing the state_dim x state_dim Gram matrix.
            dual_gram = prev_n @ prev_n.T + alpha * np.eye(n_pairs)
            dual_coefficients = np.linalg.solve(dual_gram, nxt_n)
            pred = nxt_n - alpha * dual_coefficients
            rank_matrix = dual_coefficients
            method = "dual_ridge_state"
            if checkpoint and state_dim <= 4096:
                transition = prev_n.T @ dual_coefficients
        else:
            gram = prev_n.T @ prev_n + alpha * np.eye(state_dim)
            transition = np.linalg.solve(gram, prev_n.T @ nxt_n)
            pred = prev_n @ transition
            rank_matrix = transition
        ss_tot = float(np.sum((nxt_n - nxt_n.mean(axis=0)) ** 2))
        ss_res = float(np.sum((nxt_n - pred) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12)) if ss_tot > 0 else float("nan")

    rank = _effective_rank(rank_matrix)
    low_rank = rank <= max(1, int(low_rank_ratio * state_dim))

    if checkpoint:
        checkpoint_arrays: dict[str, np.ndarray] = {
            "prev_states_sample": prev_n_sample,
        }
        if transition is not None:
            checkpoint_arrays["transition"] = transition
        else:
            checkpoint_arrays["dual_coefficients"] = dual_coefficients
        save_latent_checkpoint(
            store_root,
            run_id,
            "state_dynamics",
            arrays=checkpoint_arrays,
            metadata={
                "method": method,
                "train_r_squared": r2,
                "n_pairs": n_pairs,
                "state_dim": state_dim,
                "transition_rank": rank,
                "low_rank": low_rank,
                "checkpoint_layout": (
                    "dense_transition"
                    if transition is not None
                    else "dual_coefficients"
                ),
            },
        )

    return StateDynamicsResult(
        train_r_squared=r2,
        n_pairs=n_pairs,
        state_dim=state_dim,
        transition_rank=rank,
        low_rank=low_rank,
        method=method,
    )
