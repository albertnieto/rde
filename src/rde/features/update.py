"""Update-step descriptors for r-dynamics (QueryIntent.UPDATE)."""

from __future__ import annotations

import numpy as np


def elementwise_update_ratio(prev: np.ndarray, nxt: np.ndarray) -> float:
    a = np.asarray(prev, dtype=float).ravel()
    b = np.asarray(nxt, dtype=float).ravel()
    if a.size != b.size:
        return float("nan")
    denom = np.linalg.norm(a)
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(b - a) / denom)


def weighted_step_operator_norm(prev: np.ndarray, nxt: np.ndarray, w_amp: np.ndarray) -> float:
    """||Psi_{r+1} - normalize(w_amp ⊙ Psi_r)||_2 after one multiply step."""
    a = np.asarray(prev, dtype=float).ravel()
    b = np.asarray(nxt, dtype=float).ravel()
    w = np.asarray(w_amp, dtype=float).ravel()
    if a.size != b.size or a.size != w.size:
        return float("nan")
    predicted = w * a
    norm = float(np.linalg.norm(predicted))
    if norm <= 0:
        return float("nan")
    predicted = predicted / norm
    return float(np.linalg.norm(b - predicted))


def compute_update_descriptors(
    indices: list[int],
    slices: list[np.ndarray],
    w_amp: np.ndarray | None = None,
) -> dict[str, float]:
    if len(indices) < 2:
        return {}
    order = sorted(range(len(indices)), key=lambda i: indices[i])
    sorted_slices = [
        np.asarray(slices[i], dtype=float).ravel()
        for i in order
    ]
    same_shape = bool(sorted_slices) and len({arr.shape for arr in sorted_slices}) == 1
    if same_shape:
        prev = np.stack(sorted_slices[:-1])
        nxt = np.stack(sorted_slices[1:])
        denominators = np.linalg.norm(prev, axis=1)
        differences = np.linalg.norm(nxt - prev, axis=1)
        arr = np.divide(
            differences,
            denominators,
            out=np.full_like(differences, np.nan),
            where=denominators > 0,
        )
    else:
        arr = np.asarray(
            [
                elementwise_update_ratio(sorted_slices[i], sorted_slices[i + 1])
                for i in range(len(sorted_slices) - 1)
            ],
            dtype=float,
        )
    out: dict[str, float] = {
        "update.n_steps": float(arr.size),
        "update.step_ratio_mean": float(np.mean(arr)),
        "update.step_ratio_max": float(np.max(arr)),
        "update.step_ratio_std": float(np.std(arr)),
    }
    if w_amp is not None:
        w = np.asarray(w_amp, dtype=float).ravel()
        if same_shape and w.size == prev.shape[1]:
            predicted = prev * w[None, :]
            predicted_norms = np.linalg.norm(predicted, axis=1)
            normalized = np.divide(
                predicted,
                predicted_norms[:, None],
                out=predicted.copy(),
                where=predicted_norms[:, None] > 0,
            )
            op_errors = np.linalg.norm(nxt - normalized, axis=1)
            op_errors[predicted_norms <= 0] = np.nan
            on = op_errors
        elif same_shape:
            on = np.full(arr.size, np.nan, dtype=float)
        else:
            on = np.asarray(
                [
                    weighted_step_operator_norm(
                        sorted_slices[i],
                        sorted_slices[i + 1],
                        w,
                    )
                    for i in range(len(sorted_slices) - 1)
                ],
                dtype=float,
            )
        out["update.weighted_operator_norm_mean"] = float(np.mean(on))
        out["update.weighted_operator_norm_max"] = float(np.max(on))
        out["update.operator_norm_mean"] = float(np.mean(on))
        for r, error in zip(
            [indices[j] for j in order[:-1]],
            on,
        ):
            out[f"update.w_amp_model_error_r{r}"] = float(error)
        vals = on
        if vals.size:
            out["update.w_amp_model_error_mean"] = float(np.mean(vals))
    return out
