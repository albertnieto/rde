"""Matrix descriptors for square coefficient arrays (e.g. QUBO Q)."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice
from rde.features.spectral import effective_rank


def _prefix(name: str) -> str:
    return f"matrix.{name}" if name == "array" else f"matrix.{name}"


def descriptor_for_matrix(
    matrix: np.ndarray,
    name: str = "array",
) -> dict[str, float]:
    """Spectral and norm descriptors for a square matrix."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return {}
    p = _prefix(name)
    sym = 0.5 * (mat + mat.T)
    eigvals = np.linalg.eigvalsh(sym)
    eigvals_sorted = np.sort(np.abs(eigvals))[::-1]
    s = np.linalg.svd(mat, compute_uv=False)
    fro = float(np.linalg.norm(mat, ord="fro"))
    nuclear = float(np.sum(s))
    out: dict[str, float] = {
        f"{p}.size": float(mat.shape[0]),
        f"{p}.frobenius_norm": fro,
        f"{p}.nuclear_norm": nuclear,
        f"{p}.max_abs_entry": float(np.max(np.abs(mat))),
        f"{p}.trace": float(np.trace(mat)),
        f"{p}.effective_rank": effective_rank(s),
    }
    if eigvals.size:
        out[f"{p}.spectral_radius"] = float(np.max(np.abs(eigvals)))
        out[f"{p}.eigenvalue_entropy"] = _eigen_entropy(eigvals_sorted)
        gaps = np.diff(np.sort(eigvals))
        out[f"{p}.spectral_gap_min"] = float(np.min(np.abs(gaps))) if gaps.size else 0.0
        out[f"{p}.spectral_gap_max"] = float(np.max(np.abs(gaps))) if gaps.size else 0.0
        try:
            out[f"{p}.condition_number"] = float(s[0] / s[-1]) if s[-1] > 1e-15 else float("inf")
        except Exception:
            out[f"{p}.condition_number"] = float("nan")
        _, eigvecs = np.linalg.eigh(sym)
        participation = np.sum(eigvecs**2, axis=0)
        out[f"{p}.eigenvector_participation_mean"] = float(np.mean(participation))
        out[f"{p}.eigenvector_participation_max"] = float(np.max(participation))
    for k in (2, 3, 4):
        out[f"{p}.trace_power_{k}"] = float(np.trace(np.linalg.matrix_power(mat, k)))
    n = mat.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    off_vals = mat[off_mask]
    diag_vals = np.diag(mat)
    out[f"{p}.sparsity"] = float(np.mean(np.abs(mat) <= 1e-12))
    out[f"{p}.off_diagonal_fro_ratio"] = float(
        np.linalg.norm(off_vals) / max(np.linalg.norm(mat), 1e-12)
    )
    out[f"{p}.symmetry_error"] = float(np.linalg.norm(mat - mat.T) / max(fro, 1e-12))
    if off_vals.size:
        out[f"{p}.off_diagonal_mean_abs"] = float(np.mean(np.abs(off_vals)))
        out[f"{p}.off_diagonal_max_abs"] = float(np.max(np.abs(off_vals)))
    if diag_vals.size:
        dom = np.sum(np.abs(diag_vals))
        off_sum = float(np.sum(np.abs(off_vals))) if off_vals.size else 0.0
        out[f"{p}.diagonal_dominance"] = float(dom / max(dom + off_sum, 1e-12))
    row_vars = np.var(mat, axis=1)
    out[f"{p}.row_variance_mean"] = float(np.mean(row_vars))
    out[f"{p}.row_variance_max"] = float(np.max(row_vars))
    return out


def _eigen_entropy(values: np.ndarray) -> float:
    v = np.abs(values).astype(float)
    total = v.sum()
    if total <= 0:
        return 0.0
    p = v[v > 0] / total
    return float(-np.sum(p * np.log(p)))


def descriptor(
    _instance: InstanceRecord,
    _slice_: SimpleFamilySlice | None,
    array: np.ndarray | None,
) -> dict[str, float]:
    if array is None:
        return {}
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return descriptor_for_matrix(arr)
    return {}
