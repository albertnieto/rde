"""Materialize generated descriptor columns from stored slice arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rde.descriptor_gen.compute import evaluate_template
from rde.descriptor_gen.spec import DescriptorTemplate
from rde.features.fourier import walsh_hadamard_batch
from rde.io.store import Store

_FAMILY_ALIASES = {
    "massive_walsh": "fourier",
    "massive_spectral": "spectral",
    "massive_autocorr": "autocorr",
    "massive_stats": "stats",
}


def _array_name_from_ref(array_ref: str) -> str:
    # arrays/{instance_id}/{name}.npz
    name = array_ref.rsplit("/", 1)[-1]
    if name.endswith(".npz"):
        name = name[: -len(".npz")]
    return name


def load_row_arrays(
    run_id: str,
    store_root: Path | str,
) -> dict[tuple[str, int], np.ndarray]:
    """Map (instance_id, family_index) -> slice array payload."""
    store = Store(store_root)
    out: dict[tuple[str, int], np.ndarray] = {}
    for row in store.read_features(run_id):
        instance_id = row.get("instance_id")
        family_index = row.get("family_index")
        array_ref = row.get("array_ref")
        if instance_id is None or family_index is None or not array_ref:
            continue
        name = _array_name_from_ref(str(array_ref))
        try:
            out[(str(instance_id), int(family_index))] = store.load_array(
                run_id, str(instance_id), name
            )
        except (FileNotFoundError, OSError, KeyError):
            continue
    return out


def materialize_template_column(
    rows: list[dict[str, Any]],
    template: DescriptorTemplate,
    arrays: dict[tuple[str, int], np.ndarray],
    *,
    walsh_cache: dict[int, np.ndarray] | None = None,
    spectral_cache: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate one template across flat feature rows."""
    values = np.full(len(rows), np.nan, dtype=float)
    family = _FAMILY_ALIASES.get(template.family, template.family)
    if family in {"stats", "norm", "compress"}:
        return _materialize_reduction_column(rows, template, arrays, values)
    if family in {"fourier", "spectral", "autocorr"}:
        return _materialize_transform_column(rows, template, arrays, values)
    cache = walsh_cache if walsh_cache is not None else {}
    svd_cache = spectral_cache if spectral_cache is not None else {}
    for i, row in enumerate(rows):
        key = (str(row.get("instance_id", "")), int(row.get("family_index", -1)))
        array = arrays.get(key)
        if array is None:
            continue
        values[i] = evaluate_template(
            template,
            array,
            walsh_cache=cache,
            spectral_cache=svd_cache,
        )
    return values


def _materialize_transform_column(
    rows: list[dict[str, Any]],
    template: DescriptorTemplate,
    arrays: dict[tuple[str, int], np.ndarray],
    values: np.ndarray,
) -> np.ndarray:
    """Batch transform-based templates by compatible array shape.

    Fourier, spectral, and autocorrelation templates all share expensive
    per-array work.  Grouping compatible payloads lets NumPy execute that
    work over a batch instead of entering ``evaluate_template`` once per row.
    """
    family = _FAMILY_ALIASES.get(template.family, template.family)
    groups: dict[tuple[int, ...], list[tuple[int, np.ndarray]]] = {}
    for i, row in enumerate(rows):
        key = (str(row.get("instance_id", "")), int(row.get("family_index", -1)))
        array = arrays.get(key)
        if array is None:
            continue
        arr = np.asarray(array, dtype=float)
        shape_key = arr.shape if family == "spectral" else (arr.size,)
        groups.setdefault(shape_key, []).append((i, arr))

    for entries in groups.values():
        idx = np.fromiter((i for i, _ in entries), dtype=int, count=len(entries))
        if family == "fourier":
            stacked = np.stack([arr.ravel() for _, arr in entries])
            n = stacked.shape[1]
            if n == 0 or (n & (n - 1)) != 0:
                continue
            in_unit_interval = np.all(
                (stacked >= 0.0) & (stacked <= 1.0),
                axis=1,
            )
            pm1 = stacked.copy()
            pm1[in_unit_interval] = 2.0 * pm1[in_unit_interval] - 1.0
            abs_coeffs = np.abs(walsh_hadamard_batch(pm1))
            totals = np.sum(abs_coeffs, axis=1)
            lo = max(0, min(int(template.params["lo"]), n))
            hi = max(lo + 1, min(int(template.params["hi"]), n))
            numerators = np.sum(abs_coeffs[:, lo:hi], axis=1)
            valid = totals > 1e-15
            values[idx[valid]] = numerators[valid] / totals[valid]
        elif family == "spectral":
            stacked = np.stack([arr for _, arr in entries])
            if stacked.ndim == 2:
                profiles = np.sort(np.abs(stacked), axis=1)[:, ::-1]
            else:
                profiles = np.linalg.svd(stacked, compute_uv=False)
            positive = profiles > 1e-15
            counts = np.sum(positive, axis=1)
            positive_profiles = np.where(positive, profiles, 0.0)
            kind = template.params["kind"]
            if kind == "sv_ratio":
                i = int(template.params["i"])
                j = int(template.params["j"])
                # Match compute.py: out-of-range singular-value indices are NaN,
                # not IndexError (massive catalog asks for i,j up to 6 on small N).
                n_sv = int(profiles.shape[1]) if getattr(profiles, "ndim", 0) >= 2 else 0
                if n_sv <= 0 or i < 0 or j < 0 or i >= n_sv or j >= n_sv:
                    continue
                valid = counts > max(i, j)
                numerator = positive_profiles[:, i]
                denominator = positive_profiles[:, j]
                valid &= np.abs(denominator) > 1e-15
                values[idx[valid]] = numerator[valid] / denominator[valid]
            elif kind == "top_k_energy":
                k = max(1, min(int(template.params["k"]), profiles.shape[1]))
                energy = positive_profiles**2
                numerator = np.sum(energy[:, :k], axis=1)
                denominator = np.sum(energy, axis=1)
                valid = (counts > 0) & (denominator > 1e-15)
                values[idx[valid]] = numerator[valid] / denominator[valid]
            elif kind == "p_norm":
                p = float(template.params["p"])
                if p == float("inf"):
                    values[idx[counts > 0]] = np.max(
                        positive_profiles[counts > 0], axis=1
                    )
                elif p > 0:
                    norms = np.sum(positive_profiles**p, axis=1) ** (1.0 / p)
                    values[idx[counts > 0]] = norms[counts > 0]
            elif kind == "effective_rank":
                energy = positive_profiles**2
                denominator = np.sum(energy**2, axis=1)
                numerator = np.sum(energy, axis=1)
                valid = (counts > 0) & (denominator > 1e-15)
                values[idx[valid]] = (
                    numerator[valid] ** 2 / denominator[valid]
                )
            elif kind == "gap":
                valid = counts >= 2
                values[idx[valid]] = (
                    positive_profiles[valid, 0] - positive_profiles[valid, 1]
                )
        else:
            stacked = np.stack([arr.ravel() for _, arr in entries])
            lag = int(template.params["lag"])
            if lag < 1 or stacked.shape[1] <= lag + 1:
                continue
            left = stacked[:, :-lag]
            right = stacked[:, lag:]
            finite = np.isfinite(left) & np.isfinite(right)
            count = np.sum(finite, axis=1)
            safe_left = np.where(finite, left, 0.0)
            safe_right = np.where(finite, right, 0.0)
            left_mean = np.divide(
                np.sum(safe_left, axis=1),
                count,
                out=np.zeros_like(count, dtype=float),
                where=count > 0,
            )
            right_mean = np.divide(
                np.sum(safe_right, axis=1),
                count,
                out=np.zeros_like(count, dtype=float),
                where=count > 0,
            )
            left_centered = np.where(finite, left - left_mean[:, None], 0.0)
            right_centered = np.where(finite, right - right_mean[:, None], 0.0)
            numerator = np.sum(left_centered * right_centered, axis=1)
            denominator = np.sqrt(
                np.sum(left_centered**2, axis=1)
                * np.sum(right_centered**2, axis=1)
            )
            valid = (
                (count >= 2)
                & np.all(
                    np.where(finite, np.abs(left) < 1e100, True),
                    axis=1,
                )
                & np.all(
                    np.where(finite, np.abs(right) < 1e100, True),
                    axis=1,
                )
                & (denominator > 0)
            )
            values[idx[valid]] = numerator[valid] / denominator[valid]
    return values


def _materialize_reduction_column(
    rows: list[dict[str, Any]],
    template: DescriptorTemplate,
    arrays: dict[tuple[str, int], np.ndarray],
    values: np.ndarray,
) -> np.ndarray:
    """Vectorize simple row-wise reductions where the template is a scalar statistic.

    Rows are grouped by raveled array shape (family_index slices are ragged
    across families) so each group can be stacked into one 2D array and
    reduced with a single vectorized numpy call instead of one Python-level
    numpy call per row.
    """
    kind = template.params.get("kind")
    family = _FAMILY_ALIASES.get(template.family, template.family)
    groups: dict[tuple[int, ...], list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for i, row in enumerate(rows):
        key = (str(row.get("instance_id", "")), int(row.get("family_index", -1)))
        array = arrays.get(key)
        if array is None:
            continue
        arr = np.asarray(array, dtype=float).ravel()
        if arr.size == 0:
            continue
        groups.setdefault(arr.shape, []).append((i, arr, array))

    for entries in groups.values():
        idx = np.fromiter((i for i, _, _ in entries), dtype=int, count=len(entries))
        stacked = np.stack([arr for _, arr, _ in entries])
        if family == "norm":
            p = float(template.params["p"])
            if p == float("inf"):
                values[idx] = np.max(np.abs(stacked), axis=1)
            elif p > 0:
                values[idx] = np.sum(np.abs(stacked) ** p, axis=1) ** (1.0 / p)
        elif family == "stats" and kind == "quantile":
            values[idx] = np.quantile(stacked, float(template.params["q"]), axis=1)
        elif family == "stats" and kind == "mad":
            med = np.median(stacked, axis=1, keepdims=True)
            values[idx] = np.median(np.abs(stacked - med), axis=1)
        elif family == "stats" and kind == "iqr":
            q1, q3 = np.quantile(stacked, [0.25, 0.75], axis=1)
            values[idx] = q3 - q1
        elif family == "stats" and kind == "trimmed_mean":
            frac = float(template.params["frac"])
            if 0.0 <= frac < 0.5:
                sorted_values = np.sort(stacked, axis=1)
                lo = int(np.floor(frac * stacked.shape[1]))
                hi = int(np.ceil((1.0 - frac) * stacked.shape[1]))
                values[idx] = np.mean(sorted_values[:, lo:hi], axis=1)
        else:
            for i, _, array in entries:
                values[i] = evaluate_template(template, array)
    return values
