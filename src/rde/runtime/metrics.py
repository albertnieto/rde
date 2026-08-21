"""Composite representation-complexity metrics (not simple descriptor passthroughs)."""

from __future__ import annotations

import math

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import QueryIntent, SimpleFamilySlice
from rde.core.registry import Registry


def compressibility_ratio(descriptors: dict[str, float]) -> float:
    raw = descriptors.get("compress.raw_bytes", 0.0)
    gzip_bits = descriptors.get("compress.gzip_bits", raw)
    if raw <= 0:
        return float("nan")
    return float(gzip_bits / raw)


def representation_complexity(
    instance: InstanceRecord,
    slice_: SimpleFamilySlice,
    descriptors: dict[str, float],
) -> float:
    """Higher score => less compressible / higher effective dimension."""
    dim = max(float(slice_.values.size), 1.0)
    eff_rank = descriptors.get("spectral.effective_rank", math.sqrt(dim))
    rank_term = eff_rank / math.sqrt(dim)
    compress_term = compressibility_ratio(descriptors)
    if math.isnan(compress_term):
        compress_term = 1.0
    fourier_spread = 1.0 - descriptors.get("fourier.top_mass_fraction", 0.0)
    participation = descriptors.get("stats.participation_ratio", 1.0)
    participation_term = 1.0 / max(participation, 1e-12)
    size_bonus = math.log2(dim) / 16.0
    del instance
    return float(rank_term + compress_term + fourier_spread + 0.25 * participation_term + size_bonus)


def spectral_concentration(descriptors: dict[str, float]) -> float:
    norm = descriptors.get("spectral.l2_norm", 0.0)
    if norm <= 0:
        return float("nan")
    # For vectors: max abs / L2 — near 1 means concentrated on one component
    max_coeff = descriptors.get("fourier.max_coeff", 0.0)
    l1 = descriptors.get("fourier.l1_norm", 0.0)
    if l1 > 0:
        return float(max_coeff / (l1 / math.sqrt(max(descriptors.get("array_size", 1.0), 1.0))))
    return float(max_coeff / norm) if norm > 0 else float("nan")


def max_amplitude(
    _instance: InstanceRecord,
    slice_: SimpleFamilySlice,
    _descriptors: dict[str, float],
) -> float:
    arr = np.abs(np.asarray(slice_.values, dtype=float))
    return float(np.max(arr)) if arr.size else float("nan")


def log_max_amplitude(
    _instance: InstanceRecord,
    slice_: SimpleFamilySlice,
    _descriptors: dict[str, float],
) -> float:
    m = max_amplitude(_instance, slice_, _descriptors)
    return float(math.log2(m)) if m > 0 and math.isfinite(m) else float("nan")


def peak_probability_mass(
    _instance: InstanceRecord,
    slice_: SimpleFamilySlice,
    _descriptors: dict[str, float],
) -> float:
    """Probability mass at the largest-magnitude amplitude (EVALUATE proxy)."""
    arr = np.asarray(slice_.values, dtype=float)
    if arr.size == 0:
        return float("nan")
    idx = int(np.argmax(np.abs(arr)))
    return float(abs(arr[idx]) ** 2)


def register_composite_metrics(registry: Registry) -> None:
    registry.register_metric_fn(
        "compressibility_ratio",
        QueryIntent.COMPRESS,
        lambda _i, _s, d: compressibility_ratio(d),
    )
    registry.register_metric_fn(
        "representation_complexity",
        QueryIntent.COMPRESS,
        representation_complexity,
    )
    registry.register_metric_fn(
        "spectral_concentration",
        QueryIntent.RANK,
        lambda _i, _s, d: spectral_concentration(d),
    )
    registry.register_metric_fn(
        "overlap_consecutive_mean",
        QueryIntent.OVERLAP,
        lambda _i, _s, d: float(d.get("overlap.consecutive_mean", float("nan"))),
    )
    registry.register_metric_fn(
        "update_step_ratio_mean",
        QueryIntent.UPDATE,
        lambda _i, _s, d: float(d.get("update.step_ratio_mean", float("nan"))),
    )
    registry.register_metric_fn(
        "recurrence_estimated_order",
        QueryIntent.RANK,
        lambda _i, _s, d: float(d.get("recurrence.estimated_order", float("nan"))),
    )
    registry.register_metric_fn("max_amplitude", QueryIntent.EVALUATE, max_amplitude)
    registry.register_metric_fn("log_max_amplitude", QueryIntent.EVALUATE, log_max_amplitude)
    registry.register_metric_fn("peak_probability_mass", QueryIntent.EVALUATE, peak_probability_mass)
