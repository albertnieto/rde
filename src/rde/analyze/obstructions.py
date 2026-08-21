"""Obstruction witness panel (RDE v0.2 Track C1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rde.analyze.baselines import BaselineMatch, compare_baselines
from rde.analyze.scaling import ScalingFit, fit_all_witnesses


@dataclass
class WitnessRecord:
    name: str
    median_by_size: dict[int, float] = field(default_factory=dict)
    scaling: ScalingFit | None = None
    baseline_matches: list[BaselineMatch] = field(default_factory=list)


_WITNESS_PATTERNS = (
    "balanced_slice_rank",
    "log_slice_rank",
    "recurrence.estimated_order",
    "spectral.effective_rank",
    "walsh_log.",
    "compress.gzip_bits",
    "compress.kolmogorov_proxy",
    "dynamics.slice_rank_end",
    "dynamics.entropy_end",
)


def discover_witness_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    cols: list[str] = []
    keys = set(rows[0].keys())
    for pattern in _WITNESS_PATTERNS:
        if pattern.endswith("."):
            cols.extend(sorted(k for k in keys if k.startswith(pattern)))
        else:
            for k in keys:
                if pattern in k and k not in cols:
                    cols.append(k)
    return cols[:24]


def _median_by_size(rows: list[dict[str, Any]], column: str) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for row in rows:
        size = row.get("size")
        val = row.get(column)
        if size is None or val is None:
            continue
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            buckets.setdefault(int(size), []).append(fv)
    return {s: float(np.median(vs)) for s, vs in buckets.items() if vs}


def witness_panel(rows: list[dict[str, Any]], target: str) -> list[WitnessRecord]:
    del target
    columns = discover_witness_columns(rows)
    fits = {f.witness: f for f in fit_all_witnesses(rows, columns)}
    all_baselines = compare_baselines(list(fits.values()), rows)
    baseline_by_witness: dict[str, list[BaselineMatch]] = {}
    for bm in all_baselines:
        baseline_by_witness.setdefault(bm.witness, []).append(bm)

    records: list[WitnessRecord] = []
    for col in columns:
        records.append(
            WitnessRecord(
                name=col,
                median_by_size=_median_by_size(rows, col),
                scaling=fits.get(col),
                baseline_matches=baseline_by_witness.get(col, []),
            )
        )
    return records


def panel_summary(records: list[WitnessRecord]) -> dict[str, Any]:
    exponential = [r.name for r in records if r.scaling and r.scaling.exponential_growth]
    baseline_hits = [
        {"witness": r.name, "baseline": bm.baseline_id, "reason": bm.reason}
        for r in records
        for bm in r.baseline_matches
        if bm.matched
    ]
    return {
        "n_witnesses": len(records),
        "exponential_witnesses": exponential,
        "baseline_hits": baseline_hits,
        "negative_outcome": len(exponential) >= 2 and len(baseline_hits) >= 1,
    }
