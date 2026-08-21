"""Phase-2 held-out screen: size-normalized collision rate vs planted families."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rde.analyze.query import correlate_with_target
from rde_domains.hsp_functions.functions import (
    FAMILIES_DISCOVERY,
    FAMILIES_HELD_OUT,
    FAMILIES_PHASE3_DISCOVERY,
)

FROZEN_COLUMN = "hsp_sample.f.collision_rate"
SHAPE_COLUMN = "hsp_sample.f.difference_span_dim_fraction"
TARGET = "metric.structure_strength"
HELD_OUT_FAMILIES = FAMILIES_HELD_OUT
DISCOVERY_FAMILIES = FAMILIES_DISCOVERY
RANDOM_FAMILY = "generic_random_control"
DROP_COUNT_SUBSTRINGS = (
    "n_collisions_found",
    "query_budget",
    "difference_profile_query_cost",
)
MIN_DISCOVERY_ABS_R = 0.35
MIN_RECALL = 0.80
MAX_FPR = 0.05


def _family(row: Mapping[str, Any]) -> str:
    return str(row.get("generator") or row.get("family") or "")


def _finite_rate(row: Mapping[str, Any]) -> float | None:
    return _finite_col(row, FROZEN_COLUMN)


def _finite_col(row: Mapping[str, Any], column: str) -> float | None:
    val = row.get(column)
    if not isinstance(val, (int, float)) or not np.isfinite(float(val)):
        return None
    return float(val)


def threshold_from_random(rows: Sequence[Mapping[str, Any]]) -> float:
    """99th percentile of discovery-fold random-control collision rates."""
    rates: list[float] = []
    for row in rows:
        if _family(row) not in (RANDOM_FAMILY, "hsp_recipe.random"):
            continue
        rate = _finite_rate(row)
        if rate is not None:
            rates.append(rate)
    if not rates:
        return 0.0
    return float(np.quantile(np.asarray(rates, dtype=float), 0.99))


def evaluate_heldout_screen(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_discovery_abs_r: float = MIN_DISCOVERY_ABS_R,
    min_recall: float = MIN_RECALL,
    max_fpr: float = MAX_FPR,
    discovery_families: Sequence[str] | None = None,
    held_out_families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score a frozen collision-rate threshold on held-out exact families, per N."""
    discovery_names = tuple(discovery_families or DISCOVERY_FAMILIES)
    held_out_names = tuple(held_out_families or HELD_OUT_FAMILIES)
    discovery = [row for row in rows if _family(row) in discovery_names]
    held_out = [row for row in rows if _family(row) in held_out_names]
    tau = threshold_from_random(discovery)

    hits = correlate_with_target(list(discovery), TARGET, min_abs_r=0.0)
    frozen = next((hit for hit in hits if hit.get("column") == FROZEN_COLUMN), None)
    discovery_abs_r = float(abs(frozen["pearson_r"])) if frozen else 0.0

    per_n: dict[str, dict[str, Any]] = {}
    sizes = sorted({int(row["size"]) for row in rows if row.get("size") is not None})
    recall_ok = True
    fpr_ok = True
    for size in sizes:
        held_n = [row for row in held_out if int(row["size"]) == size]
        rand_n = [
            row
            for row in discovery
            if int(row["size"]) == size
            and _family(row) in (RANDOM_FAMILY, "hsp_recipe.random")
        ]
        held_rates = [_finite_rate(row) for row in held_n]
        rand_rates = [_finite_rate(row) for row in rand_n]
        held_rates = [rate for rate in held_rates if rate is not None]
        rand_rates = [rate for rate in rand_rates if rate is not None]
        recall = (
            float(np.mean([rate > tau for rate in held_rates])) if held_rates else float("nan")
        )
        fpr = (
            float(np.mean([rate > tau for rate in rand_rates])) if rand_rates else float("nan")
        )
        per_n[str(size)] = {
            "n_held_out": len(held_rates),
            "n_random": len(rand_rates),
            "recall": recall,
            "fpr": fpr,
        }
        if not (np.isfinite(recall) and recall >= min_recall):
            recall_ok = False
        if not (np.isfinite(fpr) and fpr <= max_fpr):
            fpr_ok = False

    passed = bool(
        discovery_abs_r >= min_discovery_abs_r and recall_ok and fpr_ok
    )
    return {
        "passed": passed,
        "frozen_column": FROZEN_COLUMN,
        "threshold": tau,
        "discovery_abs_r": discovery_abs_r,
        "min_discovery_abs_r": min_discovery_abs_r,
        "min_recall": min_recall,
        "max_fpr": max_fpr,
        "n_discovery": len(discovery),
        "n_held_out": len(held_out),
        "per_n": per_n,
        "discovery_families": list(discovery_names),
        "held_out_families": list(held_out_names),
    }


def summarize_pairing_shape(
    rows: Sequence[Mapping[str, Any]],
    *,
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Diagnostic: mean collision rate vs XOR-difference span, by family and N.

    Not a gated verdict. Span is only defined for gf2 (XOR) pairings;
    cyclic families (Shor, dihedral) are recorded as missing, not zero.
    """
    names = tuple(families) if families is not None else tuple(
        dict.fromkeys(
            list(HELD_OUT_FAMILIES)
            + list(DISCOVERY_FAMILIES)
            + list(FAMILIES_PHASE3_DISCOVERY)
        )
    )
    sizes = sorted({int(row["size"]) for row in rows if row.get("size") is not None})
    per_n: dict[str, dict[str, Any]] = {}
    per_n_by_rank: dict[str, dict[str, Any]] = {}
    for size in sizes:
        per_n[str(size)] = {}
        for name in names:
            group = [
                row
                for row in rows
                if int(row["size"]) == size and _family(row) == name
            ]
            per_n[str(size)][name] = _shape_stats(group)
        rank_bucket: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            if int(row["size"]) != size:
                continue
            rank_val = row.get("hsp_recipe.hidden_rank")
            if not isinstance(rank_val, (int, float)) or not np.isfinite(float(rank_val)):
                continue
            key = f"rank_{int(rank_val)}"
            rank_bucket.setdefault(key, []).append(row)
        per_n_by_rank[str(size)] = {
            key: _shape_stats(group) for key, group in sorted(rank_bucket.items())
        }
    return {
        "kind": "diagnostic",
        "gated": False,
        "collision_column": FROZEN_COLUMN,
        "shape_column": SHAPE_COLUMN,
        "note": (
            "XOR-difference span is defined only for gf2 pairings. "
            "Cyclic Shor/dihedral rows may have no span value. "
            "per_n_by_rank uses hsp_recipe.hidden_rank when present."
        ),
        "per_n": per_n,
        "per_n_by_rank": per_n_by_rank,
    }


def _shape_stats(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coll = [_finite_col(row, FROZEN_COLUMN) for row in group]
    span = [_finite_col(row, SHAPE_COLUMN) for row in group]
    coll_ok = [val for val in coll if val is not None]
    span_ok = [val for val in span if val is not None]
    return {
        "n": len(group),
        "n_collision": len(coll_ok),
        "n_span": len(span_ok),
        "mean_collision_rate": float(np.mean(coll_ok)) if coll_ok else float("nan"),
        "mean_span_fraction": float(np.mean(span_ok)) if span_ok else float("nan"),
        "hit_collision": float(np.mean([val > 0.0 for val in coll_ok])) if coll_ok else float("nan"),
    }
