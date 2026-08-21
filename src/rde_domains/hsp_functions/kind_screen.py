"""Catalog kind screen: literature algorithm class, not planted strength.

Charter job is signatures of a *query-complexity class* (junk / Kuperberg
dihedral / abelian HSP), not “did we plant a pairing.” Collision rate
answers the second and cannot answer the first. Heisenberg / Q8 / blends
have no closed-form quantum bound (ALGO-062) — they are scored, never fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rde.analyze.query import correlate_with_target
from rde.io.store import Store

CLASS_JUNK = 0.0
CLASS_DIHEDRAL = 1.0
CLASS_ABELIAN = 2.0
KIND_TARGET = "metric.algorithm_class"
SPAN_COLUMN = "hsp_sample.f.difference_span_dim_fraction"
PERIOD_COLUMN = "hsp_sample.f.detected_period_divisor_fraction"
COLLISION_COLUMN = "hsp_sample.f.collision_rate"

DROP_KIND_SUBSTRINGS = (
    "n_collisions_found",
    "query_budget",
    "difference_profile_query_cost",
    "collision_rate",
    "mean_collision_prob",
    "max_collision_prob",
    "unique_label_fraction",
)

ABELIAN_GENERATORS = frozenset(
    {"simon", "shor_cyclic", "hsp_recipe.xor", "hsp_recipe.cyclic"}
)
DIHEDRAL_GENERATORS = frozenset({"dihedral_kuperberg", "hsp_recipe.dihedral"})
JUNK_GENERATORS = frozenset({"generic_random_control", "hsp_recipe.random"})

MIN_DISCOVERY_ABS_R = 0.35
MIN_RECALL = 0.80


def algorithm_class_for_generator(generator: str) -> float:
    """Literature class, or NaN when Q_quantum is not a known closed form."""
    name = str(generator or "")
    if name in JUNK_GENERATORS:
        return CLASS_JUNK
    if name in DIHEDRAL_GENERATORS:
        return CLASS_DIHEDRAL
    if name in ABELIAN_GENERATORS:
        return CLASS_ABELIAN
    return float("nan")


def _family(row: Mapping[str, Any]) -> str:
    return str(row.get("generator") or row.get("family") or "")


def _finite(row: Mapping[str, Any], column: str) -> float | None:
    val = row.get(column)
    if not isinstance(val, (int, float)) or not np.isfinite(float(val)):
        return None
    return float(val)


def patch_run_algorithm_class(store_root: Path | str, run_id: str) -> int:
    """Write ``algorithm_class`` onto an existing run so leak-clean can see it."""
    store = Store(store_root)
    run_dir = store.run_dir(run_id)
    n_written = 0
    feat = run_dir / "features.jsonl"
    if feat.is_file():
        lines: list[str] = []
        for raw in feat.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            klass = algorithm_class_for_generator(str(row.get("generator") or ""))
            metrics = dict(row.get("metrics") or {})
            if np.isfinite(klass):
                metrics["algorithm_class"] = float(klass)
            else:
                metrics.pop("algorithm_class", None)
            row["metrics"] = metrics
            lines.append(json.dumps(row))
            n_written += 1
        feat.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    inst = run_dir / "instance_features.jsonl"
    if inst.is_file():
        lines = []
        for raw in inst.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            scalars = dict(row.get("scalars") or {})
            gen = str(row.get("generator") or scalars.get("generator") or "")
            klass = algorithm_class_for_generator(gen)
            if np.isfinite(klass):
                scalars["algorithm_class"] = float(klass)
            else:
                scalars.pop("algorithm_class", None)
            row["scalars"] = scalars
            lines.append(json.dumps(row))
        inst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return n_written


def evaluate_algorithm_class_screen(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_discovery_abs_r: float = MIN_DISCOVERY_ABS_R,
    min_recall: float = MIN_RECALL,
    held_out_families: Sequence[str],
) -> dict[str, Any]:
    """2-D span+period nearest kind; collision is baseline only.

    A 1-D span classifier is a bug: dihedral/cyclic rows have no XOR-span, so
    they collapse onto junk at 0. The kinds live in (span, period) together.
    Held-out textbooks must match the corresponding discovery kind:
    Simon→xor, Shor→cyclic, Kuperberg→dihedral.
    """
    held_names = tuple(held_out_families)
    labeled = [row for row in rows if _finite(row, KIND_TARGET) is not None]
    discovery = [row for row in labeled if _family(row) not in held_names]
    held_out = [row for row in labeled if _family(row) in held_names]
    hits = correlate_with_target(list(discovery), KIND_TARGET, min_abs_r=0.0)
    collision_hit = next((h for h in hits if h.get("column") == COLLISION_COLUMN), None)
    collision_abs_r = float(abs(collision_hit["pearson_r"])) if collision_hit else 0.0
    geometry_hits = [
        h
        for h in hits
        if str(h.get("column") or "").endswith("difference_span_dim_fraction")
        or str(h.get("column") or "").endswith("detected_period_divisor_fraction")
    ]
    best_geom = max(
        geometry_hits, key=lambda h: abs(float(h.get("pearson_r") or 0.0)), default=None
    )
    discovery_abs_r = float(abs(best_geom["pearson_r"])) if best_geom else 0.0
    winner = str(best_geom["column"]) if best_geom else ""

    kind_names = ("hsp_recipe.random", "hsp_recipe.dihedral", "hsp_recipe.xor", "hsp_recipe.cyclic")
    held_to_kind = {
        "simon": "hsp_recipe.xor",
        "shor_cyclic": "hsp_recipe.cyclic",
        "dihedral_kuperberg": "hsp_recipe.dihedral",
    }
    per_n: dict[str, dict[str, Any]] = {}
    recall_ok = True
    sizes = sorted({int(row["size"]) for row in rows if row.get("size") is not None})
    for size in sizes:
        disc_n = [row for row in discovery if int(row["size"]) == size]
        held_n = [row for row in held_out if int(row["size"]) == size]
        means: dict[str, np.ndarray] = {}
        for kind in kind_names:
            pts = [_xy(row) for row in disc_n if _family(row) == kind]
            if pts:
                means[kind] = np.mean(np.stack(pts, axis=0), axis=0)
        correct = 0
        n_held = 0
        for row in held_n:
            expected = held_to_kind.get(_family(row))
            if expected is None or not means:
                continue
            n_held += 1
            x = _xy(row)
            pred = min(means, key=lambda k: float(np.linalg.norm(means[k] - x)))
            if pred == expected:
                correct += 1
        recall = float(correct / n_held) if n_held else float("nan")
        per_n[str(size)] = {
            "n_held_out": n_held,
            "recall": recall,
            "kind_means": {k: v.tolist() for k, v in means.items()},
        }
        if not (np.isfinite(recall) and recall >= min_recall):
            recall_ok = False

    passed = bool(discovery_abs_r >= min_discovery_abs_r and recall_ok and winner)
    return {
        "passed": passed,
        "target": KIND_TARGET,
        "winning_column": winner,
        "classifier": "span_period_2d_nearest_kind",
        "discovery_abs_r": discovery_abs_r,
        "collision_abs_r": collision_abs_r,
        "min_discovery_abs_r": min_discovery_abs_r,
        "min_recall": min_recall,
        "n_discovery": len(discovery),
        "n_held_out": len(held_out),
        "per_n": per_n,
        "geometry_hits": [
            {"column": h.get("column"), "pearson_r": h.get("pearson_r")}
            for h in geometry_hits[:8]
        ],
        "note": (
            "Collision rate is a rediscovery baseline, not the decision column. "
            "Classifier is 2-D (span, period) nearest of xor/cyclic/dihedral/random. "
            "Heisenberg/Q8/blend rows have NaN class and are excluded from the fit."
        ),
    }


def _xy(row: Mapping[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(_finite(row, SPAN_COLUMN) or 0.0),
            float(_finite(row, PERIOD_COLUMN) or 0.0),
        ],
        dtype=float,
    )
