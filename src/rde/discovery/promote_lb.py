"""Emit scoped lower-bound conjectures (v0.2 Track C5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rde.analyze.obstructions import WitnessRecord, panel_summary, witness_panel


def build_lower_bound_conjectures(
    rows: list[dict[str, Any]],
    *,
    target: str,
    run_id: str,
    representation_class: str = "bound_trajectory",
    family: str = "random_dense_QUBO",
) -> list[dict[str, Any]]:
    """Build proof-obligation templates per foundations §22.3."""
    records = witness_panel(rows, target)
    summary = panel_summary(records)
    conjectures: list[dict[str, Any]] = []
    for rec in records:
        if rec.scaling is None or not rec.scaling.exponential_growth:
            continue
        baselines = [bm.baseline_id for bm in rec.baseline_matches if bm.matched]
        conjectures.append(
            {
                "claim_type": "scoped_lower_bound_conjecture",
                "run_id": run_id,
                "family": family,
                "representation_class": representation_class,
                "target": target,
                "query_intents": ["EVALUATE", "OVERLAP", "UPDATE"],
                "accuracy": "exact",
                "resource_model": "classical_time_poly_N",
                "witness": rec.name,
                "empirical_scaling": rec.scaling.empirical_scaling,
                "slope": rec.scaling.slope,
                "known_baselines": baselines or ["none_matched"],
                "proof_obligations": [
                    f"Prove witness {rec.name} lower bound for all Q in {family}",
                    f"Show no poly-size {representation_class} representation avoids witness growth",
                ],
            }
        )
    if summary["negative_outcome"] and not conjectures:
        conjectures.append(
            {
                "claim_type": "scoped_lower_bound_conjecture",
                "run_id": run_id,
                "family": family,
                "representation_class": representation_class,
                "witness": "panel_aggregate",
                "empirical_scaling": "multiple exponential witnesses",
                "known_baselines": [h["baseline"] for h in summary["baseline_hits"]],
                "proof_obligations": ["Characterize obstruction class saturating across witnesses"],
            }
        )
    return conjectures


def write_lower_bound_conjectures_jsonl(
    conjectures: list[dict[str, Any]],
    path: Path | str,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for rec in conjectures:
            handle.write(json.dumps(rec) + "\n")
    return out
