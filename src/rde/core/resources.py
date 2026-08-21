"""Resource model and query-cost ledger (RDE v0.2 Track B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rde.core.protocols import QueryIntent, ResourceModel

_DESCRIPTOR_COST: dict[str, float] = {
    "matrix.": 1.0,
    "graph.": 1.0,
    "landscape.": 2.0,
    "spectral.": 1.5,
    "stats.": 0.5,
    "compress.": 3.0,
    "fourier.": 4.0,
    "walsh_log.": 5.0,
    "dynamics.": 2.0,
    "overlap.": 2.0,
    "update.": 2.0,
    "recurrence.": 2.5,
    "gen.": 0.25,
}

_INTENT_COST: dict[QueryIntent, float] = {
    QueryIntent.EVALUATE: 1.0,
    QueryIntent.OVERLAP: 2.0,
    QueryIntent.UPDATE: 2.0,
    QueryIntent.COMPRESS: 1.5,
    QueryIntent.RANK: 3.0,
}

_DEFAULT_RESOURCE = ResourceModel(
    classical_time="poly(N)",
    memory="poly(N)",
    gates="unknown",
    depth="unknown",
    shots="0",
    precision="machine",
)


def charge_descriptor(name: str) -> float:
    """Proxy classical cost for one descriptor column."""
    for prefix, cost in _DESCRIPTOR_COST.items():
        if name.startswith(prefix):
            return cost
    if name.startswith("metric."):
        return 0.1
    if name.startswith("latent."):
        return 4.0
    return 1.0


def charge_intent(intent: QueryIntent) -> float:
    return _INTENT_COST.get(intent, 1.0)


def ledger_for_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate descriptor charges over flat feature rows."""
    if not rows:
        return {"n_rows": 0, "descriptor_charges": {}, "total_charge": 0.0}
    columns = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    charges = {col: charge_descriptor(col) for col in columns if not col.startswith("metric.")}
    per_row = sum(charges.values())
    sizes = sorted({int(r["size"]) for r in rows if r.get("size") is not None})
    return {
        "n_rows": len(rows),
        "n_descriptor_columns": len(charges),
        "descriptor_charges": charges,
        "charge_per_row": per_row,
        "total_charge": per_row * len(rows),
        "sizes": sizes,
        "resource_model": _DEFAULT_RESOURCE.to_dict(),
    }


def draft_poly_fields(
    *,
    n_sizes: int,
    max_latent_dim: float,
    query_intents: list[str] | None = None,
) -> dict[str, str]:
    """Draft ALGO-card poly_gates / poly_shots fields for certifier handoff."""
    intents = query_intents or ["EVALUATE", "COMPRESS"]
    gates = "yes" if max_latent_dim <= 64 * max(n_sizes, 1) else "no"
    shots = "yes" if all(i in ("COMPRESS", "RANK", "EVALUATE") for i in intents) else "theoretical-only"
    return {"poly_gates": gates, "poly_shots": shots}
