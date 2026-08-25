"""Durable reports for representation-search results — a standalone file, or the store.

Gap closure: `Certificate`/`SearchCandidate` results had `.to_payload()`
methods that went nowhere durable. `write_search_report`/
`write_diagonalization_report` write a JSON file, mirroring the shape of
the existing `rde.discovery` report writers (`write_phase4_report` et al.
in `rde/discovery/report.py`) rather than forcing representation-search
results into the campaign feature-table schema (`rde.core.schema`) — that
schema is purpose-built for per-instance descriptor/metric rows keyed by
`(instance_id, family_index)`, and representation search has neither;
reusing it would need a schema change made carelessly under an unrelated
task, not a genuine fit.

`write_search_report_to_store`/`write_diagonalization_report_to_store`
append the same payload through `rde.io.store.Store.append_representation_report`
instead — a `run_id`-scoped, appendable JSONL file living alongside a run's
other artifacts, for callers who already have a `Store` and want
representation-search results discoverable there rather than at a
one-off path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from rde.representation.operator import DiagonalizationCandidate
from rde.representation.search import SearchCandidate

if TYPE_CHECKING:
    from rde.io.store import Store


def search_report_payload(candidates: Sequence[SearchCandidate]) -> dict[str, Any]:
    """JSON-serializable payload for a `search.rank_representations` result."""
    return {
        "kind": "representation_search_report",
        "candidates": [
            {
                "representation_id": c.representation_id,
                "complexity": c.complexity,
                "conversion_cost": c.conversion_cost,
                "certificate": c.certificate.to_payload(),
            }
            for c in candidates
        ],
    }


def write_search_report(candidates: Sequence[SearchCandidate], path: str | Path) -> Path:
    """Write `search_report_payload(candidates)` to `path` as indented JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(search_report_payload(candidates), indent=2))
    return out


def diagonalization_report_payload(candidates: Sequence[DiagonalizationCandidate]) -> dict[str, Any]:
    """JSON-serializable payload for an `operator.rank_by_diagonalization` result.

    Excludes `transported_operator` (a NumPy array, and reconstructible from
    the representation id + the original operator) — only the ranked
    summary is durable, not a redundant copy of derived data.
    """
    return {
        "kind": "operator_diagonalization_report",
        "candidates": [
            {"representation_id": c.representation_id, "off_diagonal_energy": c.off_diagonal_energy}
            for c in candidates
        ],
    }


def write_diagonalization_report(
    candidates: Sequence[DiagonalizationCandidate], path: str | Path
) -> Path:
    """Write `diagonalization_report_payload(candidates)` to `path` as indented JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagonalization_report_payload(candidates), indent=2))
    return out


def write_search_report_to_store(
    candidates: Sequence[SearchCandidate], store: "Store", run_id: str
) -> None:
    """Append `search_report_payload(candidates)` via `Store.append_representation_report`."""
    store.append_representation_report(run_id, search_report_payload(candidates))


def write_diagonalization_report_to_store(
    candidates: Sequence[DiagonalizationCandidate], store: "Store", run_id: str
) -> None:
    """Append `diagonalization_report_payload(candidates)` via `Store.append_representation_report`."""
    store.append_representation_report(run_id, diagonalization_report_payload(candidates))
