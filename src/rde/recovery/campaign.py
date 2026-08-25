"""Protocol-search campaign: one tape per instance, score the full catalog.

Discovery uses even seeds; confirmatory uses odd seeds. The search never
peeks at planted K inside an extractor.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rde.recovery.programs import (
    DISCOVERY_FAMILIES,
    PIPELINE_PROTOCOL_BY_FAMILY,
    TEXTBOOK_PROTOCOL_IDS,
)
from rde.recovery.search import RecoveryRow

PIPELINE_MIN_RECALL = 0.80
DISCOVERY_MIN_RECALL = 0.80
DISCOVERY_SIZES = (8, 10, 12)
CONFIRMATORY_SIZES = (8, 10, 12, 16, 20, 24)


@dataclass(frozen=True)
class ProtocolSearchVerdict:
    verdict: str
    grade: int
    pipeline_ok: bool
    candidates: tuple[str, ...]
    confirmed: tuple[str, ...]
    criteria: dict[str, Any]
    matrix_confirm: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "grade": self.grade,
            "pipeline_ok": self.pipeline_ok,
            "candidates": list(self.candidates),
            "confirmed": list(self.confirmed),
            "criteria": self.criteria,
            "matrix_confirm": self.matrix_confirm,
        }


def _rate(rows: Sequence[RecoveryRow], protocol_id: str, family: str, size: int | None = None) -> float:
    selected = [
        row
        for row in rows
        if row.protocol_id == protocol_id
        and row.family == family
        and (size is None or row.size == size)
    ]
    if not selected:
        return float("nan")
    return float(sum(row.matched for row in selected)) / float(len(selected))


def split_rows(rows: Sequence[RecoveryRow]) -> tuple[list[RecoveryRow], list[RecoveryRow]]:
    """Even seeds → discovery; odd seeds → confirmatory held-out."""
    disc = [row for row in rows if int(row.seed) % 2 == 0]
    conf = [row for row in rows if int(row.seed) % 2 == 1]
    return disc, conf


def _holds_all_sizes(
    rows: Sequence[RecoveryRow],
    protocol_id: str,
    family: str,
    sizes: Sequence[int],
    min_recall: float,
) -> bool:
    for size in sizes:
        rate = _rate(rows, protocol_id, family, size)
        if rate != rate or rate < min_recall:
            return False
    return True


def assess_protocol_search(rows: Sequence[RecoveryRow]) -> ProtocolSearchVerdict:
    disc, conf = split_rows(rows)
    pipeline_rates: dict[str, dict[str, float]] = {}
    pipeline_ok = True
    for family, protocol_id in PIPELINE_PROTOCOL_BY_FAMILY.items():
        per_n: dict[str, float] = {}
        for size in CONFIRMATORY_SIZES:
            rate = _rate(conf, protocol_id, family, size)
            per_n[str(size)] = rate
            if rate != rate or rate < PIPELINE_MIN_RECALL:
                pipeline_ok = False
        pipeline_rates[family] = per_n

    protocols = sorted({row.protocol_id for row in rows})
    candidates: list[str] = []
    for protocol_id in protocols:
        if protocol_id in TEXTBOOK_PROTOCOL_IDS:
            continue
        if any(
            _holds_all_sizes(disc, protocol_id, family, DISCOVERY_SIZES, DISCOVERY_MIN_RECALL)
            for family in DISCOVERY_FAMILIES
        ):
            candidates.append(protocol_id)

    confirmed: list[str] = []
    for protocol_id in candidates:
        if any(
            _holds_all_sizes(conf, protocol_id, family, CONFIRMATORY_SIZES, DISCOVERY_MIN_RECALL)
            for family in DISCOVERY_FAMILIES
        ):
            confirmed.append(protocol_id)

    if not pipeline_ok:
        verdict, grade = "NULL", 0
    elif confirmed:
        verdict, grade = "SIGNAL", 1
    else:
        verdict, grade = "NULL", 0

    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for protocol_id in protocols:
        for family in sorted({row.family for row in conf}):
            matrix[protocol_id][family] = _rate(conf, protocol_id, family)

    criteria = {
        "pipeline_min_recall": PIPELINE_MIN_RECALL,
        "discovery_min_recall": DISCOVERY_MIN_RECALL,
        "pipeline_ok": pipeline_ok,
        "catalog_size": float(len(protocols)),
        "n_candidates": float(len(candidates)),
        "n_confirmed": float(len(confirmed)),
        "pipeline_simon_n24": pipeline_rates.get("simon", {}).get("24", float("nan")),
        "pipeline_shor_n24": pipeline_rates.get("shor_cyclic", {}).get("24", float("nan")),
        "pipeline_dihedral_n24": pipeline_rates.get("dihedral_kuperberg", {}).get("24", float("nan")),
        "pipeline_rates": pipeline_rates,
    }
    return ProtocolSearchVerdict(
        verdict=verdict,
        grade=grade,
        pipeline_ok=pipeline_ok,
        candidates=tuple(candidates),
        confirmed=tuple(confirmed),
        criteria=criteria,
        matrix_confirm=dict(matrix),
    )


def append_instance_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def load_done_keys(path: Path) -> set[tuple[int, str, int]]:
    if not path.is_file():
        return set()
    done: set[tuple[int, str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((int(row["size"]), str(row["family"]), int(row["seed"])))
    return done


def rows_from_instance_records(records: Sequence[dict[str, Any]]) -> list[RecoveryRow]:
    out: list[RecoveryRow] = []
    for record in records:
        size = int(record["size"])
        family = str(record["family"])
        seed = int(record["seed"])
        budget = int(record["queries_used"])
        planted = record.get("planted")
        for protocol_id, payload in (record.get("results") or {}).items():
            out.append(
                RecoveryRow(
                    protocol_id=str(protocol_id),
                    family=family,
                    size=size,
                    seed=seed,
                    matched=bool(payload["matched"]),
                    recovered=payload.get("recovered"),
                    planted=planted,
                    queries_used=budget,
                )
            )
    return out


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
