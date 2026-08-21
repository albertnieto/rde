"""Evaluate a RecoveryProtocol catalog against a RecoveryDomain population."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from rde.core.protocols import RecoveryDomain, RecoveryProtocol


@dataclass(frozen=True)
class RecoveryRow:
    protocol_id: str
    family: str
    size: int
    seed: int
    matched: bool
    recovered: Any
    planted: Any
    queries_used: int


@dataclass(frozen=True)
class RecoveryReport:
    rows: tuple[RecoveryRow, ...]
    budget_axis: str

    def rate(self, protocol_id: str, family: str, size: int | None = None) -> float:
        selected = [
            row
            for row in self.rows
            if row.protocol_id == protocol_id
            and row.family == family
            and (size is None or row.size == size)
        ]
        if not selected:
            return float("nan")
        return float(sum(row.matched for row in selected)) / float(len(selected))

    def matrix(self) -> dict[str, dict[str, float]]:
        """protocol → family → recovery rate (sizes pooled)."""
        grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for row in self.rows:
            grouped[(row.protocol_id, row.family)].append(row.matched)
        out: dict[str, dict[str, float]] = {}
        for (protocol_id, family), hits in grouped.items():
            out.setdefault(protocol_id, {})[family] = float(sum(hits)) / float(len(hits))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_axis": self.budget_axis,
            "n_rows": len(self.rows),
            "matrix": self.matrix(),
        }


def evaluate_protocols(
    domain: RecoveryDomain,
    instances: Sequence[Any],
    protocols: Sequence[RecoveryProtocol],
    *,
    rng: np.random.Generator | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> RecoveryReport:
    """Score every protocol on every instance. Protocols never see ``planted``."""
    rng = np.random.default_rng() if rng is None else rng
    rows: list[RecoveryRow] = []
    total = len(instances) * len(protocols)
    done = 0
    for instance in instances:
        tape = domain.draw_tape(instance, rng)
        planted = domain.planted(instance)
        family = domain.family_of(instance)
        size = domain.size_of(instance)
        for protocol in protocols:
            recovered = protocol.extract(tape)
            matched = bool(domain.match(recovered, planted))
            rows.append(
                RecoveryRow(
                    protocol_id=protocol.protocol_id,
                    family=family,
                    size=int(size),
                    seed=int(getattr(instance, "seed", 0)),
                    matched=matched,
                    recovered=recovered,
                    planted=planted,
                    queries_used=int(tape.budget),
                )
            )
            done += 1
            if on_progress is not None:
                on_progress(f"{done}/{total} {protocol.protocol_id} {family} n={size}")
    return RecoveryReport(rows=tuple(rows), budget_axis="oracle_queries")
