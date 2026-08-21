"""Load and query calibrated (N, B) backend crossover tables."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

_TABLE_PATH = Path(__file__).with_name("crossover_table.json")
_MLX_CROSSOVER_N = 14


@lru_cache(maxsize=1)
def load_crossover_table() -> dict[str, Any] | None:
    if not _TABLE_PATH.is_file():
        return None
    try:
        return json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cells_by_n(table: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for cell in table.get("cells", []):
        n = int(cell["n"])
        grouped.setdefault(n, []).append(cell)
    for n in grouped:
        grouped[n].sort(key=lambda c: int(c["batch"]))
    return grouped


def _nearest_batch(cells: list[dict[str, Any]], batch_size: int) -> dict[str, Any] | None:
    if not cells:
        return None
    exact = [c for c in cells if int(c["batch"]) == batch_size]
    if exact:
        return exact[0]
    lower = [c for c in cells if int(c["batch"]) <= batch_size]
    if lower:
        return lower[-1]
    return cells[0]


def _fallback_backend(size: int) -> str:
    try:
        from rde.backends.resolve import mlx_usable

        if size >= _MLX_CROSSOVER_N and mlx_usable():
            return "mlx"
    except ImportError:
        pass
    return "numpy"


def backend_for_size_and_batch(size: int, batch_size: int = 1) -> str:
    """Return best backend for (N, B) from the shipped table, or size-only fallback."""
    table = load_crossover_table()
    if table is None:
        return _fallback_backend(size)
    grouped = _cells_by_n(table)
    n_cells = grouped.get(size)
    if not n_cells:
        nearest_n = min(grouped.keys(), key=lambda n: abs(n - size), default=None)
        if nearest_n is None:
            return _fallback_backend(size)
        n_cells = grouped[nearest_n]
    cell = _nearest_batch(n_cells, batch_size)
    if cell is None:
        return _fallback_backend(size)
    by_backend: dict[str, float] = {}
    target_batch = int(cell["batch"])
    for c in n_cells:
        if int(c["batch"]) == target_batch:
            by_backend[str(c["backend"])] = float(c["instances_per_s"])
    if not by_backend:
        return str(cell.get("backend", _fallback_backend(size)))
    return max(by_backend, key=by_backend.get)  # type: ignore[arg-type]


def recommended_batch_size(size: int, *, prefer: str | None = None) -> int:
    """Largest calibrated batch where ``prefer`` (or auto winner) is fastest."""
    table = load_crossover_table()
    if table is None:
        return 1
    grouped = _cells_by_n(table)
    n_cells = grouped.get(size)
    if not n_cells:
        nearest_n = min(grouped.keys(), key=lambda n: abs(n - size), default=None)
        if nearest_n is None:
            return 1
        n_cells = grouped[nearest_n]
    batches = sorted({int(c["batch"]) for c in n_cells})
    chosen = prefer
    best_batch = 1
    for batch in batches:
        winner = backend_for_size_and_batch(size, batch)
        if chosen is None:
            if batch == 1:
                chosen = winner
            best_batch = batch
        elif winner == chosen:
            best_batch = batch
    return max(1, best_batch)


def write_crossover_table(
    cells: list[dict[str, Any]],
    *,
    path: Path | None = None,
) -> Path:
    """Write calibration output and invalidate the loader cache."""
    out = path or _TABLE_PATH
    best: list[dict[str, Any]] = []
    grouped = _cells_by_n({"cells": cells})
    for n, n_cells in sorted(grouped.items()):
        by_batch: dict[int, dict[str, float]] = {}
        for c in n_cells:
            b = int(c["batch"])
            by_batch.setdefault(b, {})[str(c["backend"])] = float(c["instances_per_s"])
        for batch, rates in sorted(by_batch.items()):
            winner = max(rates, key=rates.get)  # type: ignore[arg-type]
            best.append({"n": n, "batch": batch, "backend": winner})
    payload = {
        "platform": platform.platform(),
        "created": datetime.now(timezone.utc).isoformat(),
        "cells": cells,
        "best": best,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    load_crossover_table.cache_clear()
    return out
