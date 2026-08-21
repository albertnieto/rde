"""Buffered JSONL write stress test (million-row scale proxy)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from rde.io.store import JsonlWriter, RunManifest, Store, count_jsonl_lines


def stress_write_jsonl(
    path: Path,
    n_rows: int,
    *,
    batch_size: int = 256,
    on_batch: Callable[[int], None] | None = None,
) -> int:
    """Write n_rows minimal JSONL records; returns bytes on disk."""
    writer = JsonlWriter(path, batch_size=batch_size)
    for i in range(n_rows):
        writer.append({"idx": i, "value": float(i % 1000) / 1000.0})
        if on_batch and (i + 1) % batch_size == 0:
            on_batch(batch_size)
    writer.close()
    remainder = n_rows % batch_size
    if on_batch and remainder:
        on_batch(remainder)
    return path.stat().st_size


def read_jsonl_rows(path: Path) -> list[dict]:
    """Read all rows from a JSONL file (stress-test helper)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stress_store_features(
    store_root: Path | str,
    run_id: str,
    n_rows: int,
    *,
    batch_size: int = 512,
    on_batch: Callable[[int], None] | None = None,
) -> dict[str, int | str]:
    """Write n_rows schema-valid feature rows through Store; return stats."""
    store = Store(store_root, jsonl_batch_size=batch_size)
    manifest = RunManifest(
        run_id=run_id,
        domain_id="stress",
        n_instances=n_rows,
        size=4,
        seed=0,
        indices=[1],
    )
    store.write_manifest(manifest)
    for i in range(n_rows):
        store.append_features(
            run_id,
            {
                "run_id": run_id,
                "instance_id": f"stress-{i}",
                "domain_id": "stress",
                "size": 4 + (i % 3),
                "seed": i,
                "family_index": 1,
                "slice_kind": "stress",
                "generator": f"stress_gen_{i % 7}",
                "descriptors": {"stress.idx": float(i)},
                "metrics": {"representation_complexity": float(i % 100) / 100.0},
            },
        )
        if on_batch and (i + 1) % batch_size == 0:
            on_batch(batch_size)
    store.close()
    remainder = n_rows % batch_size
    if on_batch and remainder:
        on_batch(remainder)
    path = store.run_dir(run_id) / "features.jsonl"
    return {
        "run_id": run_id,
        "n_rows": count_jsonl_lines(path),
        "bytes": path.stat().st_size if path.exists() else 0,
        "batch_size": batch_size,
    }
