"""Checkpoint I/O for Phase 4 latent models and discovery stage resume."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rde.io.json_util import json_default

_LOG = logging.getLogger(__name__)

# Ordered discovery stages that may be resumed after a partial failure.
DISCOVERY_STAGE_ORDER = (
    "rankers",
    "phase4",
    "phase5",
    "phase6",
    "promote",
    "outcome",
)


def checkpoint_dir(store_root: Path | str, run_id: str) -> Path:
    return Path(store_root) / "discovery" / "checkpoints" / run_id


def save_latent_checkpoint(
    store_root: Path | str,
    run_id: str,
    name: str,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist model arrays (NPZ) and metadata (JSON) for resume/repro."""
    root = checkpoint_dir(store_root, run_id)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{name}.npz"
    np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in arrays.items()})
    meta_path = root / f"{name}.json"
    payload = {"run_id": run_id, "name": name, **(metadata or {})}
    meta_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return npz_path


def load_latent_checkpoint(
    store_root: Path | str,
    run_id: str,
    name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = checkpoint_dir(store_root, run_id)
    npz_path = root / f"{name}.npz"
    meta_path = root / f"{name}.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return arrays, metadata


def stage_progress_path(store_root: Path | str, run_id: str) -> Path:
    return checkpoint_dir(store_root, run_id) / "stages.json"


def population_fingerprint(rows: Sequence[dict[str, Any]], target: str) -> str:
    """Identity of the row population a checkpoint's results were computed on.

    A run_id is not sufficient to key a discovery checkpoint. Re-running an
    experiment with a changed population (a wider generator grid, different
    sizes, a new seed) rewrites the run's feature rows while keeping the same
    run_id, so resuming on run_id alone silently restores ranker output computed
    on the *previous* population and reports it against the new rows.

    The fingerprint covers row count, target, and the sorted column set plus a
    digest of the target column, which is what every resumable stage consumes.
    """
    columns = sorted({key for row in rows for key in row})
    hasher = hashlib.sha256()
    hasher.update(f"{len(rows)}|{target}|".encode())
    hasher.update("\x1f".join(columns).encode())
    for row in rows:
        hasher.update(f"|{row.get(target)}".encode())
    return hasher.hexdigest()


def _empty_progress() -> dict[str, Any]:
    return {"completed": [], "errors": [], "partial": {}}


def load_stage_progress(
    store_root: Path | str,
    run_id: str,
    *,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    """Load resumable stage progress, discarding it if the population changed.

    ``fingerprint`` is compared against the value recorded when the checkpoint
    was written. A mismatch (or a checkpoint written before fingerprints existed)
    discards the checkpoint rather than resuming, because stale stage output
    scored against a new population is worse than recomputing it.
    """
    path = stage_progress_path(store_root, run_id)
    if not path.is_file():
        return _empty_progress()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("ignoring corrupt stage progress %s: %s", path, exc)
        return _empty_progress()
    if not isinstance(payload, dict):
        return _empty_progress()
    if fingerprint is not None:
        recorded = payload.get("population_fingerprint")
        if recorded != fingerprint:
            _LOG.warning(
                "discarding discovery checkpoint for %s: population fingerprint "
                "%s does not match %s; recomputing rather than resuming stale "
                "stage output",
                run_id,
                recorded,
                fingerprint,
            )
            return _empty_progress()
    payload.setdefault("completed", [])
    payload.setdefault("errors", [])
    payload.setdefault("partial", {})
    return payload


def save_stage_progress(
    store_root: Path | str,
    run_id: str,
    *,
    completed: list[str],
    errors: list[dict[str, Any]],
    partial: dict[str, Any],
    fingerprint: str | None = None,
) -> Path:
    """Atomically persist discovery stage progress for resume after failures."""
    root = checkpoint_dir(store_root, run_id)
    root.mkdir(parents=True, exist_ok=True)
    path = stage_progress_path(store_root, run_id)
    tmp = path.with_suffix(".json.tmp")
    payload = {
        "run_id": run_id,
        "population_fingerprint": fingerprint,
        "completed": list(completed),
        "errors": list(errors),
        "partial": partial,
    }
    tmp.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    tmp.replace(path)
    return path


def stage_completed(progress: dict[str, Any], stage: str) -> bool:
    return stage in set(progress.get("completed") or [])

