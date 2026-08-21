"""Deprecated SQLite-backed resume index (restored compatibility shim)."""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path


class ResumeIndex:
    """Track completed (instance_id, family_index) without scanning JSONL."""

    def __init__(self, db_path: Path | str) -> None:
        warnings.warn(
            "rde.io.resume_index.ResumeIndex is deprecated; prefer manifest resume metadata",
            DeprecationWarning,
            stacklevel=2,
        )
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS completed_slices "
            "(instance_id TEXT NOT NULL, family_index INTEGER NOT NULL, PRIMARY KEY (instance_id, family_index))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS completed_instances (instance_id TEXT PRIMARY KEY)"
        )
        self._conn.commit()

    def mark_slice_done(self, instance_id: str, family_index: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO completed_slices VALUES (?, ?)",
            (instance_id, int(family_index)),
        )
        self._conn.commit()

    def mark_instance_done(self, instance_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO completed_instances VALUES (?)",
            (instance_id,),
        )
        self._conn.commit()

    def completed_slices(self) -> set[tuple[str, int]]:
        cur = self._conn.execute("SELECT instance_id, family_index FROM completed_slices")
        return {(str(a), int(b)) for a, b in cur.fetchall()}

    def completed_instances(self) -> set[str]:
        cur = self._conn.execute("SELECT instance_id FROM completed_instances")
        return {str(row[0]) for row in cur.fetchall()}

    def close(self) -> None:
        self._conn.close()
