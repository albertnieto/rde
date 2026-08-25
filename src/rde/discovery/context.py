"""Shared in-memory context for discovery — load features once, reuse across phases."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.analyze.query import flatten_features, numeric_columns
from rde.analyze.tables import contract_excluded_columns
from rde.discovery.datasets import (
    FeatureDataset,
    RepresentationBatch,
    build_feature_matrix_from_rows,
    load_representation_batch,
)
from rde.runtime.heartbeat import ProgressCallback, StageHeartbeat
from rde.runtime.targets import resolve_target_for_run


@dataclass
class DiscoveryContext:
    """One-run cache: flattened rows, feature matrix, optional representation batch."""

    run_id: str
    store_root: Path
    target: str
    rows: list[dict[str, Any]]
    _table: FeatureTable | None = field(default=None, repr=False)
    _X: np.ndarray | None = field(default=None, repr=False)
    _cols: list[str] | None = field(default=None, repr=False)
    _representation_batch: RepresentationBatch | None = field(default=None, repr=False)
    _feature_dataset: FeatureDataset | None = field(default=None, repr=False)
    _env_cache: dict[tuple[str, ...], dict[str, np.ndarray]] = field(default_factory=dict, repr=False)
    _domain_id_loaded: bool = field(default=False, repr=False)
    _domain_id: str | None = field(default=None, repr=False)
    timings: dict[str, float] = field(default_factory=dict)

    def domain_id(self) -> str | None:
        """Best-effort manifest lookup, cached; `None` if unreadable (fail-open)."""
        if not self._domain_id_loaded:
            from rde.io.store import Store

            try:
                self._domain_id = Store(self.store_root).read_manifest(self.run_id).domain_id
            except (FileNotFoundError, OSError, ValueError, KeyError):
                self._domain_id = None
            self._domain_id_loaded = True
        return self._domain_id

    @classmethod
    def load(
        cls,
        run_id: str,
        store_root: Path | str,
        *,
        target: str | None = None,
    ) -> DiscoveryContext:
        store_root = Path(store_root)
        resolved = resolve_target_for_run(run_id, store_root, override=target)
        rows = flatten_features(run_id, store_root)
        ctx = cls(run_id=run_id, store_root=store_root, target=resolved, rows=rows)
        ctx._table = FeatureTable.from_rows(rows, target=resolved)
        return ctx

    def feature_table(self) -> FeatureTable:
        if self._table is None:
            self._table = FeatureTable.from_rows(self.rows, target=self.target)
        return self._table

    def feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        table = self.feature_table()
        if self._X is None or self._cols is None:
            self._X, self._cols = table.data, list(table.columns)
        return self._X, self._cols

    def feature_dataset(
        self,
        *,
        feature_columns: list[str] | None = None,
        exclude_prefixes: tuple[str, ...] = ("metric.", "run_id"),
    ) -> FeatureDataset:
        if self._feature_dataset is None:
            X, cols = self.feature_matrix()
            excluded = contract_excluded_columns(self.rows, self.domain_id())
            use_cols = feature_columns or [
                c
                for c in cols
                if c != self.target and c not in excluded and not any(c.startswith(p) for p in exclude_prefixes)
            ]
            idx = [cols.index(c) for c in use_cols if c in cols]
            target_arr = self.feature_table().column(self.target)
            self._feature_dataset = FeatureDataset(X[:, idx], target_arr, use_cols, self.rows)
        return self._feature_dataset

    def representation_batch(self) -> RepresentationBatch:
        if self._representation_batch is None:
            self._representation_batch = load_representation_batch(
                self.run_id,
                self.store_root,
                target=self.target,
            )
        return self._representation_batch

    def env_for(self, variables: list[str]) -> dict[str, np.ndarray]:
        """Cached column env for expression evaluation (avoids repeated row scans)."""
        key = tuple(variables)
        cached = self._env_cache.get(key)
        if cached is None:
            cached = self.feature_table().env(variables)
            self._env_cache[key] = cached
        return cached

    def target_array(self) -> np.ndarray:
        return self.feature_table().column(self.target)


class StageTimer:
    """Wall-clock timing for discovery stages."""

    def __init__(
        self,
        on_stage: Callable[[str], None] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.timings: dict[str, float] = {}
        self._on_stage = on_stage
        self._on_progress = on_progress
        self._current: str | None = None
        self._started = 0.0
        self._heartbeat: StageHeartbeat | None = None

    def begin(self, name: str) -> None:
        self._end_current()
        self._current = name
        self._started = time.perf_counter()
        if self._on_stage is not None:
            self._on_stage(name)
        self._heartbeat = StageHeartbeat(self._on_progress, label=name)
        self._heartbeat.start()

    def finish(self) -> None:
        self._end_current()

    def _end_current(self) -> None:
        if self._current is not None:
            if self._heartbeat is not None:
                self._heartbeat.stop(detail="complete")
                self._heartbeat = None
            elapsed = time.perf_counter() - self._started
            self.timings[self._current] = self.timings.get(self._current, 0.0) + elapsed
            self._current = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.begin(name)
        try:
            yield
        finally:
            self.finish()
