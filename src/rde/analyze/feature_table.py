"""Columnar in-memory feature table (avoids repeated dict-of-rows scans)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rde.analyze.tables import group_indices_by_size, numeric_columns
from rde.features.numeric import safe_pearson_r


@dataclass
class FeatureTable:
    """Numeric columns stored as a single matrix with name indexing."""

    columns: list[str]
    data: np.ndarray
    rows: list[dict[str, Any]]
    _index: dict[str, int] = field(default_factory=dict, repr=False)
    _correlate_cache: dict[tuple[str, float], list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._index = {name: i for i, name in enumerate(self.columns)}

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        columns: list[str] | None = None,
        target: str | None = None,
    ) -> FeatureTable:
        cols = list(columns or numeric_columns(rows, exclude=set()))
        if target and target not in cols:
            cols.append(target)
        if not rows:
            return cls(columns=cols, data=np.zeros((0, len(cols))), rows=[])
        if not cols:
            return cls(columns=[], data=np.zeros((len(rows), 0)), rows=rows)
        data = np.column_stack(
            [
                np.asarray(
                    [row.get(column, float("nan")) for row in rows],
                    dtype=float,
                )
                for column in cols
            ]
        )
        return cls(columns=cols, data=data, rows=rows)

    def __len__(self) -> int:
        return self.data.shape[0]

    def take(self, indices: np.ndarray | list[int]) -> FeatureTable:
        """Return a row subset without rematerializing numeric columns."""
        selected = np.asarray(indices)
        if selected.dtype == bool:
            if selected.ndim != 1 or selected.size != len(self):
                raise ValueError("boolean indices must match the table row count")
            selected = np.flatnonzero(selected)
        selected = selected.astype(int, copy=False).reshape(-1)
        return FeatureTable(
            columns=list(self.columns),
            data=self.data[selected],
            rows=[self.rows[int(index)] for index in selected],
        )

    def has_column(self, name: str) -> bool:
        return name in self._index

    def column(self, name: str) -> np.ndarray:
        if name not in self._index:
            return np.full(len(self), float("nan"), dtype=float)
        return self.data[:, self._index[name]]

    def env(self, variables: list[str]) -> dict[str, np.ndarray]:
        return {var: self.column(var) for var in variables}

    def correlate_with_target(
        self,
        target: str,
        *,
        min_abs_r: float = 0.3,
        exclude: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        cache_key = (target, min_abs_r)
        if cache_key in self._correlate_cache:
            return list(self._correlate_cache[cache_key])

        from rde.analyze.query import cross_n_sign_stability
        from rde.expression.evaluate import pearson_r, regression_r_squared

        skip = (exclude or set()) | {target}
        y = self.column(target)
        if np.isfinite(y).sum() < 3:
            return []

        groups = group_indices_by_size(self.rows)
        results: list[dict[str, Any]] = []
        for col in self.columns:
            if col in skip:
                continue
            x = self.column(col)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            xv, yv = x[mask], y[mask]
            if float(np.std(xv)) == 0.0 or float(np.std(yv)) == 0.0:
                continue
            r = safe_pearson_r(xv, yv)
            if abs(r) >= min_abs_r:
                results.append(
                    {
                        "column": col,
                        "pearson_r": r,
                        "r_squared": regression_r_squared(x, y),
                        "cross_n_sign_stability": cross_n_sign_stability(
                            self.rows,
                            col,
                            target,
                            groups=groups,
                            x=x,
                            y=y,
                        ),
                        "n": int(mask.sum()),
                    }
                )
        results.sort(key=lambda d: -abs(d["pearson_r"]))
        self._correlate_cache[cache_key] = results
        return list(results)
