"""Rank parameterized and derived descriptor candidates."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rde.analyze.ranker import Conjecture, ConjectureRanker, extrapolation_r_squared
from rde.analyze.tables import default_train_test_split, group_indices_by_size, numeric_columns
from rde.descriptor_gen.compute import template_family_executable
from rde.descriptor_gen.materialize import load_row_arrays, materialize_template_column
from rde.descriptor_gen.spec import DescriptorTemplate
from rde.expression import pearson_r, regression_r_squared
from rde.expression.batch import ProgressCallback
from rde.expression.enumerate import enumerate_expressions


@dataclass
class DescriptorCandidate:
    """One ranked descriptor generator candidate."""

    kind: str
    key: str
    target: str
    r_squared: float
    pearson_r: float
    mdl: float
    n_rows: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def level_hint(self) -> int:
        if self.r_squared >= 0.75:
            return 1
        return 0


def descriptor_variable_columns(rows: list[dict[str, Any]], *, target: str) -> list[str]:
    """Numeric descriptor columns suitable for derived-generator search."""
    skip = {
        target,
        "size",
        "seed",
        "family_index",
    }
    out: list[str] = []
    for col in numeric_columns(rows, exclude=skip):
        if col.startswith("metric."):
            continue
        if col.startswith("gen."):
            continue
        out.append(col)
    return out


@dataclass
class DescriptorTemplateRanker:
    """Score parameterized templates against a target using stored slice arrays."""

    target_column: str
    min_abs_r: float = 0.25
    max_results: int = 20

    def rank_templates(
        self,
        rows: list[dict[str, Any]],
        templates: list[DescriptorTemplate],
        *,
        run_id: str,
        store_root: Path | str,
        on_progress: ProgressCallback | None = None,
    ) -> list[DescriptorCandidate]:
        if not rows or not templates:
            return []
        arrays = load_row_arrays(run_id, store_root)
        if not arrays:
            return []

        target = np.array([row.get(self.target_column, float("nan")) for row in rows], dtype=float)
        groups = group_indices_by_size(rows)
        split = default_train_test_split(rows)
        heap: list[tuple[tuple[float, float, float, float], int, DescriptorCandidate]] = []
        tie = 0
        templates = [t for t in templates if template_family_executable(t.family)]
        total = len(templates)
        report_every = max(1, total // 100)
        walsh_cache: dict[int, np.ndarray] = {}
        spectral_cache: dict[int, np.ndarray] = {}

        from rde.runtime.resilience import SOFT_FAIL_EXCEPTIONS

        for i, template in enumerate(templates, start=1):
            try:
                values = materialize_template_column(
                    rows,
                    template,
                    arrays,
                    walsh_cache=walsh_cache,
                    spectral_cache=spectral_cache,
                )
                if not np.any(np.isfinite(values)):
                    continue
                pr = pearson_r(values, target)
                if not np.isfinite(pr) or abs(pr) < self.min_abs_r:
                    continue
                r2 = regression_r_squared(values, target)
                mdl = float(len(template.params) + 1)
                meta = {
                    "template_id": template.template_id,
                    "family": template.family,
                    "params": template.params,
                    "canonical_id": template.canonical_id,
                    "provenance": template.provenance,
                    "cross_n_sign_stability": _values_cross_n_stability(
                        rows, values, target, groups=groups
                    ),
                    "extrapolation_r_squared": extrapolation_r_squared(
                        rows, values, target, split=split
                    ),
                }
                cand = DescriptorCandidate(
                    kind="template",
                    key=template.key,
                    target=self.target_column,
                    r_squared=r2,
                    pearson_r=pr,
                    mdl=mdl,
                    n_rows=len(rows),
                    metadata=meta,
                )
                key = _rank_key(cand)
                tie += 1
                entry = (key, tie, cand)
                if len(heap) < self.max_results:
                    heapq.heappush(heap, entry)
                elif key > heap[0][0]:
                    heapq.heapreplace(heap, entry)
            except SOFT_FAIL_EXCEPTIONS:
                # One bad template must never abort a massive catalog pass.
                continue
            if on_progress is not None and (i % report_every == 0 or i == total):
                on_progress(i, total, f"templates ranked={len(heap)}")

        ranked = [c for _, _, c in heap]
        ranked.sort(key=lambda c: _rank_key(c), reverse=True)
        return ranked


@dataclass
class DerivedDescriptorRanker:
    """Rank expression-derived descriptors over fixed descriptor columns."""

    target_column: str
    min_abs_r: float = 0.25
    max_results: int = 20

    def rank_expressions(
        self,
        rows: list[dict[str, Any]],
        candidates: Iterator[Any],
        *,
        variables: list[str] | None = None,
        max_evaluated: int | None = None,
        eval_backend: Any | None = None,
        on_progress: ProgressCallback | None = None,
        force_gpu: bool = False,
        require_gpu: bool = False,
    ) -> list[DescriptorCandidate]:
        ranker = ConjectureRanker(
            target_column=self.target_column,
            min_abs_r=self.min_abs_r,
            max_results=self.max_results,
        )
        vars_ = variables or descriptor_variable_columns(rows, target=self.target_column)
        conjectures = ranker.rank_expressions_streaming(
            rows,
            candidates,
            variables=vars_,
            max_evaluated=max_evaluated,
            eval_backend=eval_backend,
            on_progress=on_progress,
            force_gpu=force_gpu,
            require_gpu=require_gpu,
        )
        return [_conjecture_to_descriptor(c) for c in conjectures]


def rank_descriptor_generators(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    store_root: Path | str,
    target: str,
    templates: list[DescriptorTemplate],
    max_derived: int = 3000,
    max_derived_depth: int = 2,
    max_derived_vars: int = 16,
    min_abs_r: float = 0.25,
    max_results: int = 20,
    eval_backend: Any | None = None,
    include_templates: bool = True,
    include_derived: bool = True,
    on_progress: ProgressCallback | None = None,
    force_gpu: bool = False,
    require_gpu: bool = False,
) -> list[DescriptorCandidate]:
    """Rank template and derived descriptor generators together."""
    out: list[DescriptorCandidate] = []

    def report(done: int, total: int, detail: str) -> None:
        if on_progress is not None:
            on_progress(done, total, detail)

    if include_templates and templates:
        tmpl_ranker = DescriptorTemplateRanker(
            target_column=target,
            min_abs_r=min_abs_r,
            max_results=max_results,
        )
        out.extend(
            tmpl_ranker.rank_templates(
                rows,
                templates,
                run_id=run_id,
                store_root=store_root,
                on_progress=lambda d, t, detail: report(d, t, f"descriptor templates: {detail}"),
            )
        )
    if include_derived:
        vars_ = descriptor_variable_columns(rows, target=target)[:max_derived_vars]
        if vars_:
            derived_ranker = DerivedDescriptorRanker(
                target_column=target,
                min_abs_r=min_abs_r,
                max_results=max_results,
            )
            cand_iter = enumerate_expressions(
                vars_,
                max_depth=max_derived_depth,
                max_candidates=max_derived,
            )
            out.extend(
                derived_ranker.rank_expressions(
                    rows,
                    cand_iter,
                    variables=vars_,
                    max_evaluated=max_derived,
                    eval_backend=eval_backend,
                    on_progress=lambda d, t, detail: report(d, t, f"descriptor derived: {detail}"),
                    force_gpu=force_gpu,
                    require_gpu=require_gpu,
                )
            )
    out.sort(key=_rank_key, reverse=True)
    return out[:max_results]


def _conjecture_to_descriptor(conj: Conjecture) -> DescriptorCandidate:
    return DescriptorCandidate(
        kind="derived",
        key=f"derived.{conj.expression}",
        target=conj.target,
        r_squared=conj.r_squared,
        pearson_r=conj.pearson_r,
        mdl=conj.mdl,
        n_rows=conj.n_rows,
        metadata=dict(conj.metadata),
    )


def _rank_key(cand: DescriptorCandidate) -> tuple[float, float, float, float]:
    extrap = float(cand.metadata.get("extrapolation_r_squared", float("nan")))
    extrap_key = extrap if np.isfinite(extrap) else -1.0
    cross = float(cand.metadata.get("cross_n_sign_stability", float("nan")))
    cross_key = cross if np.isfinite(cross) else -1.0
    return (extrap_key, cross_key, cand.r_squared, -cand.mdl)


def _values_cross_n_stability(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    target: np.ndarray,
    *,
    groups: dict[Any, np.ndarray] | None = None,
) -> float:
    if len(rows) < 6:
        return float("nan")
    global_r = pearson_r(values, target)
    if not np.isfinite(global_r) or global_r == 0.0:
        return float("nan")
    if groups is None:
        groups = group_indices_by_size(rows)
    if len(groups) < 2:
        return float("nan")
    global_sign = np.sign(global_r)
    signs: list[float] = []
    for idxs in groups.values():
        if len(idxs) < 3:
            continue
        r = pearson_r(values[idxs], target[idxs])
        if np.isfinite(r) and r != 0.0:
            signs.append(float(np.sign(r) == global_sign))
    return float(np.mean(signs)) if signs else float("nan")


def candidates_to_records(candidates: list[DescriptorCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "kind": c.kind,
            "key": c.key,
            "target": c.target,
            "r_squared": c.r_squared,
            "pearson_r": c.pearson_r,
            "mdl": c.mdl,
            "n_rows": c.n_rows,
            "level_hint": c.level_hint,
            **c.metadata,
        }
        for c in candidates
    ]
