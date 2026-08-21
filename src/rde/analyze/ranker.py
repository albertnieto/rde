"""Rank candidate metrics and conjectures (Phase 3–6)."""

from __future__ import annotations

import heapq
import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.analyze.tables import default_train_test_split, group_indices_by_size, numeric_columns
from rde.expression import Expr, pearson_r, predictive_r_squared, regression_r_squared
from rde.expression.batch import EvalBackend, ProgressCallback, evaluate_chunk, normalize_expr_backend, prepare_device_envs


def _expr_to_payload(expr: Expr) -> dict[str, Any]:
    """Serialize an expression AST without relying on its display string."""
    payload: dict[str, Any] = {"kind": expr.kind, "value": expr.value}
    if expr.left is not None:
        payload["left"] = _expr_to_payload(expr.left)
    if expr.right is not None:
        payload["right"] = _expr_to_payload(expr.right)
    return payload


def _expr_from_payload(payload: dict[str, Any]) -> Expr:
    """Restore the exact AST stored in a ranker checkpoint."""
    return Expr(
        kind=payload["kind"],
        value=payload.get("value"),
        left=_expr_from_payload(payload["left"]) if payload.get("left") is not None else None,
        right=_expr_from_payload(payload["right"]) if payload.get("right") is not None else None,
    )


def candidate_universe_fingerprint(
    variables: list[str],
    *,
    max_depth: int,
    max_candidates: int,
    generator_version: str = "ratio-product-unary-expression-v1",
) -> str:
    """Identify the deterministic expression generator used for resume."""
    payload = {
        "generator": "rde.expression.generators.enumerate_metric_candidates",
        "generator_version": generator_version,
        "variables": list(variables),
        "max_depth": int(max_depth),
        "max_candidates": int(max_candidates),
        "include_ratios": True,
        "include_products": True,
        "include_unary_chains": True,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_fingerprint(
    *,
    target_column: str,
    variables: list[str],
    env: dict[str, np.ndarray],
    target: np.ndarray,
    min_abs_r: float,
    max_results: int,
    chunk_size: int,
    max_evaluated: int | None,
    requested_backend: Any,
    compute_metadata: bool,
    candidate_universe_id: str | None,
) -> str:
    """Bind resume state to the exact data and search configuration."""
    digest = hashlib.sha256()
    digest.update(np.asarray(target, dtype=np.float64).tobytes())
    missing = [variable for variable in variables if variable not in env]
    if missing:
        raise ValueError(
            "expression ranker env is missing variables required for checkpointing: "
            f"{missing}. env keys={sorted(env)}; variables={list(variables)}. "
            "When passing a prebuilt env=, variables= must match exactly."
        )
    for variable in variables:
        digest.update(variable.encode("utf-8"))
        digest.update(np.asarray(env[variable], dtype=np.float64).tobytes())
    payload = {
        "target_column": target_column,
        "variables": list(variables),
        "min_abs_r": min_abs_r,
        "max_results": max_results,
        "chunk_size": chunk_size,
        "max_evaluated": max_evaluated,
        "requested_backend": str(requested_backend),
        "compute_metadata": compute_metadata,
        "candidate_universe_id": candidate_universe_id,
        "data_sha256": digest.hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class Conjecture:
    """One ranked candidate metric or formula."""

    expression: str
    target: str
    r_squared: float
    pearson_r: float
    mdl: float
    n_rows: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def level_hint(self) -> int:
        """Rough outcome-level guess from predictive power (see roadmap outcome levels)."""
        if self.r_squared >= 0.95:
            return 1
        if self.r_squared >= 0.75:
            return 1
        if self.r_squared >= 0.5:
            return 0
        return 0


@dataclass
class ConjectureRanker:
    """Score expression candidates against a target column in flat feature rows."""

    target_column: str
    min_abs_r: float = 0.3
    max_results: int = 10

    def _rows_to_env(self, rows: list[dict[str, Any]], variables: list[str]) -> dict[str, np.ndarray]:
        return FeatureTable.from_rows(rows, columns=variables).env(variables)

    def _make_conjecture(
        self,
        expr: Expr,
        r2: float,
        pr: float,
        values: np.ndarray,
        rows: list[dict[str, Any]],
        target: np.ndarray,
        *,
        groups: dict[Any, np.ndarray] | None = None,
        split: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Conjecture:
        mdl = float(expr.size()) + 0.5 * float(expr.depth())
        return Conjecture(
            expression=str(expr),
            target=self.target_column,
            r_squared=r2,
            pearson_r=pr,
            mdl=mdl,
            n_rows=len(rows),
            metadata={
                "cross_n_sign_stability": _expr_cross_n_stability(rows, values, target, groups=groups),
                "extrapolation_r_squared": extrapolation_r_squared(rows, values, target, split=split),
            },
        )

    @staticmethod
    def _rank_key(conjecture: Conjecture) -> tuple[float, float, float, float, float]:
        """Larger = better (for max-heap via min-heap eviction)."""
        extrap = float(conjecture.metadata.get("extrapolation_r_squared", float("nan")))
        extrap_key = min(1.0, extrap) if np.isfinite(extrap) else -1.0
        cross_n = float(conjecture.metadata.get("cross_n_sign_stability", float("nan")))
        cross_key = min(1.0, max(0.0, cross_n)) if np.isfinite(cross_n) else -1.0
        return (extrap_key, cross_key, conjecture.r_squared, -conjecture.mdl, abs(conjecture.pearson_r))

    def _safe_evaluate_chunk(
        self,
        chunk: list[Expr],
        env: dict[str, np.ndarray],
        target: np.ndarray,
        *,
        backend: Any,
        mlx_env: Any,
        torch_env: Any,
        target_mlx: Any,
        force_gpu: bool,
    ) -> list[tuple[Expr, float, float, np.ndarray]]:
        """Evaluate a chunk; on failure score members one-by-one so one bad expr cannot discard peers."""
        from rde.runtime.resilience import SOFT_FAIL_EXCEPTIONS

        try:
            return evaluate_chunk(
                chunk,
                env,
                target,
                min_abs_r=self.min_abs_r,
                backend=backend,
                mlx_env=mlx_env,
                torch_env=torch_env,
                target_mlx=target_mlx,
                force_gpu=force_gpu,
            )
        except SOFT_FAIL_EXCEPTIONS:
            hits: list[tuple[Expr, float, float, np.ndarray]] = []
            for expr in chunk:
                try:
                    one = evaluate_chunk(
                        [expr],
                        env,
                        target,
                        min_abs_r=self.min_abs_r,
                        backend=backend,
                        mlx_env=mlx_env,
                        torch_env=torch_env,
                        target_mlx=target_mlx,
                        force_gpu=force_gpu,
                    )
                except SOFT_FAIL_EXCEPTIONS:
                    continue
                hits.extend(one)
            return hits

    def rank_expressions(
        self,
        rows: list[dict[str, Any]],
        candidates: list[Expr],
        *,
        variables: list[str] | None = None,
    ) -> list[Conjecture]:
        return self.rank_expressions_streaming(rows, iter(candidates), variables=variables)

    @staticmethod
    def _streaming_rank_key(r2: float, pr: float, mdl: float) -> tuple[float, float, float]:
        """Cheap heap key during streaming (metadata computed only for survivors)."""
        return (r2, -mdl, abs(pr))

    @staticmethod
    def _optimistic_metadata_key(
        r2: float,
        pr: float,
        mdl: float,
    ) -> tuple[float, float, float, float, float]:
        """Upper bound on a metadata-aware key before metadata is evaluated."""
        # Cross-size stability is an average of Boolean indicators and is at
        # most 1.  Predictive R² is mathematically at most 1; nextafter keeps
        # the prune conservative against tiny floating-point overshoots.
        return (
            float(np.nextafter(1.0, np.inf)),
            1.0,
            r2,
            -mdl,
            abs(pr),
        )

    def rank_expressions_streaming(
        self,
        rows: list[dict[str, Any]],
        candidates: Iterator[Expr],
        *,
        variables: list[str] | None = None,
        chunk_size: int = 512,
        max_evaluated: int | None = None,
        compute_metadata: bool = True,
        eval_backend: EvalBackend | None = None,
        env: dict[str, np.ndarray] | None = None,
        target: np.ndarray | None = None,
        on_progress: ProgressCallback | None = None,
        force_gpu: bool = False,
        require_gpu: bool = False,
        checkpoint_path: Path | str | None = None,
        resume: bool = False,
        checkpoint_every: int = 10_000,
        candidate_universe_id: str | None = None,
    ) -> list[Conjecture]:
        """Score candidates from an iterator with bounded optional resume state.

        The checkpoint stores only the bounded top-k heap and the number of
        candidates consumed.  The caller's deterministic candidate iterator is
        advanced by that count on resume; no million-expression list is ever
        materialized.
        """
        requested_backend = normalize_expr_backend(eval_backend)
        if not rows:
            return []
        vars_ = (
            numeric_columns(rows, exclude={self.target_column})
            if variables is None
            else list(variables)
        )
        table: FeatureTable | None = None
        if env is None or target is None:
            table = FeatureTable.from_rows(
                rows,
                columns=[*vars_, self.target_column],
            )
        if target is None:
            if table is None:
                raise RuntimeError("target table was not initialized")
            target_arr = table.column(self.target_column)
        else:
            target_arr = target
        if env is None:
            if table is None:
                raise RuntimeError("feature table was not initialized")
            env_dict = table.env(vars_)
        else:
            env_dict = env
        backend, mlx_env, torch_env, target_mlx, fallback_reason = prepare_device_envs(
            env_dict,
            requested_backend,
            len(target_arr),
            target_arr,
            force_gpu=force_gpu,
            require_gpu=require_gpu,
            requested_backend=requested_backend,
        )
        if require_gpu and backend != requested_backend:
            raise RuntimeError(
                f"GPU backend required but effective={backend!r} "
                f"(requested={requested_backend!r}, reason={fallback_reason!r}, n_rows={len(target_arr)})"
            )
        self.last_eval_backend = backend
        self.last_eval_backend_requested = requested_backend
        self.last_eval_fallback_reason = fallback_reason
        checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        groups = group_indices_by_size(rows) if compute_metadata else None
        split = default_train_test_split(rows) if compute_metadata else None
        heap: list[
            tuple[tuple[float, ...], int, Expr, float, float, np.ndarray, Conjecture | None]
        ] = []
        chunk: list[Expr] = []
        n_evaluated = 0
        best_r2 = 0.0
        tie_counter = 0
        checkpoint_identity = _checkpoint_fingerprint(
            target_column=self.target_column,
            variables=vars_,
            env=env_dict,
            target=np.asarray(target_arr, dtype=float),
            min_abs_r=self.min_abs_r,
            max_results=self.max_results,
            chunk_size=chunk_size,
            max_evaluated=max_evaluated,
            requested_backend=requested_backend,
            compute_metadata=compute_metadata,
            candidate_universe_id=candidate_universe_id,
        )
        checkpoint_payload: dict[str, Any] | None = None
        if checkpoint_file is not None and checkpoint_file.exists() and resume:
            try:
                checkpoint_payload = json.loads(
                    checkpoint_file.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid expression checkpoint {checkpoint_file}") from exc
            if checkpoint_payload.get("fingerprint") != checkpoint_identity:
                raise ValueError(
                    f"expression checkpoint configuration mismatch: {checkpoint_file}"
                )
            n_evaluated = int(checkpoint_payload.get("n_evaluated", 0))
            if n_evaluated < 0:
                raise ValueError("expression checkpoint has negative n_evaluated")
            tie_counter = int(checkpoint_payload.get("tie_counter", 0))
            best_r2 = float(checkpoint_payload.get("best_r2", 0.0))
            for record in checkpoint_payload.get("survivors", []):
                expr = _expr_from_payload(dict(record["expr"]))
                preview: Conjecture | None = None
                if compute_metadata:
                    preview = Conjecture(
                        expression=str(expr),
                        target=self.target_column,
                        r_squared=float(record["r_squared"]),
                        pearson_r=float(record["pearson_r"]),
                        mdl=float(record["mdl"]),
                        n_rows=int(record["n_rows"]),
                        metadata=dict(record.get("metadata") or {}),
                    )
                key = tuple(float(value) for value in record["heap_key"])
                entry = (
                    key,
                    int(record["tie"]),
                    expr,
                    float(record["r_squared"]),
                    float(record["pearson_r"]),
                    np.array([], dtype=float),
                    preview,
                )
                heapq.heappush(heap, entry)
            if checkpoint_payload.get("completed"):
                ranked = [
                    preview
                    for *_rest, preview in heap
                    if preview is not None
                ]
                if not compute_metadata:
                    ranked = [
                        Conjecture(
                            expression=str(expr),
                            target=self.target_column,
                            r_squared=r2,
                            pearson_r=pr,
                            mdl=float(expr.size()) + 0.5 * float(expr.depth()),
                            n_rows=len(rows),
                        )
                        for _, _, expr, r2, pr, _values, _preview in heap
                    ]
                ranked.sort(key=self._rank_key, reverse=True)
                return ranked
            candidates = itertools.islice(candidates, n_evaluated, None)
        progress_interval = max(1_000, chunk_size * 4)
        last_progress = 0
        last_progress_t = time.monotonic()

        def consider(expr: Expr, r2: float, pr: float, values: np.ndarray) -> None:
            nonlocal tie_counter, best_r2
            best_r2 = max(best_r2, r2)
            mdl = float(expr.size()) + 0.5 * float(expr.depth())
            preview: Conjecture | None = None
            if compute_metadata:
                if len(heap) >= self.max_results:
                    floor_key = heap[0][0]
                    optimistic_key = self._optimistic_metadata_key(r2, pr, mdl)
                    if floor_key[0] >= 1.0 and floor_key[1] >= 1.0:
                        # Make the common perfect-metadata case explicit:
                        # only the trailing in-sample key can decide.
                        if (r2, -mdl, abs(pr)) <= floor_key[2:]:
                            return
                    elif optimistic_key <= floor_key:
                        return
                # Use the same cross-size/extrapolation-aware key during
                # eviction as at final ordering.  Ranking a streaming heap
                # by in-sample R² first can discard the candidates that have
                # the strongest out-of-sample behavior before metadata is
                # ever computed.
                preview = self._make_conjecture(
                    expr, r2, pr, values, rows, target_arr, groups=groups, split=split
                )
                key = self._rank_key(preview)
            else:
                key = self._streaming_rank_key(r2, pr, mdl)
            tie_counter += 1
            entry = (key, tie_counter, expr, r2, pr, values, preview)
            if len(heap) < self.max_results:
                heapq.heappush(heap, entry)
            elif key > heap[0][0]:
                heapq.heapreplace(heap, entry)

        def maybe_report() -> None:
            nonlocal last_progress, last_progress_t
            if on_progress is None or max_evaluated is None:
                return
            now = time.monotonic()
            if (
                n_evaluated - last_progress >= progress_interval
                or now - last_progress_t >= 5.0
                or n_evaluated >= max_evaluated
            ):
                last_progress = n_evaluated
                last_progress_t = now
                detail = f"backend={backend} best_R2={best_r2:.3f}"
                on_progress(n_evaluated, max_evaluated, detail)

        def persist_checkpoint(*, completed: bool = False) -> None:
            if checkpoint_file is None:
                return
            survivors = []
            for key, tie, expr, r2, pr, _values, preview in heap:
                survivors.append(
                    {
                        "expr": _expr_to_payload(expr),
                        "heap_key": list(key),
                        "tie": tie,
                        "r_squared": r2,
                        "pearson_r": pr,
                        "mdl": (
                            preview.mdl
                            if preview is not None
                            else float(expr.size()) + 0.5 * float(expr.depth())
                        ),
                        "n_rows": len(rows),
                        "metadata": {} if preview is None else preview.metadata,
                    }
                )
            _write_checkpoint(
                checkpoint_file,
                {
                    "schema": "expression-ranker-checkpoint-v1",
                    "fingerprint": checkpoint_identity,
                    "target_column": self.target_column,
                    "requested_backend": requested_backend,
                    "effective_backend": backend,
                    "fallback_reason": fallback_reason,
                    "n_evaluated": n_evaluated,
                    "max_evaluated": max_evaluated,
                    "completed": completed,
                    "best_r2": best_r2,
                    "tie_counter": tie_counter,
                    "survivors": survivors,
                },
            )

        for expr in candidates:
            if max_evaluated is not None and n_evaluated >= max_evaluated:
                break
            chunk.append(expr)
            n_evaluated += 1
            if len(chunk) >= chunk_size:
                for hit in self._safe_evaluate_chunk(
                    chunk,
                    env_dict,
                    target_arr,
                    backend=backend,
                    mlx_env=mlx_env,
                    torch_env=torch_env,
                    target_mlx=target_mlx,
                    force_gpu=force_gpu,
                ):
                    consider(hit[0], hit[1], hit[2], hit[3])
                chunk.clear()
                maybe_report()
                if checkpoint_file is not None and (
                    n_evaluated % checkpoint_every == 0
                    or n_evaluated >= (max_evaluated or n_evaluated)
                ):
                    persist_checkpoint()

        if chunk:
            for hit in self._safe_evaluate_chunk(
                chunk,
                env_dict,
                target_arr,
                backend=backend,
                mlx_env=mlx_env,
                torch_env=torch_env,
                target_mlx=target_mlx,
                force_gpu=force_gpu,
            ):
                consider(hit[0], hit[1], hit[2], hit[3])
            maybe_report()
            persist_checkpoint()

        if on_progress is not None and max_evaluated is not None:
            on_progress(n_evaluated, max_evaluated, f"backend={backend} done best_R2={best_r2:.3f}")
        persist_checkpoint(completed=True)

        survivors = [
            (expr, r2, pr, values, preview)
            for _, _, expr, r2, pr, values, preview in heap
        ]
        if compute_metadata:
            ranked = [preview for *_rest, preview in survivors if preview is not None]
        else:
            ranked = [
                Conjecture(
                    expression=str(expr),
                    target=self.target_column,
                    r_squared=r2,
                    pearson_r=pr,
                    mdl=float(expr.size()) + 0.5 * float(expr.depth()),
                    n_rows=len(rows),
                )
                for expr, r2, pr, _values, _preview in survivors
            ]
        ranked.sort(key=self._rank_key, reverse=True)
        return ranked

    def to_records(self, conjectures: list[Conjecture]) -> list[dict[str, Any]]:
        return [
            {
                "expression": c.expression,
                "target": c.target,
                "r_squared": c.r_squared,
                "pearson_r": c.pearson_r,
                "mdl": c.mdl,
                "n_rows": c.n_rows,
                "level_hint": c.level_hint,
                **c.metadata,
            }
            for c in conjectures
        ]


def extrapolation_r_squared(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    target: np.ndarray,
    *,
    train_sizes: set[int] | None = None,
    test_sizes: set[int] | None = None,
    split: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Fit linear map on train sizes, score regression R² on held-out sizes.

    ``split`` lets callers pass a precomputed ``default_train_test_split(rows)``
    when scoring many candidates against the same ``rows`` in one ranking
    pass, instead of rescanning ``rows`` per candidate.
    """
    if split is not None and train_sizes is None and test_sizes is None:
        train_idx, test_idx = split
    else:
        all_sizes = sorted({int(row["size"]) for row in rows if row.get("size") is not None})
        if len(all_sizes) < 2:
            return float("nan")
        if train_sizes is None or test_sizes is None:
            n_split = max(1, len(all_sizes) // 2)
            train_sizes = set(all_sizes[:n_split])
            test_sizes = set(all_sizes[n_split:])
        train_idx = [i for i, row in enumerate(rows) if row.get("size") in train_sizes]
        test_idx = [i for i, row in enumerate(rows) if row.get("size") in test_sizes]
    if len(train_idx) < 3 or len(test_idx) < 3:
        return float("nan")
    xv = values[train_idx]
    yv = target[train_idx]
    # A derived expression (ratio/product of otherwise well-behaved columns)
    # can still be non-finite for individual rows even when neither base
    # column is -- e.g. a near-zero denominator. np.polyfit's internal SVD
    # does not degrade gracefully on non-finite input (raises LinAlgError
    # rather than returning); guard it the same way the zero-variance case
    # just below already is, rather than letting a single degenerate
    # candidate crash the whole ranking pass.
    if not (np.all(np.isfinite(xv)) and np.all(np.isfinite(yv))):
        return float("nan")
    if float(np.std(xv)) == 0.0:
        return float("nan")
    coeffs = np.polyfit(xv, yv, 1)
    pred = np.polyval(coeffs, values[test_idx])
    return predictive_r_squared(pred, target[test_idx])


def _expr_cross_n_stability(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    target: np.ndarray,
    *,
    groups: dict[Any, np.ndarray] | None = None,
) -> float:
    """Cross-size sign stability for an expression's evaluated values.

    ``groups`` lets callers pass a precomputed ``group_indices_by_size(rows)``
    when scoring many candidates against the same ``rows`` in one ranking
    pass, instead of rescanning ``rows`` per candidate.
    """
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
