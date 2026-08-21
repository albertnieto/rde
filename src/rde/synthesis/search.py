"""Backward (target-first) algorithm synthesis search (Mode 2).

Forward RDE (Phases 0-6) asks "what structure is hiding in data already
generated for a fixed object." This module asks the complementary question:
given a declared resource budget and a domain that exposes structural
decomposition hooks (``SynthesisDomain``), which algorithm skeleton, if any,
both (a) solves the recurrence within budget and (b) is actually realizable
and correct on the domain — verified against the domain's own brute-force
oracle, never assumed.

Two-stage pruning keeps this cheap and honest:

1. **Symbolic stage** (``rde.synthesis.recurrence``) — solve each candidate
   skeleton's recurrence and reject anything that does not meet the target
   *before* touching the domain at all. This is "the complexity engine
   before quantum circuits" from the reverse-engineering brief.
2. **Verification stage** — for surviving skeletons, actually execute the
   decomposition through the domain and cross-check the resulting cost
   against ``domain.brute_force`` on real instances. A skeleton is accepted
   only if it is both cheap (stage 1) and correct (stage 2) — this is what
   keeps the search from ever "discovering" a skeleton that only works
   because it was allowed to look at the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rde.core.protocols import SynthesisDomain
from rde.synthesis.recurrence import meets_target, solve_recurrence
from rde.synthesis.skeleton import AlgorithmSkeleton, default_skeleton_catalog


def execute_skeleton(
    domain: SynthesisDomain,
    instance: Any,
    skeleton: AlgorithmSkeleton,
    *,
    base_threshold: int = 1,
    max_depth: int = 20,
) -> Any | None:
    """Run `skeleton` against `instance` through the domain's own hooks.

    Returns None if the domain does not support this skeleton's
    decomposition on this instance (unsupported, not incorrect).
    """
    rec = skeleton.recurrence
    if rec.shape == "subtract":
        # No SynthesisDomain hook for size-minus-constant decomposition yet
        # (see ALGO card Notes) — symbolic pruning still applies to these,
        # execution/verification does not.
        return None
    return _execute(domain, instance, rec, base_threshold=base_threshold, max_depth=max_depth)


def _execute(domain: SynthesisDomain, instance: Any, rec, *, base_threshold: int, max_depth: int) -> Any | None:
    n = domain.size_of(instance)
    if rec.shape == "base" or n <= base_threshold:
        return domain.brute_force(instance)
    if max_depth <= 0:
        return None
    if rec.shape == "flat":
        subs = domain.decompose_flat(instance)
        if not subs:
            return None
        return domain.combine(instance, [domain.brute_force(s) for s in subs])
    # divide
    subs = domain.decompose_divide(instance, rec.branches)
    if not subs:
        return None
    sub_solutions = []
    for sub in subs:
        solved = _execute(domain, sub, rec, base_threshold=base_threshold, max_depth=max_depth - 1)
        if solved is None:
            return None
        sub_solutions.append(solved)
    return domain.combine(instance, sub_solutions)


@dataclass(frozen=True)
class VerifyResult:
    n_checked: int
    n_matched: int
    n_unsupported: int
    max_relative_error: float


def verify_skeleton(
    domain: SynthesisDomain,
    instances: list[Any],
    skeleton: AlgorithmSkeleton,
    *,
    base_threshold: int = 1,
    tol: float = 1e-6,
    max_depth: int = 20,
    truths: list[Any] | None = None,
) -> VerifyResult:
    """Cross-check a skeleton's cost against `domain.brute_force` ground truth.

    `truths`, if given, must be `domain.brute_force(instances[i])` precomputed
    once per instance (see `synthesize`, which does exactly this). It exists
    because `brute_force` is exponential by design — it is the whole-instance
    reference oracle a decomposition is trying to avoid — so recomputing it
    once per (skeleton, instance) pair across a whole catalog, instead of
    once per instance total, turns an already-expensive call into the
    dominant cost of the entire search.
    """
    checked = matched = unsupported = 0
    max_err = 0.0
    for i, instance in enumerate(instances):
        candidate = execute_skeleton(domain, instance, skeleton, base_threshold=base_threshold, max_depth=max_depth)
        if candidate is None:
            unsupported += 1
            continue
        checked += 1
        truth = truths[i] if truths is not None else domain.brute_force(instance)
        true_cost = domain.cost(instance, truth)
        cand_cost = domain.cost(instance, candidate)
        denom = abs(true_cost) if abs(true_cost) > 1e-12 else 1.0
        rel_err = abs(cand_cost - true_cost) / denom
        max_err = max(max_err, rel_err)
        if rel_err <= tol:
            matched += 1
    return VerifyResult(n_checked=checked, n_matched=matched, n_unsupported=unsupported, max_relative_error=max_err)


@dataclass
class SynthesisCandidate:
    name: str
    verbs: list[str]
    recurrence_shape: str
    recurrence_description: str
    cost_class: str
    degree: float | None
    status: str  # accepted | rejected_complexity | rejected_unsupported | rejected_incorrect
    detail: str
    n_checked: int = 0
    n_matched: int = 0
    n_unsupported: int = 0
    max_relative_error: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verbs": self.verbs,
            "recurrence_shape": self.recurrence_shape,
            "recurrence": self.recurrence_description,
            "cost_class": self.cost_class,
            "degree": self.degree,
            "status": self.status,
            "detail": self.detail,
            "n_checked": self.n_checked,
            "n_matched": self.n_matched,
            "n_unsupported": self.n_unsupported,
            "max_relative_error": self.max_relative_error,
        }


@dataclass
class SynthesisReport:
    domain_id: str
    target_degree: float | None
    n_instances: int
    candidates: list[SynthesisCandidate] = field(default_factory=list)

    @property
    def accepted(self) -> list[SynthesisCandidate]:
        return [c for c in self.candidates if c.status == "accepted"]

    def best(self) -> SynthesisCandidate | None:
        accepted = self.accepted
        if not accepted:
            return None
        return min(accepted, key=lambda c: c.degree if c.degree is not None else float("inf"))

    def summary(self) -> dict[str, Any]:
        best = self.best()
        return {
            "domain_id": self.domain_id,
            "target_degree": self.target_degree,
            "n_instances": self.n_instances,
            "n_candidates": len(self.candidates),
            "n_accepted": len(self.accepted),
            "n_rejected_complexity": sum(1 for c in self.candidates if c.status == "rejected_complexity"),
            "n_rejected_unsupported": sum(1 for c in self.candidates if c.status == "rejected_unsupported"),
            "n_rejected_incorrect": sum(1 for c in self.candidates if c.status == "rejected_incorrect"),
            "best": best.name if best else None,
            "best_cost_class": best.cost_class if best else None,
        }


def synthesize(
    domain: SynthesisDomain,
    instances: list[Any],
    *,
    target_degree: float | None = None,
    catalog: list[AlgorithmSkeleton] | None = None,
    base_threshold: int = 1,
    tol: float = 1e-6,
    base_exponent: float = 1.0,
    max_depth: int = 20,
    on_progress: Callable[[str], None] | None = None,
) -> SynthesisReport:
    """Search backward from a target resource budget to a verified skeleton.

    `target_degree=None` means "any polynomial"; a numeric value additionally
    caps the polynomial degree (e.g. `target_degree=2` rejects an accepted-
    but-cubic skeleton in favor of a quadratic or linear one, if any exists).

    `on_progress`, if given, is called once per unit of real completed work:
    once per instance while precomputing brute-force ground truth (the one
    step that is exponential by design and can run long for a careless
    `--size`), then once per skeleton evaluated. This is the CLAUDE.md-
    mandated live-progress hook for any non-trivial RDE run — the caller
    (e.g. `cmd_synthesize`) wires it to `RdeConsole.task_progress`, which
    already gives both a live TTY bar and durable `--log-progress` lines for
    free, so this function itself stays console-agnostic.
    """
    if catalog is None:
        catalog = default_skeleton_catalog(base_exponent=base_exponent)

    # Computed lazily, once, on first skeleton that actually needs domain
    # verification -- if every skeleton is pruned symbolically (or the
    # catalog is empty), the exponential brute_force ground truth is never
    # touched at all.
    truths: list[Any] | None = None

    def ensure_truths() -> list[Any]:
        nonlocal truths
        if truths is None:
            computed: list[Any] = []
            for i, instance in enumerate(instances):
                computed.append(domain.brute_force(instance))
                if on_progress is not None:
                    on_progress(f"ground truth {i + 1}/{len(instances)} (brute force, exponential by design)")
            truths = computed
        return truths

    candidates: list[SynthesisCandidate] = []
    for skeleton in catalog:
        solved = solve_recurrence(skeleton.recurrence)
        if not meets_target(solved, target_degree):
            candidates.append(
                SynthesisCandidate(
                    name=skeleton.name,
                    verbs=list(skeleton.verbs),
                    recurrence_shape=skeleton.recurrence.shape,
                    recurrence_description=skeleton.recurrence.describe(),
                    cost_class=solved.pretty(),
                    degree=solved.degree,
                    status="rejected_complexity",
                    detail=f"{solved.pretty()} does not meet the declared budget — rejected before any domain call",
                )
            )
            if on_progress is not None:
                on_progress(f"skeleton {skeleton.name}: rejected_complexity")
            continue

        if skeleton.recurrence.shape == "subtract":
            candidates.append(
                SynthesisCandidate(
                    name=skeleton.name,
                    verbs=list(skeleton.verbs),
                    recurrence_shape=skeleton.recurrence.shape,
                    recurrence_description=skeleton.recurrence.describe(),
                    cost_class=solved.pretty(),
                    degree=solved.degree,
                    status="rejected_unsupported",
                    detail="meets budget symbolically, but no SynthesisDomain hook exists yet for "
                    "size-minus-constant decomposition — not executed or verified",
                )
            )
            if on_progress is not None:
                on_progress(f"skeleton {skeleton.name}: rejected_unsupported")
            continue

        verify = verify_skeleton(
            domain,
            instances,
            skeleton,
            base_threshold=base_threshold,
            tol=tol,
            max_depth=max_depth,
            truths=ensure_truths(),
        )
        if verify.n_checked == 0:
            status, detail = (
                "rejected_unsupported",
                "domain does not support this skeleton's decomposition on the given instances",
            )
        elif verify.n_matched == verify.n_checked:
            status, detail = (
                "accepted",
                f"verified against brute-force ground truth on {verify.n_matched}/{verify.n_checked} instances",
            )
        else:
            status, detail = (
                "rejected_incorrect",
                f"cost mismatch on {verify.n_checked - verify.n_matched}/{verify.n_checked} instances "
                f"(max relative error {verify.max_relative_error:.3g})",
            )
        candidates.append(
            SynthesisCandidate(
                name=skeleton.name,
                verbs=list(skeleton.verbs),
                recurrence_shape=skeleton.recurrence.shape,
                recurrence_description=skeleton.recurrence.describe(),
                cost_class=solved.pretty(),
                degree=solved.degree,
                status=status,
                detail=detail,
                n_checked=verify.n_checked,
                n_matched=verify.n_matched,
                n_unsupported=verify.n_unsupported,
                max_relative_error=verify.max_relative_error,
            )
        )
        if on_progress is not None:
            on_progress(f"skeleton {skeleton.name}: {status}")

    return SynthesisReport(
        domain_id=getattr(domain, "domain_id", "unknown"),
        target_degree=target_degree,
        n_instances=len(instances),
        candidates=candidates,
    )


def write_synthesis_conjectures_jsonl(report: SynthesisReport, path: str | Path) -> None:
    """Persist one row per candidate skeleton, mirroring conjectures.jsonl /
    lower_bound_conjectures.jsonl (same discovery/ hand-off contract)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for candidate in report.candidates:
            row = {
                "domain_id": report.domain_id,
                "target_degree": report.target_degree,
                "n_instances": report.n_instances,
                **candidate.to_dict(),
            }
            fh.write(json.dumps(row) + "\n")
