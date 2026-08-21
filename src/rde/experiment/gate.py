"""Mechanical gate for RDE experiments.

This module exists because documentation does not stop shortcuts. Every
`experiments/EXP-NNN_*/run.py` that claims a scientific verdict must drive an
`ExperimentGate`, which **raises** rather than advises when the run does not
meet the design bar in `rde/docs/experiment-playbook.md`.

The gate enforces, in code:

1. A `DomainContract` exists for the domain.
2. A `PREREGISTRATION.md` exists, and its hash is recorded before results.
3. The plan spans enough sizes and instances to have statistical power.
4. The realized population is genuinely varied — measured by counting
   **distinct structural feature vectors per size**, which is the one check a
   "one landscape reused across a categorical grid" run cannot pass and cannot
   be gamed by relabeling instances.
5. Held-out generator families exist and actually have rows.
6. The mechanism under study is numerically visible to the predictors, not
   present only as a categorical `generator` label.
7. The leak audit ran: raw rows reconstruct the target, cleaned rows do not.
8. The required discovery phases actually executed.
9. Every criterion the pre-registered decision rule reads actually computed —
   a verdict may not rest on a `NaN`.

Only after all of that will `finalize()` write a `receipt.json`. A CI test
(`tests/rde/test_experiment_receipts.py`) fails the suite when an experiment
ships a `results.md` without a valid receipt, so skipping the gate is visible
as a red test rather than as a self-report.

Receipts account for **work done** (instances, distinct instances, candidates
scored, phases run) — never wall-clock — so neither a fast run nor a padded
one can be mistaken for rigor.

Receipt versions are not cosmetic. A `receipt_version` 2 receipt asserts that
checks 6 and 9 ran, which a version 1 receipt cannot claim: those runs were
gated before the checks existed. Old receipts therefore fail validation by
design, and the correct response is to re-run or retract the affected
experiment — never to relax the validator.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

RECEIPT_FILENAME = "receipt.json"
RECEIPT_VERSION = 2

# Prefixes that count as input-derived predictors. `operator.*` matters as much
# as `matrix.*`: if the mechanism under study reaches discovery only as a
# categorical label, no regressor or symbolic search can express a law about it,
# and a null verdict is guaranteed before any search quality question arises.
STRUCTURAL_PREFIXES: tuple[str, ...] = (
    "matrix.",
    "graph.",
    "operator.",
    "align.",
    "optimum.",
    "landscape.",
    "hsp_sample.",
    "spectral.",
    "fourier.",
)

# Phases that constitute "the discovery loop actually ran". An experiment may
# narrow this only by passing `required_phases` explicitly *and* recording the
# justification in its pre-registration.
DEFAULT_REQUIRED_PHASES: tuple[str, ...] = (
    "population",
    "expression_ranker",
    "latent",
    "symbolic",
    "phase6",
    "gated_outcome",
)

RECOVERY_REQUIRED_PHASES: tuple[str, ...] = (
    "population",
    "protocol_search",
    "pipeline_check",
    "heldout_confirm",
    "gated_outcome",
)


class ExperimentPreflightError(RuntimeError):
    """Raised when an experiment does not meet the RDE design bar."""


@dataclass
class PhaseRecord:
    name: str
    work_units: int
    work_kind: str
    detail: dict[str, Any] = field(default_factory=dict)
    soft_failed: bool = False


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def structural_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Structural (input-derived) predictor columns present in the rows."""
    cols: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith(STRUCTURAL_PREFIXES):
                cols.add(key)
    return sorted(cols)


def mechanism_visibility(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    label_key: str = "generator",
    pairing_key: str | None = None,
) -> dict[str, Any]:
    """Is the mechanism under study numerically represented in the predictors?

    A categorical ``generator`` label is not a predictor. If every row sharing
    an instance has the same structural feature vector regardless of its
    mechanism, then no function of the predictors can distinguish mechanisms and
    a null result is an artifact of the feature set, not evidence about nature.

    With ``pairing_key`` (a crossed design where one instance is evaluated under
    every mechanism) this is exact: inside a block the instance-derived columns
    are constant by construction, so any variation must come from mechanism
    features. Without pairing it falls back to a global between-group check,
    which is weaker because instance noise can masquerade as separation.
    """
    blocks: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get(pairing_key) if pairing_key else "__all__"
        blocks.setdefault(key, []).append(row)

    informative = 0
    considered = 0
    for block_rows in blocks.values():
        labels = {str(r.get(label_key, "")) for r in block_rows}
        if len(labels) < 2:
            continue
        considered += 1
        vectors = {
            tuple(_finite(r.get(c)) for c in columns) for r in block_rows
        }
        if len(vectors) > 1:
            informative += 1

    return {
        "blocks_considered": considered,
        "blocks_with_mechanism_variation": informative,
        "paired": bool(pairing_key),
        "visible": considered > 0 and informative == considered,
    }


def vacuous_criteria(criteria: dict[str, Any]) -> list[str]:
    """Criteria that could not be evaluated (None / NaN / non-finite).

    A verdict must never rest on a criterion that did not compute. A `NaN`
    separation score means "not measured", and reading it as "no effect" turns a
    broken measurement into a scientific conclusion.
    """
    bad: list[str] = []
    for name, value in (criteria or {}).items():
        if isinstance(value, bool) or isinstance(value, str):
            continue
        if value is None:
            bad.append(name)
        elif isinstance(value, (int, float)) and _finite(value) is None:
            bad.append(name)
    return sorted(bad)


def distinct_structural_instances(
    rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> int:
    """Count distinct structural feature vectors.

    This is the population-variety measure. Reusing one problem instance across
    a categorical grid collapses every row onto the same vector, so the count
    drops to 1 no matter how the instances are labelled or seeded.
    """
    seen: set[tuple] = set()
    for row in rows:
        seen.add(tuple(_finite(row.get(c)) for c in columns))
    return len(seen)


class ExperimentGate:
    """Enforces the RDE experiment design bar; raises on violation."""

    def __init__(
        self,
        *,
        experiment_dir: Path | str,
        domain_id: str,
        target: str,
        min_sizes: int = 3,
        min_instances_per_size: int = 50,
        min_generator_groups: int = 2,
        min_distinct_fraction: float = 0.5,
        require_held_out: bool = True,
        require_mechanism_visibility: bool = True,
        mechanism_pairing_key: str | None = None,
        required_phases: Sequence[str] = DEFAULT_REQUIRED_PHASES,
        receipt_filename: str | None = None,
        gate_kind: str = "discovery",
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.domain_id = domain_id
        self.target = target
        self.min_sizes = int(min_sizes)
        self.min_instances_per_size = int(min_instances_per_size)
        self.min_generator_groups = int(min_generator_groups)
        self.min_distinct_fraction = float(min_distinct_fraction)
        self.require_held_out = bool(require_held_out)
        self.require_mechanism_visibility = bool(require_mechanism_visibility)
        self.mechanism_pairing_key = mechanism_pairing_key
        self.gate_kind = str(gate_kind)
        if self.gate_kind == "recovery" and tuple(required_phases) == tuple(DEFAULT_REQUIRED_PHASES):
            required_phases = RECOVERY_REQUIRED_PHASES
        self.required_phases = tuple(required_phases)
        self.receipt_filename = str(receipt_filename or RECEIPT_FILENAME)

        self._phases: dict[str, PhaseRecord] = {}
        self._prereg_hash: str | None = None
        self._contract_id: str | None = None
        self._population: dict[str, Any] = {}
        self._leak_audit: dict[str, Any] = {}
        self._plan: dict[str, Any] = {}

    # -- phase 1: before anything runs -----------------------------------

    def check_plan(self, *, sizes: Sequence[int], n_per_size: int) -> None:
        """Contract + pre-registration + statistical-power plan. Raises."""
        from rde.core.domain_contract import domain_contract

        try:
            contract = domain_contract(self.domain_id)
        except KeyError as exc:
            raise ExperimentPreflightError(
                f"no DomainContract for domain {self.domain_id!r}. An experiment "
                "without a contract cannot declare a target, predictors, or "
                "held-out groups. Add one in rde/core/domain_contract.py."
            ) from exc
        self._contract_id = contract.domain_id

        if contract.primary_target != self.target:
            # Not fatal, but must be deliberate and recorded.
            self._plan["target_differs_from_contract"] = contract.primary_target

        prereg = self.experiment_dir / "PREREGISTRATION.md"
        if not prereg.is_file():
            raise ExperimentPreflightError(
                f"missing {prereg}. The decision rule must be written down "
                "before the run, not chosen after seeing the numbers."
            )
        self._prereg_hash = _sha256_file(prereg)

        uniq_sizes = sorted({int(s) for s in sizes})
        if len(uniq_sizes) < self.min_sizes:
            raise ExperimentPreflightError(
                f"need >= {self.min_sizes} distinct sizes for cross-N checks; "
                f"got {uniq_sizes}."
            )
        if int(n_per_size) < self.min_instances_per_size:
            raise ExperimentPreflightError(
                f"need >= {self.min_instances_per_size} instances per size for "
                f"statistical power; got {n_per_size}."
            )

        if self.require_held_out and not contract.held_out_generator_groups:
            raise ExperimentPreflightError(
                f"contract {self.domain_id!r} declares no held_out_generator_groups; "
                "without a held-out family there is no generalization test."
            )

        self._plan.update(
            {
                "sizes": uniq_sizes,
                "n_per_size": int(n_per_size),
                "held_out_generator_groups": list(contract.held_out_generator_groups),
            }
        )

    # -- phase 2: after the population is materialized --------------------

    def check_population(self, rows: Sequence[dict[str, Any]]) -> None:
        """Population variety, size coverage, generator groups, held-out rows."""
        if not rows:
            raise ExperimentPreflightError("population is empty.")

        cols = structural_columns(rows)
        if not cols:
            raise ExperimentPreflightError(
                "no structural predictor columns (matrix.*/graph.*) in the feature "
                "rows. The domain must expose its input primitive (e.g. Q) from "
                "primitive_features so instance descriptors are computed."
            )

        by_size: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            size = row.get("size")
            if size is None:
                continue
            by_size.setdefault(int(size), []).append(row)

        if len(by_size) < self.min_sizes:
            raise ExperimentPreflightError(
                f"population spans {sorted(by_size)} sizes; need >= {self.min_sizes}."
            )

        per_size_stats: dict[str, dict[str, int]] = {}
        for size, size_rows in sorted(by_size.items()):
            distinct = distinct_structural_instances(size_rows, cols)
            needed = max(
                2, int(self.min_distinct_fraction * len(size_rows))
            )
            per_size_stats[str(size)] = {
                "rows": len(size_rows),
                "distinct_structural_instances": distinct,
                "required_distinct": needed,
            }
            if distinct < needed:
                raise ExperimentPreflightError(
                    f"size {size}: only {distinct} distinct structural instances "
                    f"across {len(size_rows)} rows (need >= {needed}). This is the "
                    "signature of reusing one problem instance across a categorical "
                    "grid. Generate an independent random instance per row."
                )

        groups = sorted({str(r.get("generator", "")) for r in rows if r.get("generator")})
        if len(groups) < self.min_generator_groups:
            raise ExperimentPreflightError(
                f"only {len(groups)} generator group(s) {groups}; need >= "
                f"{self.min_generator_groups} for separation / held-out tests."
            )

        held_out = list(self._plan.get("held_out_generator_groups") or [])
        n_held_out_rows = sum(
            1
            for r in rows
            if any(g and (str(r.get("generator", "")).startswith(g)) for g in held_out)
        )
        if self.require_held_out and n_held_out_rows == 0:
            raise ExperimentPreflightError(
                f"no rows belong to the held-out families {held_out}; the "
                "generalization test would be vacuous."
            )

        visibility = mechanism_visibility(
            rows, cols, pairing_key=self.mechanism_pairing_key
        )
        if self.require_mechanism_visibility and not visibility["visible"]:
            raise ExperimentPreflightError(
                "the mechanism under study is not numerically visible to the "
                f"predictors ({visibility}). Every row sharing an instance has the "
                "same structural feature vector regardless of its generator, so no "
                "function of the predictors can distinguish mechanisms and a null "
                "verdict is an artifact of the feature set. Expose operator-level "
                "descriptors (e.g. operator.*) from primitive_features."
            )

        self._population = {
            "n_rows": len(rows),
            "n_structural_columns": len(cols),
            "generator_groups": groups,
            "per_size": per_size_stats,
            "n_held_out_rows": n_held_out_rows,
            "mechanism_visibility": visibility,
        }
        self.record_phase(
            "population", work_units=len(rows), work_kind="feature_rows"
        )

    def check_recovery_population(self, records: Sequence[dict[str, Any]]) -> None:
        """Population check for Mode 2 recovery campaigns (no descriptor table)."""
        if not records:
            raise ExperimentPreflightError("recovery population is empty.")
        by_size: dict[int, list[dict[str, Any]]] = {}
        for row in records:
            by_size.setdefault(int(row["size"]), []).append(row)
        if len(by_size) < self.min_sizes:
            raise ExperimentPreflightError(
                f"recovery population spans {sorted(by_size)} sizes; need >= {self.min_sizes}."
            )
        per_size_stats: dict[str, dict[str, int]] = {}
        for size, size_rows in sorted(by_size.items()):
            distinct = len({(str(r.get("family") or r.get("generator")), int(r["seed"])) for r in size_rows})
            needed = max(2, int(self.min_distinct_fraction * len(size_rows)))
            per_size_stats[str(size)] = {
                "rows": len(size_rows),
                "distinct_structural_instances": distinct,
                "required_distinct": needed,
            }
            if distinct < needed:
                raise ExperimentPreflightError(
                    f"size {size}: only {distinct} distinct recovery instances "
                    f"across {len(size_rows)} rows (need >= {needed})."
                )
        groups = sorted({str(r.get("family") or r.get("generator") or "") for r in records})
        if len(groups) < self.min_generator_groups:
            raise ExperimentPreflightError(
                f"only {len(groups)} family group(s) {groups}; need >= {self.min_generator_groups}."
            )
        held_out = list(self._plan.get("held_out_generator_groups") or [])
        n_held_out_rows = sum(
            1
            for r in records
            if any(g and str(r.get("family") or r.get("generator") or "").startswith(g) for g in held_out)
        )
        if self.require_held_out and n_held_out_rows == 0:
            raise ExperimentPreflightError(
                f"no recovery rows belong to the held-out families {held_out}."
            )
        self._population = {
            "n_rows": len(records),
            "n_structural_columns": 0,
            "generator_groups": groups,
            "per_size": per_size_stats,
            "n_held_out_rows": n_held_out_rows,
            "mechanism_visibility": {
                "visible": True,
                "paired": True,
                "reason": "recovery_exact_K",
            },
        }
        self.record_phase("population", work_units=len(records), work_kind="recovery_instances")

    def check_extractor_isolation(
        self, *, extract_sees_planted: bool, extract_sees_family: bool
    ) -> None:
        """Recovery leak analog: extractors must not see planted K or family."""
        if extract_sees_planted or extract_sees_family:
            raise ExperimentPreflightError(
                "extractor isolation failed: protocol.extract saw planted K or "
                f"the family label (planted={extract_sees_planted}, family={extract_sees_family})."
            )
        self._leak_audit = {
            "kind": "extractor_blindness",
            "extract_sees_planted": False,
            "extract_sees_family": False,
        }

    # -- phase 3: leak audit ----------------------------------------------

    def check_leak_audit(self, *, raw_best_abs_r: float, clean_best_abs_r: float) -> None:
        """Raw rows must leak the target; cleaned rows must not.

        If the raw rows do *not* leak, the caller is almost certainly measuring
        the wrong thing (outcome columns absent), and if the cleaned rows still
        leak, the predictors are target-derived and any "discovery" is an
        identity.
        """
        raw = float(raw_best_abs_r)
        clean = float(clean_best_abs_r)
        if not raw > 0.99:
            raise ExperimentPreflightError(
                f"leak audit did not confirm a raw leak (best |r| = {raw:.4f}). "
                "Expected outcome-derived columns to reconstruct the target; "
                "verify the raw row set actually contains them."
            )
        if clean > 0.99:
            raise ExperimentPreflightError(
                f"cleaned rows still reconstruct the target (best |r| = {clean:.4f}). "
                "Target-derived columns survived the leak filter."
            )
        self._leak_audit = {"raw_best_abs_r": raw, "clean_best_abs_r": clean}

    def check_clean_predictors(
        self,
        clean_rows: Sequence[dict[str, Any]],
        *,
        predictor_prefixes: Sequence[str],
    ) -> None:
        """Clean rows must still expose contract predictors after leak filtering."""
        from rde.experiment.merge import predictor_columns, validate_clean_predictors

        validate_clean_predictors(
            clean_rows,
            predictor_prefixes=predictor_prefixes,
            target_metric=self.target,
        )
        cols = predictor_columns(clean_rows, predictor_prefixes)
        self._leak_audit = {
            **(self._leak_audit or {}),
            "clean_predictor_columns": cols,
            "clean_predictor_prefixes": list(predictor_prefixes),
            "n_clean_predictor_columns": len(cols),
        }

    def check_discovery_report(self, report: Any) -> None:
        """Required discovery stages must not have soft-failed before finalize."""
        errors = {
            str(err.get("stage")): err
            for err in (getattr(report, "stage_errors", None) or [])
            if err.get("stage")
        }
        blocking = ("expression_ranker", "descriptor_generators", "correlations")
        for stage in blocking:
            if stage not in errors:
                continue
            detail = errors[stage].get("error", "unknown error")
            raise ExperimentPreflightError(
                f"discovery stage {stage!r} failed: {detail}. "
                "A gated receipt requires the full discovery loop; inspect "
                "runs/discovery_stages.jsonl and runs/discovery_report.json."
            )

    # -- phase 4: discovery accounting ------------------------------------

    def record_phase(
        self,
        name: str,
        *,
        work_units: int,
        work_kind: str,
        detail: dict[str, Any] | None = None,
        soft_failed: bool = False,
    ) -> None:
        self._phases[name] = PhaseRecord(
            name=name,
            work_units=int(work_units),
            work_kind=work_kind,
            detail=dict(detail or {}),
            soft_failed=bool(soft_failed),
        )

    def record_discovery_report(self, report: Any) -> None:
        """Record which `run_discovery` phases genuinely produced output."""
        self.record_phase(
            "expression_ranker",
            work_units=len(getattr(report, "metric_candidates", []) or []),
            work_kind="ranked_expressions",
        )
        latent = getattr(report, "latent", {}) or {}
        self.record_phase(
            "latent",
            work_units=len(latent),
            work_kind="latent_fields",
            detail={"ridge_r_squared": latent.get("ridge_r_squared")},
        )
        symbolic = getattr(report, "symbolic", {}) or {}
        self.record_phase(
            "symbolic",
            work_units=len(symbolic),
            work_kind="symbolic_fields",
            detail={"r_squared": symbolic.get("r_squared")},
        )
        phase6 = getattr(report, "phase6", {}) or {}
        self.record_phase(
            "phase6",
            work_units=len(phase6),
            work_kind="phase6_fields",
            soft_failed=bool((phase6.get("rediscovery") or {}).get("soft_failed")),
        )
        for err in getattr(report, "stage_errors", []) or []:
            self._phases.setdefault(
                f"error:{err.get('stage', 'unknown')}",
                PhaseRecord(
                    name=f"error:{err.get('stage', 'unknown')}",
                    work_units=0,
                    work_kind="stage_error",
                    detail=dict(err),
                    soft_failed=True,
                ),
            )

    def check_discovery_phases(self) -> None:
        missing = [p for p in self.required_phases if p not in self._phases]
        if missing:
            raise ExperimentPreflightError(
                f"required discovery phases did not run: {missing}. Running only "
                "the pipeline (or only a lightweight ranker) is a screen, not a "
                "discovery — call the full run_discovery loop."
            )

    # -- phase 5: finalize -------------------------------------------------

    def finalize(
        self,
        *,
        verdict: str,
        grade: int | None = None,
        level: int | None = None,
        criteria: dict[str, Any] | None = None,
        decisive_criteria: Sequence[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate everything, then write receipt.json. Raises on violation.

        ``decisive_criteria`` names the criteria the pre-registered decision rule
        actually reads. It is required whenever ``criteria`` is supplied: a
        verdict must not rest on a criterion that never computed, and only the
        experiment knows which of its criteria are load-bearing versus which are
        legitimately not-applicable for its domain.

        ``level=`` is accepted as a deprecated alias for ``grade=`` so older
        experiment scripts fail with a gate message instead of an uncaught
        ``TypeError``.
        """
        if grade is not None and level is not None and int(grade) != int(level):
            raise ExperimentPreflightError(
                f"finalize(): grade={grade!r} and deprecated level={level!r} disagree."
            )
        used_level_alias = level is not None and grade is None
        resolved_grade = grade if grade is not None else level
        if resolved_grade is None:
            raise ExperimentPreflightError(
                "finalize() requires grade=<int> (legacy scripts may pass level=)."
            )
        grade = int(resolved_grade)
        extra = {
            **dict(extra or {}),
            **({"deprecated_finalize_level_alias": True} if used_level_alias else {}),
        }

        if self._prereg_hash is None:
            raise ExperimentPreflightError("check_plan() was never called.")
        if not self._population:
            raise ExperimentPreflightError("check_population() was never called.")
        if not self._leak_audit:
            raise ExperimentPreflightError("check_leak_audit() was never called.")
        self.record_phase("gated_outcome", work_units=1, work_kind="assessment")
        self.check_discovery_phases()

        criteria = criteria or {}
        criteria_audit = self._audit_criteria(criteria, decisive_criteria)

        receipt = {
            "receipt_version": RECEIPT_VERSION,
            "experiment": self.experiment_dir.name,
            "domain_id": self.domain_id,
            "contract_id": self._contract_id,
            "target": self.target,
            "gate_kind": self.gate_kind,
            "preregistration_sha256": self._prereg_hash,
            "verdict": verdict,
            "grade": int(grade),
            "plan": self._plan,
            "population": self._population,
            "leak_audit": self._leak_audit,
            "phases": {
                name: {
                    "work_units": p.work_units,
                    "work_kind": p.work_kind,
                    "soft_failed": p.soft_failed,
                    **({"detail": p.detail} if p.detail else {}),
                }
                for name, p in sorted(self._phases.items())
            },
            "criteria": criteria,
            "criteria_audit": criteria_audit,
            "environment": {
                "platform": platform.system().lower(),
                "machine": platform.machine(),
                "git_sha": _git_sha(),
            },
            **(extra or {}),
        }
        out = self.experiment_dir / self.receipt_filename
        out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        return receipt

    def _audit_criteria(
        self, criteria: dict[str, Any], decisive: Sequence[str] | None
    ) -> dict[str, Any]:
        """Refuse a verdict that rests on a criterion which never computed."""
        if criteria and decisive is None:
            raise ExperimentPreflightError(
                "finalize() requires decisive_criteria when criteria are supplied: "
                "name the criteria the pre-registered decision rule reads. Without "
                "it a NaN criterion silently reads as 'no effect', turning a broken "
                "measurement into a scientific conclusion."
            )
        decisive = tuple(decisive or ())
        vacuous = vacuous_criteria(criteria)

        absent = [name for name in decisive if name not in criteria]
        if absent:
            raise ExperimentPreflightError(
                f"decisive criteria {absent} are not present in criteria "
                f"{sorted(criteria)}; the decision rule reads values the run never "
                "produced."
            )
        unevaluated = sorted(name for name in decisive if name in vacuous)
        if unevaluated:
            raise ExperimentPreflightError(
                f"decisive criteria {unevaluated} did not evaluate (None/NaN). The "
                "verdict would rest on an unmeasured quantity. Either fix the "
                "measurement or drop the criterion from the decision rule and "
                "record why in PREREGISTRATION.md."
            )
        return {
            "decisive": list(decisive),
            "vacuous": vacuous,
            "not_applicable": sorted(set(vacuous) - set(decisive)),
        }


def _git_sha() -> str | None:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def validate_receipt(payload: dict[str, Any]) -> list[str]:
    """Structural validation of a receipt; returns a list of problems."""
    if payload.get("gate_kind") == "recovery":
        return _validate_recovery_receipt(payload)
    problems: list[str] = []
    if payload.get("receipt_version") != RECEIPT_VERSION:
        problems.append(
            f"receipt_version {payload.get('receipt_version')!r} != {RECEIPT_VERSION}"
        )
    for key in ("domain_id", "target", "preregistration_sha256", "verdict", "phases"):
        if not payload.get(key):
            problems.append(f"missing {key}")

    population = payload.get("population") or {}
    if not population.get("per_size"):
        problems.append("missing population.per_size")
    for size, stats in (population.get("per_size") or {}).items():
        distinct = int(stats.get("distinct_structural_instances", 0))
        required = int(stats.get("required_distinct", 0))
        if distinct < required:
            problems.append(
                f"size {size}: distinct instances {distinct} < required {required}"
            )

    visibility = population.get("mechanism_visibility")
    if not visibility:
        problems.append("missing population.mechanism_visibility")
    elif not visibility.get("visible"):
        problems.append("mechanism is not numerically visible to the predictors")

    audit = payload.get("criteria_audit")
    if audit is None:
        problems.append("missing criteria_audit")
    else:
        decisive = list(audit.get("decisive") or [])
        if not decisive:
            problems.append("criteria_audit declares no decisive criteria")
        rested_on_nan = sorted(set(decisive) & set(audit.get("vacuous") or []))
        if rested_on_nan:
            problems.append(f"verdict rests on unevaluated criteria: {rested_on_nan}")

    leak = payload.get("leak_audit") or {}
    if not leak:
        problems.append("missing leak_audit")
    else:
        if float(leak.get("raw_best_abs_r", 0.0)) <= 0.99:
            problems.append("leak_audit did not confirm a raw leak")
        if float(leak.get("clean_best_abs_r", 1.0)) > 0.99:
            problems.append("leak_audit shows cleaned rows still leak")

    phases = payload.get("phases") or {}
    missing = [p for p in DEFAULT_REQUIRED_PHASES if p not in phases]
    if missing:
        problems.append(f"missing required phases: {missing}")
    return problems


def _validate_recovery_receipt(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if payload.get("receipt_version") != RECEIPT_VERSION:
        problems.append(
            f"receipt_version {payload.get('receipt_version')!r} != {RECEIPT_VERSION}"
        )
    for key in ("domain_id", "target", "preregistration_sha256", "verdict", "phases"):
        if not payload.get(key):
            problems.append(f"missing {key}")
    population = payload.get("population") or {}
    if not population.get("per_size"):
        problems.append("missing population.per_size")
    for size, stats in (population.get("per_size") or {}).items():
        distinct = int(stats.get("distinct_structural_instances", 0))
        required = int(stats.get("required_distinct", 0))
        if distinct < required:
            problems.append(
                f"size {size}: distinct instances {distinct} < required {required}"
            )
    audit = payload.get("criteria_audit")
    if audit is None:
        problems.append("missing criteria_audit")
    else:
        decisive = list(audit.get("decisive") or [])
        if not decisive:
            problems.append("criteria_audit declares no decisive criteria")
        rested_on_nan = sorted(set(decisive) & set(audit.get("vacuous") or []))
        if rested_on_nan:
            problems.append(f"verdict rests on unevaluated criteria: {rested_on_nan}")
    leak = payload.get("leak_audit") or {}
    if leak.get("kind") != "extractor_blindness":
        problems.append("recovery leak_audit must be extractor_blindness")
    elif leak.get("extract_sees_planted") or leak.get("extract_sees_family"):
        problems.append("extractor isolation failed")
    phases = payload.get("phases") or {}
    missing = [p for p in RECOVERY_REQUIRED_PHASES if p not in phases]
    if missing:
        problems.append(f"missing required phases: {missing}")
    return problems
