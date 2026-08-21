"""G5 candidate certifier with provenance/split/replication records (v0.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rde.analyze.leak_audit import audit_conjecture
from rde.core.protocols import ResourceModel
from rde.core.resources import _DEFAULT_RESOURCE, draft_poly_fields, ledger_for_run


@dataclass
class CertifyResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    resource_model: ResourceModel = _DEFAULT_RESOURCE
    poly_gates: str = "unknown"
    poly_shots: str = "unknown"
    query_intents: list[str] = field(default_factory=list)


def certify_representation_candidate(
    conjecture: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    target: str,
    resource_model: ResourceModel | None = None,
    query_intents: list[str] | None = None,
    n_sizes: int | None = None,
    split_record: dict[str, Any] | None = None,
    replication_record: dict[str, Any] | None = None,
    frozen_candidate: dict[str, Any] | None = None,
    domain_id: str,
) -> CertifyResult:
    """Pre-check a promoted conjecture for G5 handoff."""
    rm = resource_model or _DEFAULT_RESOURCE
    intents = query_intents or ["EVALUATE", "COMPRESS", "RANK"]
    violations: list[str] = []
    checks: dict[str, bool] = {}

    leak = audit_conjecture(conjecture, target=target, domain_id=domain_id)
    checks["leak_clean"] = leak.passed
    if not leak.passed:
        violations.extend(leak.violations)

    checks["catalog_checksum_present"] = bool(leak.catalog_checksum)
    if not leak.catalog_checksum:
        violations.append("missing feature catalog checksum")

    sizes = sorted({int(r["size"]) for r in rows if r.get("size") is not None})
    n_sz = n_sizes if n_sizes is not None else len(sizes)
    checks["multi_size"] = n_sz >= 2
    if n_sz < 2:
        violations.append("need >= 2 sizes for cross-N certification")

    checks["split_record_present"] = split_record is not None and bool(split_record.get("split_fold"))
    if not checks["split_record_present"]:
        violations.append("missing grouped split record on candidate")

    checks["replication_record_present"] = replication_record is not None and replication_record.get(
        "promotion_eligible"
    ) is not None
    if not checks["replication_record_present"]:
        violations.append("missing replication record")

    checks["frozen_candidate_present"] = frozen_candidate is not None
    if frozen_candidate is None:
        violations.append("candidate must be frozen before certification")

    cross_n = float(conjecture.get("cross_n_sign_stability", float("nan")))
    checks["cross_n_stable"] = np.isfinite(cross_n) and cross_n >= 0.6
    if not checks["cross_n_stable"]:
        extrap = float(conjecture.get("extrapolation_r_squared", float("nan")))
        checks["cross_n_stable"] = np.isfinite(extrap) and extrap >= 0.5
    if not checks["cross_n_stable"]:
        violations.append("missing cross-N stability or extrapolation R2")

    latent_dim = float(conjecture.get("latent_dim", 16.0))
    max_n = max(sizes) if sizes else 10
    checks["poly_dimension_proxy"] = latent_dim <= 8 * max_n
    if not checks["poly_dimension_proxy"]:
        violations.append(f"latent_dim proxy {latent_dim} too large vs N={max_n}")

    checks["query_intents_declared"] = len(intents) >= 1
    poly = draft_poly_fields(n_sizes=max(n_sz, 1), max_latent_dim=latent_dim, query_intents=intents)

    ledger = ledger_for_run(rows)
    checks["ledger_computed"] = ledger.get("n_rows", 0) > 0

    passed = all(checks.values()) and len(violations) == 0
    return CertifyResult(
        passed=passed,
        checks=checks,
        violations=violations,
        resource_model=rm,
        poly_gates=poly["poly_gates"],
        poly_shots=poly["poly_shots"],
        query_intents=intents,
    )
