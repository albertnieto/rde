"""Leak and tautology audit using provenance-aware feature catalogs (v0.3)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rde.core.feature_contract import FeatureCatalog, FeatureOrigin, catalog_for_domain


class FeatureClass(str, Enum):
    INPUT_PRIMITIVE = "input_primitive"
    ENUMERATION = "enumeration"
    TARGET_DERIVED = "target_derived"
    OUTCOME_LEAK = "outcome_leak"
    METADATA = "metadata"
    UNKNOWN = "unknown"


_DYNAMICS_PREFIXES = ("dynamics.", "overlap.", "update.", "recurrence.")

_LEGACY_INPUT_PREFIXES = (
    "matrix.",
    "graph.",
)


@dataclass
class LeakAuditResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    allowed_features: list[str] = field(default_factory=list)
    blocked_features: list[str] = field(default_factory=list)
    feature_classes: dict[str, str] = field(default_factory=dict)
    catalog_checksum: str = ""


def _origin_to_class(origin: FeatureOrigin) -> FeatureClass:
    mapping = {
        FeatureOrigin.POLYNOMIAL_INPUT: FeatureClass.INPUT_PRIMITIVE,
        FeatureOrigin.ENUMERATED_ORACLE: FeatureClass.ENUMERATION,
        FeatureOrigin.TARGET_DERIVED: FeatureClass.TARGET_DERIVED,
        FeatureOrigin.OUTCOME: FeatureClass.OUTCOME_LEAK,
        FeatureOrigin.METADATA: FeatureClass.METADATA,
        FeatureOrigin.UNKNOWN: FeatureClass.UNKNOWN,
    }
    return mapping.get(origin, FeatureClass.UNKNOWN)


def classify_feature(
    name: str,
    *,
    target: str,
    catalog: FeatureCatalog | None = None,
    domain_id: str | None = None,
) -> FeatureClass:
    cat = catalog
    if cat is None and domain_id is not None:
        try:
            cat = catalog_for_domain(domain_id)
        except KeyError:
            # Outcome summaries may inspect legacy or synthetic stores without
            # a registered contract; retain the conservative prefix fallback.
            cat = None
    if cat is None:
        from rde.core.feature_contract import FeatureCatalog

        cat = FeatureCatalog(domain_id="<unbound>")
    spec = cat.lookup(name)
    if spec is not None:
        cls = _origin_to_class(spec.origin)
        if spec.origin == FeatureOrigin.TARGET_DERIVED:
            return FeatureClass.OUTCOME_LEAK
        if not spec.predictor_eligible:
            if spec.origin in (FeatureOrigin.TARGET_DERIVED, FeatureOrigin.OUTCOME):
                return FeatureClass.OUTCOME_LEAK if spec.origin == FeatureOrigin.OUTCOME else FeatureClass.TARGET_DERIVED
            if spec.origin == FeatureOrigin.ENUMERATED_ORACLE:
                return FeatureClass.ENUMERATION
            if spec.origin == FeatureOrigin.METADATA:
                return FeatureClass.METADATA
        return cls

    # Legacy prefix fallback — always conservative
    if name.startswith("metric."):
        return FeatureClass.OUTCOME_LEAK
    if name.startswith("latent."):
        return FeatureClass.TARGET_DERIVED
    if name.startswith(_DYNAMICS_PREFIXES):
        return FeatureClass.TARGET_DERIVED
    if name.startswith(("landscape.", "walsh_log.", "compress.", "spectral.", "fourier.", "gen.", "stats.")):
        return FeatureClass.ENUMERATION
    if name.startswith(_LEGACY_INPUT_PREFIXES):
        return FeatureClass.INPUT_PRIMITIVE
    if name in {"size", "seed", "family_index", "array_size", "generator", "split_fold"}:
        return FeatureClass.METADATA
    return FeatureClass.UNKNOWN


def extract_feature_names_from_expression(expression: str) -> list[str]:
    tokens = set(re.findall(r"[a-zA-Z_][\w.]*", expression))
    skip = {
        "log",
        "exp",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "abs",
        "latent",
        "metric",
        "dynamics",
        "compress",
        "spectral",
        "fourier",
        "walsh_log",
        "matrix",
        "graph",
        "landscape",
        "gen",
        "overlap",
        "update",
        "recurrence",
    }
    features: list[str] = []
    for tok in sorted(tokens):
        if tok.replace(".", "").replace("_", "").isdigit():
            continue
        if tok in skip:
            continue
        if "." in tok:
            features.append(tok)
    return features


def audit_features(
    features: list[str],
    *,
    target: str,
    catalog: FeatureCatalog | None = None,
    domain_id: str,
) -> LeakAuditResult:
    cat = catalog or catalog_for_domain(domain_id)
    violations: list[str] = []
    allowed: list[str] = []
    blocked: list[str] = []
    classes: dict[str, str] = {}

    for feat in features:
        cls = classify_feature(feat, target=target, catalog=cat, domain_id=domain_id)
        classes[feat] = cls.value
        spec = cat.lookup(feat)
        if spec is None:
            violations.append(f"{feat}: unknown provenance (blocked by default)")
            blocked.append(feat)
            continue
        if not spec.predictor_eligible:
            blocked.append(feat)
            violations.append(f"{feat}: {spec.origin.value} (not predictor-eligible)")
        elif cls in (FeatureClass.OUTCOME_LEAK, FeatureClass.TARGET_DERIVED, FeatureClass.ENUMERATION):
            blocked.append(feat)
            violations.append(f"{feat}: {cls.value}")
        elif cls == FeatureClass.UNKNOWN:
            blocked.append(feat)
            violations.append(f"{feat}: unknown")
        elif cls == FeatureClass.METADATA and feat not in {"size", "family_index"}:
            blocked.append(feat)
            violations.append(f"{feat}: metadata (stratification only)")
        else:
            allowed.append(feat)

    if any(f.startswith("metric.") for f in features):
        violations.append("metric.* columns must not appear in predictors")

    return LeakAuditResult(
        passed=len(violations) == 0,
        violations=violations,
        allowed_features=allowed,
        blocked_features=blocked,
        feature_classes=classes,
        catalog_checksum=cat.checksum(),
    )


def audit_conjecture(
    record: dict[str, Any],
    *,
    target: str,
    catalog: FeatureCatalog | None = None,
    domain_id: str,
) -> LeakAuditResult:
    features = list(record.get("feature_columns") or [])
    expr = str(record.get("expression", ""))
    if not features and expr:
        features = extract_feature_names_from_expression(expr)
    result = audit_features(features, target=target, catalog=catalog, domain_id=domain_id)
    if result.passed:
        expr_lower = expr.lower()
        for prefix in _DYNAMICS_PREFIXES:
            if prefix.rstrip(".") in expr_lower:
                return LeakAuditResult(
                    passed=False,
                    violations=[*result.violations, f"expression uses {prefix} against {target}"],
                    allowed_features=result.allowed_features,
                    blocked_features=[*result.blocked_features, prefix],
                    feature_classes=result.feature_classes,
                    catalog_checksum=result.catalog_checksum,
                )
    return result


def audit_conjectures(
    records: list[dict[str, Any]],
    *,
    target: str,
    catalog: FeatureCatalog | None = None,
    domain_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    cat = catalog or catalog_for_domain(domain_id)
    for rec in records:
        audit = audit_conjecture(rec, target=target, catalog=cat, domain_id=domain_id)
        tagged = {**rec, "leak_audit": {"passed": audit.passed, "violations": audit.violations}}
        if audit.passed:
            clean.append(tagged)
        else:
            blocked.append(tagged)
    summary = {
        "n_total": len(records),
        "n_passed": len(clean),
        "n_blocked": len(blocked),
        "promotion_blocked": len(blocked) > 0 and len(clean) == 0,
        "catalog_checksum": cat.checksum(),
        "blocked_records": blocked,
    }
    return clean, summary
