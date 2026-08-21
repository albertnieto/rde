"""Provenance-aware feature catalog for RDE v0.3."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureOrigin(str, Enum):
    POLYNOMIAL_INPUT = "polynomial_input"
    ENUMERATED_ORACLE = "enumerated_oracle"
    TARGET_DERIVED = "target_derived"
    OUTCOME = "outcome"
    METADATA = "metadata"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FeatureSpec:
    key_pattern: str
    origin: FeatureOrigin
    predictor_eligible: bool
    source_primitive: str = ""
    asymptotic_cost: str = "poly"
    notes: str = ""

    def matches(self, key: str) -> bool:
        if self.key_pattern == key:
            return True
        if self.key_pattern.endswith("*") and key.startswith(self.key_pattern[:-1]):
            return True
        return False


# Trajectory domain default catalog (v0.3)
DEFAULT_TRAJECTORY_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("matrix.Q.*", FeatureOrigin.POLYNOMIAL_INPUT, True, source_primitive="Q"),
    FeatureSpec("graph.Q.*", FeatureOrigin.POLYNOMIAL_INPUT, True, source_primitive="Q"),
    FeatureSpec("A_coef", FeatureOrigin.POLYNOMIAL_INPUT, True),
    FeatureSpec("B_coef", FeatureOrigin.POLYNOMIAL_INPUT, True),
    FeatureSpec("Q_M", FeatureOrigin.POLYNOMIAL_INPUT, True),
    FeatureSpec("landscape.costs.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="costs", asymptotic_cost="exp"),
    FeatureSpec("walsh_log.w_amp.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="w_amp", asymptotic_cost="exp"),
    FeatureSpec("landscape.array.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="trajectory"),
    FeatureSpec("spectral.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="trajectory"),
    FeatureSpec("stats.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="trajectory"),
    FeatureSpec("compress.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="trajectory"),
    FeatureSpec("fourier.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="trajectory"),
    FeatureSpec("gen.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="trajectory"),
    FeatureSpec("dynamics.*", FeatureOrigin.TARGET_DERIVED, False),
    FeatureSpec("overlap.*", FeatureOrigin.TARGET_DERIVED, False),
    FeatureSpec("update.*", FeatureOrigin.TARGET_DERIVED, False),
    FeatureSpec("recurrence.*", FeatureOrigin.TARGET_DERIVED, False),
    FeatureSpec("metric.*", FeatureOrigin.OUTCOME, False),
    FeatureSpec("latent.*", FeatureOrigin.TARGET_DERIVED, False),
    FeatureSpec("size", FeatureOrigin.METADATA, False, notes="baseline covariate only"),
    FeatureSpec("seed", FeatureOrigin.METADATA, False),
    FeatureSpec("family_index", FeatureOrigin.METADATA, False, notes="baseline covariate only"),
    FeatureSpec("generator", FeatureOrigin.METADATA, False, notes="stratification only"),
    FeatureSpec("split_fold", FeatureOrigin.METADATA, False),
    FeatureSpec("array_size", FeatureOrigin.METADATA, False),
)


@dataclass
class FeatureCatalog:
    domain_id: str
    specs: tuple[FeatureSpec, ...] = DEFAULT_TRAJECTORY_SPECS
    contract_version: str = "0.3"
    _cache: dict[str, FeatureSpec | None] = field(default_factory=dict, repr=False)

    def classify(self, key: str) -> FeatureOrigin:
        spec = self.lookup(key)
        return spec.origin if spec else FeatureOrigin.UNKNOWN

    def lookup(self, key: str) -> FeatureSpec | None:
        if key in self._cache:
            return self._cache[key]
        best: FeatureSpec | None = None
        best_len = -1
        for spec in self.specs:
            if spec.matches(key):
                pat_len = len(spec.key_pattern)
                if pat_len > best_len:
                    best = spec
                    best_len = pat_len
        self._cache[key] = best
        return best

    def is_predictor_eligible(self, key: str) -> bool:
        spec = self.lookup(key)
        if spec is None:
            return False
        return spec.predictor_eligible

    def predictor_columns(self, keys: list[str]) -> list[str]:
        return [k for k in keys if self.is_predictor_eligible(k)]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "domain_id": self.domain_id,
            "specs": [
                {
                    "key_pattern": s.key_pattern,
                    "origin": s.origin.value,
                    "predictor_eligible": s.predictor_eligible,
                    "source_primitive": s.source_primitive,
                    "asymptotic_cost": s.asymptotic_cost,
                    "notes": s.notes,
                }
                for s in self.specs
            ],
        }

    def checksum(self) -> str:
        payload = json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def catalog_for_domain(domain_id: str) -> FeatureCatalog:
    from rde.core.domain_contract import domain_contract

    contract = domain_contract(domain_id)
    return FeatureCatalog(domain_id=domain_id, specs=contract.feature_specs)
