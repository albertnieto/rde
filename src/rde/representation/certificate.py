"""`Certificate`: a machine-readable record of a representation claim.

Keeps "observed numerically" separate from "proved" — a Phase 1 certificate
only ever claims `status="verified"` from a numerical roundtrip check, never
from a formal proof. Symbolic/formal verification is a later phase; when it
lands it must add its own claim type rather than overload this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rde.representation.equivalence import check_roundtrip
from rde.representation.representation import Representation


@dataclass(frozen=True)
class Certificate:
    """Verification record for one representation, against one value."""

    representation_id: str
    object_type: str
    claim: str
    status: str
    error: float
    tolerance: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "object_type": self.object_type,
            "claim": self.claim,
            "status": self.status,
            "error": self.error,
            "tolerance": self.tolerance,
        }


def certify_roundtrip(
    representation: Representation,
    value: Any,
    *,
    tolerance: float = 1e-9,
) -> Certificate:
    """Certify (or refute) a representation's roundtrip claim on `value`.

    `status` is "verified" or "refuted" — never "proved"; this is numerical
    evidence, not a formal guarantee.
    """
    result = check_roundtrip(representation, value, tolerance=tolerance)
    claim = "exact_roundtrip" if representation.exact else "approximate_roundtrip"
    status = "verified" if result.equivalent else "refuted"
    return Certificate(
        representation_id=representation.representation_id,
        object_type=representation.object_type,
        claim=claim,
        status=status,
        error=result.error,
        tolerance=result.tolerance,
    )
