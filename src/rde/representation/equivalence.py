"""Equivalence checking between an object and its round-tripped reconstruction.

Phase 1 checks exactly one equivalence notion — numerical roundtrip within a
tolerance — via `Representation.distance`. Typed equivalence notions
(isomorphism, unitary equivalence, structure-preserving maps, ...) are a
later phase; do not read more into `EquivalenceResult` than "the decoded
reconstruction is within tolerance of the original value."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rde.representation.representation import Representation


@dataclass(frozen=True)
class EquivalenceResult:
    """Outcome of comparing a value to its encode/decode roundtrip."""

    equivalent: bool
    error: float
    tolerance: float


def check_roundtrip(
    representation: Representation,
    value: Any,
    *,
    tolerance: float = 1e-9,
) -> EquivalenceResult:
    """Check `decode(encode(value)) ~= value` under `representation.distance`."""
    reconstructed = representation.decode(representation.encode(value))
    error = representation.distance(value, reconstructed)
    return EquivalenceResult(equivalent=error <= tolerance, error=error, tolerance=tolerance)
