"""Description-complexity measures for encoded representation values.

Phase 1 ships one reference measure, serialized size, so `Representation`
has something concrete to plug into `complexity=`. Ranking representations
by a multi-objective score (complexity vs. conversion cost vs. structure
exposed) is a later phase — this module only measures one representation's
own encoded payload.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ComplexityModel(Protocol):
    """Maps an encoded value to a description-complexity score (lower = simpler)."""

    def __call__(self, encoded_value: Any) -> float: ...


def serialized_size_complexity(encoded_value: Any) -> float:
    """Byte size of the encoded value as an array — a coarse complexity proxy."""
    return float(np.asarray(encoded_value).nbytes)
