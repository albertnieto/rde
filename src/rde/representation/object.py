"""The `Object` that a `Representation` encodes.

An `Object` names a mathematical entity abstractly (its type and identity),
independent of how it is encoded. The encoding is a `Representation`; the
object itself never changes when its representation does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Object:
    """An abstract mathematical object, prior to any representation choice."""

    object_id: str
    object_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
