"""`Transformation`: an edge between two representations of the same object type.

A transformation moves an encoded value from `source`'s carrier to
`target`'s carrier without going through user code — it composes the two
representations' own encode/decode: `target.encode(source.decode(value))`.
This keeps `Transformation` a thin, always-consistent edge rather than a
place to sneak in extra logic; if a domain needs a cheaper direct map, it
should express that as its own `Representation` pair, not a hand-written
transformation.

`cost` is a declared conversion cost (arbitrary units, default 1.0) that
`RepresentationGraph.find_path` minimizes — it says nothing about
correctness, which is `exact` / the roundtrip certificate's job.

Composing multiple edges into a path (`RepresentationGraph`) is a later
phase; this is the single-edge primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rde.representation.representation import Representation


@dataclass(frozen=True)
class Transformation:
    """A directed edge `source -> target` between two representations."""

    transformation_id: str
    source: Representation
    target: Representation
    cost: float = 1.0

    def __post_init__(self) -> None:
        if self.source.object_type != self.target.object_type:
            raise ValueError(
                f"Transformation {self.transformation_id!r} spans object types "
                f"{self.source.object_type!r} -> {self.target.object_type!r}; "
                "a transformation only moves between representations of the "
                "same object."
            )

    @property
    def exact(self) -> bool:
        return self.source.exact and self.target.exact

    def apply(self, encoded_in_source: Any) -> Any:
        """Map a value encoded under `source` to its encoding under `target`."""
        return self.target.encode(self.source.decode(encoded_in_source))

    def invert(self) -> Transformation:
        return Transformation(
            transformation_id=f"{self.transformation_id}_inv",
            source=self.target,
            target=self.source,
            cost=self.cost,
        )
