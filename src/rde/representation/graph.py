"""`RepresentationGraph`: representations as nodes, transformations as edges.

Phase 2 of the representation-discovery roadmap. A graph exists because you
will not always have a direct `Transformation` between two representations
you care about — you may only have `R1 -> R2` and `R2 -> R3` registered.
`find_path` finds the cheapest chain by declared `Transformation.cost`, and
`compose_path` chains those edges into one `TransformationPath`.

Chaining matters beyond convenience: each hop independently decodes and
re-encodes, so for approximate representations a multi-hop path can lose
more precision than a (if it existed) direct edge — `TransformationPath`
reports its own roundtrip behavior via `apply`, it does not assume the
result equals what a direct edge would give.

`compare` reuses Phase 1's `certify_roundtrip` to compare two representations
on the same value. Ranking representations by a multi-objective score
(description complexity vs. conversion cost vs. structure exposed) and
canonicalizing a representative are later phases — deliberately not
implemented here, since they need the Pareto/complexity machinery this
phase doesn't have yet.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any

from rde.representation.certificate import Certificate, certify_roundtrip
from rde.representation.representation import Representation
from rde.representation.transformation import Transformation


@dataclass(frozen=True)
class TransformationPath:
    """A chain of edges from `edges[0].source` to `edges[-1].target`."""

    edges: tuple[Transformation, ...]

    def __post_init__(self) -> None:
        if not self.edges:
            raise ValueError("TransformationPath requires at least one edge")
        for left, right in zip(self.edges, self.edges[1:]):
            if left.target.representation_id != right.source.representation_id:
                raise ValueError(
                    f"TransformationPath edges do not chain: "
                    f"{left.transformation_id!r} ends at "
                    f"{left.target.representation_id!r} but "
                    f"{right.transformation_id!r} starts at "
                    f"{right.source.representation_id!r}"
                )

    @property
    def source(self) -> Representation:
        return self.edges[0].source

    @property
    def target(self) -> Representation:
        return self.edges[-1].target

    @property
    def exact(self) -> bool:
        return all(edge.exact for edge in self.edges)

    @property
    def cost(self) -> float:
        return sum(edge.cost for edge in self.edges)

    def apply(self, encoded_in_source: Any) -> Any:
        """Map a value through every hop in order."""
        value = encoded_in_source
        for edge in self.edges:
            value = edge.apply(value)
        return value


class RepresentationGraph:
    """A registry of representations (nodes) and transformations (edges)."""

    def __init__(self) -> None:
        self._nodes: dict[str, Representation] = {}
        self._edges: dict[str, list[Transformation]] = {}

    def add_representation(self, representation: Representation) -> None:
        existing = self._nodes.get(representation.representation_id)
        if existing is not None and existing != representation:
            raise ValueError(
                f"Representation id {representation.representation_id!r} is "
                "already registered with a different Representation"
            )
        self._nodes[representation.representation_id] = representation
        self._edges.setdefault(representation.representation_id, [])

    def add_transformation(self, transformation: Transformation, *, bidirectional: bool = False) -> None:
        """Register an edge, auto-registering its endpoint nodes.

        `bidirectional=True` also registers `transformation.invert()` — only
        set it when the transformation's decode/encode pair is meant to be
        used both ways; it does not verify invertibility.
        """
        self.add_representation(transformation.source)
        self.add_representation(transformation.target)
        self._edges[transformation.source.representation_id].append(transformation)
        if bidirectional:
            inverse = transformation.invert()
            self._edges[inverse.source.representation_id].append(inverse)

    def representation(self, representation_id: str) -> Representation:
        return self._nodes[representation_id]

    def node_ids(self) -> tuple[str, ...]:
        """Every registered representation id — the graph's node set."""
        return tuple(self._nodes.keys())

    def edge_pairs(self) -> frozenset[tuple[str, str]]:
        """Every `(source_id, target_id)` pair with a registered edge — the graph's edge set.

        Collapses parallel edges (more than one `Transformation` between the
        same ordered pair) to one pair, since graph-structure-preservation
        (`equivalence_types.check_structure_preserving_map`) is a claim about
        which pairs are connected, not about how many edges connect them.
        """
        return frozenset(
            (source_id, edge.target.representation_id)
            for source_id, edges in self._edges.items()
            for edge in edges
        )

    def neighbors(self, representation_id: str) -> tuple[Transformation, ...]:
        return tuple(self._edges.get(representation_id, ()))

    def find_path(self, source_id: str, target_id: str) -> TransformationPath | None:
        """Cheapest chain of edges from `source_id` to `target_id`, by cost.

        Dijkstra over declared `Transformation.cost`. Returns `None` if no
        path exists, or if `source_id == target_id` (there is nothing to
        compose).
        """
        if source_id == target_id:
            return None
        if source_id not in self._nodes:
            raise KeyError(f"Unknown representation id: {source_id!r}")

        counter = itertools.count()
        frontier: list[tuple[float, int, str, tuple[Transformation, ...]]] = [
            (0.0, next(counter), source_id, ())
        ]
        best_cost: dict[str, float] = {source_id: 0.0}

        while frontier:
            cost, _, node, path = heapq.heappop(frontier)
            if node == target_id:
                return TransformationPath(edges=path)
            if cost > best_cost.get(node, float("inf")):
                continue
            for edge in self._edges.get(node, ()):
                next_id = edge.target.representation_id
                new_cost = cost + edge.cost
                if new_cost < best_cost.get(next_id, float("inf")):
                    best_cost[next_id] = new_cost
                    heapq.heappush(frontier, (new_cost, next(counter), next_id, path + (edge,)))
        return None

    def compare(
        self,
        representation_id_a: str,
        representation_id_b: str,
        value: Any,
        *,
        tolerance: float = 1e-9,
    ) -> dict[str, Certificate]:
        """Certify both representations' roundtrip claim on the same `value`."""
        rep_a = self._nodes[representation_id_a]
        rep_b = self._nodes[representation_id_b]
        return {
            representation_id_a: certify_roundtrip(rep_a, value, tolerance=tolerance),
            representation_id_b: certify_roundtrip(rep_b, value, tolerance=tolerance),
        }
