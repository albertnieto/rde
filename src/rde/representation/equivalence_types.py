"""Typed equivalence notions between a representation's domain and carrier.

`equivalence.py`'s `check_roundtrip`/`EquivalenceResult` only ever checks
one notion — "decode(encode(x)) is within tolerance of x"
(`ApproximateEquivalence`, or `ExactEquality` at zero tolerance). The
original proposal names several more (isomorphism, isometry, unitary
equivalence, homeomorphism, general structure-preserving maps); this module
implements the ones this package's actual primitives can genuinely
instantiate, checked against real `(m, n)` matrices via
`operator.probe_encode_matrix` (any flat linear map) and
`operator.linear_probe_matrices` (square carriers only, adds the decode
side) — not declared as an abstract taxonomy with no example behind it.

Verified numerically before being written here (not asserted): probing the
grammar's Gram matrices (`E^H @ E`) shows `identity` and `dft_full` are
proportional to the identity (isometric up to a real positive scale, unitary
up to that same scale) while `difference` and `polynomial_vandermonde` are
not — both a positive and a negative real example exist for every claim
type below.

One more is now checkable — `check_structure_preserving_map`, the general
`StructurePreservingMap` entry, instantiated against `graph.py`'s
`RepresentationGraph` (nodes = representations, edges = transformations),
the one graph-structured object this package actually has. It checks a
graph homomorphism (does a node map preserve every edge?) and, as a
stronger optional fact, a graph isomorphism (is that map also a bijection
whose image's edge set exactly equals the target's?) — see that function's
docstring for why this is the honest, correctly-scoped instance of the
name, not a claim about topological homeomorphism (continuous bijection
with continuous inverse between topological spaces): `RepresentationGraph`
is a discrete labeled graph, not a topological space, so that notion has
nothing real to check it against *here*. `topology.py` now gives topological
homeomorphism its own genuine, non-vacuous instance elsewhere in this
package (a plane-rotation orbit-space quotient, unrelated to
`RepresentationGraph`) — kept out of this module because it is not an
instance of `StructurePreservingMapClaim`'s graph-theoretic notion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.representation.graph import RepresentationGraph
from rde.representation.operator import linear_probe_matrices, probe_encode_matrix
from rde.representation.representation import Representation


@dataclass(frozen=True)
class EquivalenceClaim:
    """One typed equivalence claim, checked against one representation."""

    representation_id: str
    equivalence_type: str
    holds: bool
    detail: str


def check_exact_equality(
    representation: Representation, value: Any, *, tolerance: float = 0.0
) -> EquivalenceClaim:
    """`ExactEquality`: `decode(encode(x)) == x` at (by default) zero tolerance.

    Distinct from `equivalence.check_roundtrip`'s default `tolerance=1e-9`:
    `ExactEquality` is the strict `0.0` case named in the original proposal,
    not "close enough" — pass a nonzero `tolerance` only when the caller
    explicitly means "exact up to floating-point roundoff", not as a default.
    """
    reconstructed = representation.decode(representation.encode(value))
    error = float(representation.distance(value, reconstructed))
    return EquivalenceClaim(
        representation_id=representation.representation_id,
        equivalence_type="exact_equality",
        holds=error <= tolerance,
        detail=f"error={error:.3e}, tolerance={tolerance:.3e}",
    )


def check_linear_isomorphism(
    representation: Representation, n: int, *, tolerance: float = 1e-6
) -> EquivalenceClaim:
    """`LinearIsomorphism`: encode/decode are literal `(n, n)` matrices, mutually inverse.

    Requires a *square* carrier (`m == n`) — `grammar.py`'s compact `dft`
    (real FFT into `C^{n//2+1}`) is an injective linear embedding, not an
    isomorphism, and correctly fails this check; `dft_full` (`C^n`) passes.
    """
    try:
        encode_matrix, decode_matrix = linear_probe_matrices(representation, n)
    except ValueError as exc:
        # linear_probe_matrices itself rejects non-square carriers (m != n)
        # — a non-square linear map cannot be bijective, so that ValueError
        # already means "not an isomorphism"; no separate shape check needed.
        return EquivalenceClaim(
            representation_id=representation.representation_id,
            equivalence_type="linear_isomorphism",
            holds=False,
            detail=str(exc),
        )
    residual = decode_matrix @ encode_matrix - np.eye(n)
    max_error = float(np.max(np.abs(residual)))
    return EquivalenceClaim(
        representation_id=representation.representation_id,
        equivalence_type="linear_isomorphism",
        holds=max_error <= tolerance,
        detail=f"max|D @ E - I|={max_error:.3e}, tolerance={tolerance:.3e}",
    )


def _gram_matrix(encode_matrix: np.ndarray) -> np.ndarray:
    hermitian_transpose = encode_matrix.conj().T if np.iscomplexobj(encode_matrix) else encode_matrix.T
    return hermitian_transpose @ encode_matrix


def check_isometry(
    representation: Representation, n: int, *, tolerance: float = 1e-6
) -> EquivalenceClaim:
    """`Isometry` (up to a fixed positive scale): `E^H @ E` proportional to `I`.

    Equivalent to `||encode(x)|| == c * ||x||` for every `x`, for one fixed
    `c > 0` — checked directly on the Gram matrix rather than by sampling
    vectors, since `E^H @ E` proportional to `I` is exactly that statement.
    `c` is reported, not assumed to be `1` (NumPy's unnormalized `fft`
    scales norms by `sqrt(n)`, so `dft_full` is an isometry with `c = n`
    on the Gram matrix, not `c = 1`).

    Uses `probe_encode_matrix` (encode-only), not `linear_probe_matrices`
    — isometry is well-defined for a non-square (injective) linear map, so
    `dft` (compact `rfft`, `m = n//2+1 < n`) gets a real, meaningful
    answer here (verified `off-diagonal max ~2.4`, i.e. `holds=False`, not
    a vacuous rejection for being non-square).
    """
    try:
        encode_matrix = probe_encode_matrix(representation, n)
    except ValueError as exc:
        return EquivalenceClaim(
            representation_id=representation.representation_id,
            equivalence_type="isometry",
            holds=False,
            detail=str(exc),
        )
    gram = _gram_matrix(encode_matrix)
    diagonal = np.diagonal(gram).real if np.iscomplexobj(gram) else np.diagonal(gram)
    scale = float(np.mean(diagonal))
    off_diagonal = gram - np.diag(np.diagonal(gram))
    off_diagonal_max = float(np.max(np.abs(off_diagonal))) if gram.size else 0.0
    diagonal_spread = float(np.max(np.abs(diagonal - scale))) if diagonal.size else 0.0
    holds = off_diagonal_max <= tolerance * max(abs(scale), 1.0) and diagonal_spread <= tolerance * max(
        abs(scale), 1.0
    )
    return EquivalenceClaim(
        representation_id=representation.representation_id,
        equivalence_type="isometry",
        holds=holds,
        detail=(
            f"scale c={scale:.3e}, off-diagonal max={off_diagonal_max:.3e}, "
            f"diagonal spread={diagonal_spread:.3e}, tolerance={tolerance:.3e}"
        ),
    )


def check_unitary_equivalence(
    representation: Representation, n: int, *, tolerance: float = 1e-6
) -> EquivalenceClaim:
    """`UnitaryEquivalence` (up to a fixed positive scale): complex-domain isometry.

    Same Gram-matrix test as `check_isometry`, restricted to representations
    tagged `"complex_domain"` (only `dft_full` in this grammar) — unitarity
    is specifically a complex-inner-product-space notion; checking it on a
    real-valued representation would just be relabeling `check_isometry`.
    """
    if "complex_domain" not in representation.invariants:
        return EquivalenceClaim(
            representation_id=representation.representation_id,
            equivalence_type="unitary_equivalence",
            holds=False,
            detail="not tagged 'complex_domain'; unitary equivalence is a complex-domain notion",
        )
    isometry_claim = check_isometry(representation, n, tolerance=tolerance)
    return EquivalenceClaim(
        representation_id=representation.representation_id,
        equivalence_type="unitary_equivalence",
        holds=isometry_claim.holds,
        detail=isometry_claim.detail,
    )


@dataclass(frozen=True)
class StructurePreservingMapClaim:
    """A node map between two `RepresentationGraph`s, checked for graph-structure preservation.

    `holds` is true iff `node_map` is a graph homomorphism: total on
    `source_graph`'s node set, and every source edge `(u, v)` has an image
    edge `(node_map[u], node_map[v])` in `target_graph`. `is_isomorphism` is
    the strictly stronger claim — `node_map` is additionally a bijection
    onto `target_graph`'s node set whose image's edge set exactly equals
    `target_graph`'s edge set (no source edge maps to a non-edge, and no
    target edge is left without a source preimage).
    """

    holds: bool
    is_isomorphism: bool
    detail: str


def check_structure_preserving_map(
    source_graph: RepresentationGraph,
    target_graph: RepresentationGraph,
    node_map: dict[str, str],
) -> StructurePreservingMapClaim:
    """The real, scoped instance of the original proposal's "general `StructurePreservingMap`":
    does `node_map` (`source representation_id -> target representation_id`) preserve
    `RepresentationGraph`'s actual structure (which nodes are connected by an edge)?

    Not a claim about topological homeomorphism — `RepresentationGraph` is a
    discrete labeled graph (nodes = `Representation`s, edges =
    `Transformation`s), not a topological space, so "continuous bijection
    with continuous inverse" has no meaning here. What *is* real and
    checkable is the graph-theoretic notion: a homomorphism is exactly "a
    map that preserves the edge relation," which is what "structure
    preserving" means for this package's one graph-structured object.

    `node_map` must be total on `source_graph.node_ids()` (every source node
    needs an image) — a partial map is rejected outright (`holds=False`)
    rather than silently checked only on the nodes it happens to cover,
    which would let an incomplete map pass by omission.
    """
    source_nodes = set(source_graph.node_ids())
    target_nodes = set(target_graph.node_ids())

    missing = source_nodes - set(node_map)
    if missing:
        return StructurePreservingMapClaim(
            holds=False,
            is_isomorphism=False,
            detail=f"node_map is not total on source_graph: missing {sorted(missing)}",
        )
    unknown_targets = set(node_map.values()) - target_nodes
    if unknown_targets:
        return StructurePreservingMapClaim(
            holds=False,
            is_isomorphism=False,
            detail=(
                f"node_map maps to id(s) not registered in target_graph: "
                f"{sorted(unknown_targets)}"
            ),
        )

    source_edges = source_graph.edge_pairs()
    target_edges = target_graph.edge_pairs()

    violations = sorted(
        (u, v) for (u, v) in source_edges if (node_map[u], node_map[v]) not in target_edges
    )
    if violations:
        return StructurePreservingMapClaim(
            holds=False,
            is_isomorphism=False,
            detail=(
                f"not edge-preserving: {len(violations)}/{len(source_edges)} source edge(s) "
                f"have no image edge in target_graph, e.g. {violations[0]}"
            ),
        )

    # Restricted to source_nodes' own images — an irrelevant extra key in
    # node_map (one not naming a source node) must not inflate this count
    # and make a non-bijective map look bijective by coincidence.
    mapped_values = [node_map[u] for u in source_nodes]
    is_bijection = len(mapped_values) == len(set(mapped_values)) and set(mapped_values) == target_nodes
    mapped_edges = {(node_map[u], node_map[v]) for (u, v) in source_edges}
    edges_match_exactly = mapped_edges == target_edges
    is_isomorphism = is_bijection and edges_match_exactly

    return StructurePreservingMapClaim(
        holds=True,
        is_isomorphism=is_isomorphism,
        detail=(
            f"edge-preserving (homomorphism) over {len(source_edges)} source edge(s); "
            f"bijective onto target's {len(target_nodes)} node(s): {is_bijection}; "
            f"image edge set exactly equals target's {len(target_edges)} edge(s): "
            f"{edges_match_exactly}"
        ),
    )
