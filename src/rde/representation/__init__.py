"""Representation Core.

Formalizes representations of mathematical objects as first-class artifacts,
distinct from the objects they represent: an object may have many
representations, and moving between them is a transformation with its own
cost and exactness — not an implicit side effect of feature extraction.

This subpackage is domain-agnostic infrastructure, like the rest of
`rde.core`. It defines no concrete representations of any research domain's
objects; domains supply those through `rde.domains` entry points, same as
`Domain` and `DomainContract`. Every batch numeric kernel here is vectorized
(`array_backend.py`, NumPy + MLX) — see each module's docstring for exactly
which loops are genuinely unavoidable control flow (a fixed, small grammar
of primitives; symbolic/SymPy term construction) versus which would be a
correctness bug (looping over batch/sample data).

Roadmap phases implemented here:

- Phase 1 (`object.py`, `representation.py`, `transformation.py`,
  `equivalence.py`, `certificate.py`, `complexity.py`) — the primitive
  types and numerical roundtrip checking.
- Phase 2 (`graph.py`) — `RepresentationGraph`: representations as nodes,
  transformations as costed edges, cheapest-path search, multi-hop
  composition, pairwise certificate comparison.
- Phase 3 (`array_backend.py`, `grammar.py`, `search.py`) — a fixed grammar
  of batched primitives (identity, matrix reshape, compact/full DFT, DCT,
  first-difference, sorted+permutation, polynomial-in-a-fixed-basis) and
  exhaustive ranking of them against a batch of sample data.
- Phase 4 (`symbolic.py`) — SymPy exact-rational proof, kept in a distinct
  `FormalCertificate` vocabulary (`proved`/`disproved`) from the numeric
  `Certificate` (`verified`/`refuted`) so the two are never conflated.
- Phase 5 (`pareto.py`) — vectorized Pareto dominance/frontier over
  representation objectives; no default single-number Q(R) scalarization
  (see module docstring for why), but an explicit opt-in one is offered.
- Phase 6 (`operator.py`) — operator transport across representations
  (`E_R @ U @ D_R`) and `off_diagonal_energy`, reproducing the textbook
  fact that the full complex DFT diagonalizes circulant operators using
  the grammar's actual `dft_full` primitive.

Gap-closure additions (added after an explicit audit of what "Phase 0-7"
actually covered vs. the original proposal):

- `cost.py` — `computational_cost`: an asymptotic (not benchmarked)
  operation-count estimate per primitive, wired into `search.SearchCandidate`
  and `pareto.objectives_from_candidates` as a third objective, so
  "simple-looking" and "cheap to compute" are no longer tracked in
  unrelated places.
- `equivalence_types.py` — typed equivalence (`ExactEquality`,
  `LinearIsomorphism`, `Isometry`, `UnitaryEquivalence`) checked against
  real probed matrices, not just the one numeric-roundtrip notion
  `equivalence.py` offers, plus `check_structure_preserving_map` — the
  general `StructurePreservingMap` vocabulary entry, instantiated as a
  graph homomorphism/isomorphism check against `graph.py`'s
  `RepresentationGraph` (the one graph-structured object this package has).
  Topological homeomorphism specifically remains unimplemented as of this
  gap-closure pass — `RepresentationGraph` is a discrete labeled graph, not
  a topological space, so that notion has nothing real to check it against
  yet. (Superseded later in this docstring: `topology.py` closes this gap
  with a genuine, non-vacuous instance — see below.)
- `structure.py` — a small, checkable "Structure Language" (sparsity,
  periodicity via spectral concentration, low-rank/separability via
  singular-value decay, conservation via group-orbit-average deviation,
  duality via `operator.py`'s diagonalization transport) toward the
  original proposal's larger vocabulary; as of this gap-closure pass, the
  rest (compositionality, degeneracy, topology, correlation, entanglement)
  is not implemented — nothing in this package's grammar instantiates them
  yet. (Superseded later in this docstring: `layered.py`/`program_search.py`
  implement composition, and `topology.py` implements a topology instance
  — see below. `degeneracy`/`correlation`/`entanglement` remain genuinely
  absent.)
- `operator_discovery.py` — recovers an *unknown* linear operator from
  paired samples (`discover_linear_operator`, least squares) before ranking
  how well the grammar diagonalizes it, rather than assuming the operator
  is already a known matrix the way `operator.py`'s `transport_operator`
  does.
- `report.py` — durable JSON reports for search/diagonalization results
  (`Certificate.to_payload()` previously had nowhere to go).
- `rde_domains/hsp_functions/representations.py` and
  `rde_domains/tsp/representations.py` (outside this package, on the domain
  side) run the grammar/search machinery on real hsp_functions/TSP data —
  exploratory tooling, not a `DomainContract`-integrated experiment.
- `cli/commands.py`'s `rde repr-rank` — CLI access to `rank_representations`
  over demo batches or an `.npy` file; never imports `rde_domains`.
- `holdout.py` — `audit_holdout`: ranks with a primitive genuinely withheld
  (via `grammar.build_primitive_representations`'s `primitive_subset`) and
  checks the visible-only ranking doesn't falsely claim compression it
  doesn't have (`leakage_ratio`, not a binary pass/fail — see module
  docstring for a real partial-leakage example this uncovered). A
  controlled ablation over the known, fixed grammar.
- `layered.py` — representation *synthesis*, Phase 1 of
  `docs/representation-synthesis-theory.md`: composes two grammar
  primitives into a genuinely new `Representation` when the second's
  `input_carrier_kind` matches the first's `carrier_kind` (meaningful
  specifically because it's *not* the vacuous flat case `search.py`
  rules out — a second stage sees the first stage's actual carrier, e.g.
  `dft_full`'s `C^n`, not the original real vector). Ships two example
  compositions with an honestly-reported empirical result (both round-trip
  exactly and beat their own un-layered base, neither beats `identity`
  under the current sparsity-based complexity metric — see the theory
  doc §4 for why, and §5 for what's still open). Composed representations
  are plain `Representation` objects, so `search.py`/`report.py`/
  `holdout.py` all already work on them; none of those call sites include
  them automatically yet.

- `pareto.canonical_representation` — one explicit, disclosed selection rule
  (lowest complexity, then conversion cost, then roundtrip error, then id)
  restricted to the Pareto frontier — closes the "canonicalization is not
  implemented" gap noted above; no new machinery, `pareto_rank` already
  computed the frontier.
- `program_search.py` — exhaustive depth-`K` composition of `grammar.py`'s
  and `layered.py`'s typed primitives (`layered.py` fixes depth at 2;
  `enumerate_chains`/`search_chains` generalize that to any `max_depth` via
  repeated `compose_layers`). Genuinely finds compositions depth-2 cannot:
  `sorted_permutation+sorted_then_difference+sorted_then_difference`
  (second-order differencing of sorted data) beats the depth-2 chain on
  piecewise-linear-once-sorted data, verified numerically before being
  written into a test (`tests/rde/representation/test_program_search.py`).
  `search_chains` requires an independent `holdout_batch` and ranks by
  holdout, not train, complexity — the same anti-overfitting discipline
  `holdout.py` established, applied to a genuinely larger search space.
  Exhaustive DFS, not genetic/heuristic search: the carrier-kind
  compatibility graph today is small and sparse enough that exhaustive
  enumeration is already optimal (see the module docstring for the specific
  reasoning); this is deliberately not "invent novel primitives" — it only
  composes primitives that already exist. `search_chains`'s verify/holdout-
  rank loop now delegates to `rde.search.holdout_search.search_with_holdout`
  (behavior unchanged) — `rde.recovery.search_space.search_recovery_chains`
  independently converged on the identical shape for a completely different
  candidate type, which is why that shape now lives in `rde.search` instead
  of being reimplemented per module; see `docs/roadmap.md` item 10.
- `grammar.py`'s `dct` — a genuinely new eighth stage-1 primitive (not a
  composition of existing ones), scoped to one well-known, textbook-standard
  transform rather than open-ended "invent a building block nobody has
  described": the orthonormal type-II Discrete Cosine Transform, whose
  half-integer-frequency cosine basis is structurally different from
  `dft`/`dft_full`'s integer-frequency one. Reuses `matmul_shared` the same
  way `polynomial_vandermonde` does, but its basis matrix is exactly
  orthogonal (`cond == 1` for every `n` tested) rather than the famously
  ill-conditioned Vandermonde inverse. An initial compression claim (beats
  `dft` on a smooth non-periodic ramp) quietly disappeared once checked
  against the grammar's real `eps=1e-6` threshold instead of a looser one —
  documented as a real, disproven first attempt rather than quietly fixed
  (see `docs/representation-synthesis-theory.md` §11 for the honest,
  verified replacement claim: `2.0` complexity vs. `dft_full`'s `13.17` on
  data built as an exact sparse combination of `dct` basis vectors). Being a
  new stage-1 primitive (not stage-2) multiplies, not just adds to,
  `program_search.enumerate_chains`'s chain count, since it is both a new
  `start` and a new same-carrier-kind `continuation`.

Phase 7 ("open discovery": an unknown object through the whole pipeline end
to end) is exercised as an integration test
(`tests/rde/representation/test_open_discovery_integration.py`), not as new
library code — it composes Phases 1-6 as they already exist.

- `topology.py` — the first genuinely non-vacuous topological homeomorphism
  instance (`equivalence_types.py` previously declined to claim this:
  `RepresentationGraph` is a discrete labeled graph, not a topological
  space). Scoped like `grammar.py`'s `dct` (one well-known fact, not
  open-ended invention): `X = C \\ {0}`, `G = Z_n` acting by rotation (the
  continuous analogue of `structure._cyclic_group_permutations`, and the
  same rotation group `rde_domains.tsp.circulant`'s `_points_on_circle`
  already uses), and the claim that `f(z) = z**n` induces a homeomorphism
  `(C\\{0})/Z_n -> C\\{0}`. Numeric side (`HomeomorphismClaim`):
  well-definedness on orbits, sampled injectivity, sampled continuity, and
  one honest disproven-first-attempt — a single-valued "principal branch"
  inverse on a fundamental domain is *not* continuous at the domain seam
  (two points ~1e-8 apart in image are ~1.0 apart under that naive inverse),
  which is exactly why the homeomorphism claim must be about the abstract
  quotient space, not a literal fundamental-domain subset of `C`. Formal
  side reuses `symbolic.FormalCertificate`: `prove_rotation_quotient_well_defined`
  closes for every `n` tried; `prove_rotation_quotient_injective` closes for
  `n = 3, 4, 5, 6, 8` but not `n = 7` (`cos(pi/7)` is not expressible in the
  real-radical form the `sympy.nsimplify` strategy used here searches for —
  a disclosed limitation of that proof strategy, not a bug). Not done, and
  not implied by this result: homeomorphism checking for the actual
  `tsp_circulant_symmetry` distance-matrix orbit space (a discrete
  permutation group on `R^{n x n}`, a harder and separate case) remains open.
- `adaptive.py` — a bounded instance of "open-ended primitive invention"
  (theory doc §5 / roadmap.md: "search for a building block with no
  concrete target algorithm in mind" was out of scope for lacking a finish
  line). Scoped like `dct`: one well-known, textbook-standard fact (the
  Karhunen-Loeve theorem: the covariance eigenbasis is the optimal
  orthonormal linear basis for concentrating variance into few
  coefficients), not open-ended search. `build_klt_representation` fits a
  basis from a training batch — architecturally distinct from every
  `grammar.py` primitive (analytic, needs only `n`), so it is deliberately
  *not* added to `grammar.py`'s `_PRIMITIVE_BUILDERS` registry, the same way
  `operator_discovery.py` stays standalone from `operator.py`. Preregistered
  holdout comparison (`run_klt_holdout_comparison`, decided before being
  run): on an exact rank-3 Gaussian factor family no fixed analytic basis
  has any structural relationship to, `klt` reaches holdout complexity
  `3.0` (exactly the true subspace dimension) against `dft`'s `9.0` (the
  best fixed primitive) — ratio `0.333`, under the preregistered `0.5`
  margin. Honest negative result, not hidden: the same comparison with
  noise added shows *no* compression for `klt` under this grammar's real
  `eps=1e-6` threshold (noise floor exceeds `eps` almost surely) — this
  metric only rewards exact near-zero coefficients, not approximate low
  rank, which is why the preregistered target family is noise-free.
  `run_klt_noise_sensitivity` characterizes exactly where that breakdown
  happens (a smooth transition centered almost exactly on `eps=1e-6`, swept
  from `noise_scale=0` to `1e-2`) rather than leaving it at one pass/one
  fail data point; `_best_grammar_holdout_complexity` now raises a clear
  `RuntimeError` instead of an opaque `ValueError` if nothing ever verifies.
"""

from __future__ import annotations

from rde.representation.adaptive import (
    KltHoldoutComparison,
    KltNoiseSensitivityPoint,
    build_klt_representation,
    fit_klt_basis,
    low_rank_factor_batch,
    run_klt_holdout_comparison,
    run_klt_noise_sensitivity,
)
from rde.representation.array_backend import (
    ArraySearchBackend,
    MlxSearchBackend,
    NumpySearchBackend,
    get_array_backend,
)
from rde.representation.certificate import Certificate, certify_roundtrip
from rde.representation.complexity import ComplexityModel, serialized_size_complexity
from rde.representation.cost import computational_cost
from rde.representation.equivalence import EquivalenceResult, check_roundtrip
from rde.representation.equivalence_types import (
    EquivalenceClaim,
    StructurePreservingMapClaim,
    check_exact_equality,
    check_isometry,
    check_linear_isomorphism,
    check_structure_preserving_map,
    check_unitary_equivalence,
)
from rde.representation.grammar import build_primitive_representations, primitive_names
from rde.representation.graph import RepresentationGraph, TransformationPath
from rde.representation.holdout import HoldoutAudit, audit_holdout
from rde.representation.layered import (
    build_layered_representations,
    compose_layers,
    stage2_primitive_names,
)
from rde.representation.object import Object
from rde.representation.operator import (
    DiagonalizationCandidate,
    linear_probe_matrices,
    off_diagonal_energy,
    probe_encode_matrix,
    rank_by_diagonalization,
    transport_operator,
)
from rde.representation.operator_discovery import (
    OperatorRecovery,
    discover_and_rank_diagonalization,
    discover_linear_operator,
)
from rde.representation.pareto import (
    OBJECTIVE_NAMES,
    ParetoResult,
    canonical_representation,
    dominance_matrix,
    objectives_from_candidates,
    pareto_rank,
    weighted_score,
)
from rde.representation.program_search import (
    ChainSearchResult,
    atomic_registry,
    enumerate_chains,
    search_chains,
)
from rde.representation.report import (
    diagonalization_report_payload,
    search_report_payload,
    write_diagonalization_report,
    write_diagonalization_report_to_store,
    write_search_report,
    write_search_report_to_store,
)
from rde.representation.representation import Representation
from rde.representation.search import SearchCandidate, best_representation, rank_representations
from rde.representation.structure import (
    StructureClaim,
    check_conservation,
    check_duality,
    check_low_rank,
    check_periodicity,
    check_separability,
    check_sparsity,
)
from rde.representation.symbolic import (
    FormalCertificate,
    discover_parity_claim,
    prove_vandermonde_inverse,
)
from rde.representation.topology import (
    HomeomorphismClaim,
    canonical_representative,
    check_map_continuity_sampled,
    check_naive_branch_inverse_discontinuous,
    check_quotient_injective_sampled,
    check_quotient_well_defined,
    prove_rotation_quotient_injective,
    prove_rotation_quotient_well_defined,
)
from rde.representation.transformation import Transformation

__all__ = [
    "ArraySearchBackend",
    "Certificate",
    "ChainSearchResult",
    "ComplexityModel",
    "DiagonalizationCandidate",
    "EquivalenceClaim",
    "EquivalenceResult",
    "FormalCertificate",
    "HoldoutAudit",
    "HomeomorphismClaim",
    "KltHoldoutComparison",
    "KltNoiseSensitivityPoint",
    "MlxSearchBackend",
    "NumpySearchBackend",
    "OBJECTIVE_NAMES",
    "Object",
    "OperatorRecovery",
    "ParetoResult",
    "Representation",
    "RepresentationGraph",
    "SearchCandidate",
    "StructureClaim",
    "StructurePreservingMapClaim",
    "Transformation",
    "TransformationPath",
    "atomic_registry",
    "audit_holdout",
    "best_representation",
    "build_klt_representation",
    "build_layered_representations",
    "build_primitive_representations",
    "canonical_representation",
    "canonical_representative",
    "certify_roundtrip",
    "check_conservation",
    "check_duality",
    "check_exact_equality",
    "check_isometry",
    "check_linear_isomorphism",
    "check_low_rank",
    "check_map_continuity_sampled",
    "check_naive_branch_inverse_discontinuous",
    "check_periodicity",
    "check_quotient_injective_sampled",
    "check_quotient_well_defined",
    "check_roundtrip",
    "check_separability",
    "check_sparsity",
    "check_structure_preserving_map",
    "check_unitary_equivalence",
    "compose_layers",
    "computational_cost",
    "diagonalization_report_payload",
    "discover_and_rank_diagonalization",
    "discover_linear_operator",
    "discover_parity_claim",
    "dominance_matrix",
    "enumerate_chains",
    "fit_klt_basis",
    "get_array_backend",
    "linear_probe_matrices",
    "low_rank_factor_batch",
    "objectives_from_candidates",
    "off_diagonal_energy",
    "pareto_rank",
    "primitive_names",
    "probe_encode_matrix",
    "prove_rotation_quotient_injective",
    "prove_rotation_quotient_well_defined",
    "prove_vandermonde_inverse",
    "rank_by_diagonalization",
    "rank_representations",
    "run_klt_holdout_comparison",
    "run_klt_noise_sensitivity",
    "search_chains",
    "search_report_payload",
    "serialized_size_complexity",
    "stage2_primitive_names",
    "transport_operator",
    "weighted_score",
    "write_diagonalization_report",
    "write_diagonalization_report_to_store",
    "write_search_report",
    "write_search_report_to_store",
]
