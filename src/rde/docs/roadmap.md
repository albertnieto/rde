# RDE implementation chronology

This document records engineering status for the standalone RDE library. It
is not a second science ladder: the science vocabulary is PZXESO, the two
operating modes are forward discovery and reverse synthesis, and outcome
strength is G0–G5. See [methodology](methodology.md).

## Current package boundary

The library owns:

- domain protocols, registries, entry-point loading, and reference toys;
- vectorized descriptors, metrics, expression evaluation, and backends;
- durable JSONL/NPZ/Parquet storage and resumable campaigns;
- Mode 1 discovery and the top of Mode 2 synthesis;
- outcome gates, leak audits, reports, and reproducibility metadata;
- CLI, package documentation, and the standalone packaging manifest.

Domain science is external. A plugin owns its generators, contracts, metrics,
targets, experiment presets, and optional commands. RDE core never imports
plugin packages.

## Ship slices

The historical implementation slices are engineering chronology only:

0. Core protocols, instance records, descriptors, metrics, and registry.
1. Single-run pipeline with worker-safe instance preparation and durable store.
2. Multi-size campaigns, bounded workers, resume fingerprints, and progress.
3. Descriptor catalogs, leak-aware feature contracts, and analysis tables.
4. Forward discovery: rankers, latent representations, symbolic hypotheses.
5. Outcome assessment, replication, obstruction, certification, and reports.
6. Mode 2 TARGET/ALGORITHM skeleton synthesis with recurrence pruning.
7. Standalone boundary: package docs, toys-only builtins, plugin entry points,
   slim manifest, and import-firewall tests.
8. Representation Core (experimental), under `representation/`:
   Phase 1 — `Object`/`Representation`/`Transformation`/`Certificate`
   primitives and numerical-roundtrip equivalence checking.
   Phase 2 — `RepresentationGraph` (nodes/edges), cheapest-path search,
   multi-hop `TransformationPath` composition, pairwise comparison.
   Phase 3 — dual-backend (NumPy/MLX) batched kernels, a fixed six-primitive
   representation grammar, exhaustive complexity ranking against sample data.
   Phase 4 — SymPy exact-rational `FormalCertificate`
   (`proved`/`disproved`), kept distinct from the numeric `Certificate`.
   Phase 5 — vectorized Pareto dominance/frontier ranking over
   representation objectives.
   Phase 6 — operator transport (`E_R @ U @ D_R`) and `off_diagonal_energy`;
   reproduces the textbook fact that the full complex DFT diagonalizes
   circulant operators using the grammar's own primitive.
   Phase 7 — exercised as an integration test only (composes phases 1-6),
   not new library code.
   Gap closure (after an explicit audit against the original proposal):
   `cost.py` (`computational_cost`, an asymptotic op-count estimate) wired
   into `search.SearchCandidate`/`pareto.objectives_from_candidates` as a
   third objective, closing the disconnect between "looks simple" and "was
   cheap to compute"; `equivalence_types.py` (`ExactEquality`,
   `LinearIsomorphism`, `Isometry`, `UnitaryEquivalence`, and later
   `check_structure_preserving_map`) checked against real probed matrices,
   not just numeric roundtrip. `check_structure_preserving_map` is the
   original proposal's general `StructurePreservingMap` entry, instantiated
   as a graph homomorphism/isomorphism check against `graph.py`'s
   `RepresentationGraph` (nodes = representations, edges = transformations)
   — the one graph-structured object this package has; `graph.py` gained
   `node_ids()`/`edge_pairs()` accessors for it. Explicitly not a claim about
   topological homeomorphism (`RepresentationGraph` is a discrete labeled
   graph, not a topological space) — that specific notion is still
   unimplemented, honestly scoped rather than mislabeled. That audit also found
   and fixed a latent unsoundness in `operator.linear_probe_matrices`:
   probing `dft`'s (compact `rfft`) decode side produced a shape-valid but
   mathematically meaningless matrix; `probe_encode_matrix` now separates
   the always-sound encode-only probe from the square-carrier-only decode
   probe, and `dft` is correctly excluded from `rank_by_diagonalization`.
   Domain integration: `rde_domains/hsp_functions/representations.py` and
   `rde_domains/tsp/representations.py` run the grammar/search machinery on
   real domain data (hsp_functions' bounded-query difference profile;
   Euclidean TSP's pairwise-distance profile) — exploratory tooling, not a
   `DomainContract`-integrated experiment (no leak audit, held-out-family
   discipline, or preregistered decision rule; see those modules'
   docstrings). Core still never imports `rde_domains`; these modules live
   on the domain side and import core, the direction the architecture
   already allows.
   CLI + persistence: `rde repr-rank` (generic demo batches or an `.npy`
   `--input`; never imports `rde_domains`, same core/domain boundary) and
   `representation/report.py` (`write_search_report`/
   `write_diagonalization_report`) give `Certificate.to_payload()` somewhere
   durable to go, as a standalone JSON report rather than forcing
   representation results into the campaign feature-table schema.
   Operator discovery from samples: `operator_discovery.discover_linear_operator`
   recovers an *unknown* `(n, n)` operator via least squares from paired
   samples alone (never given the operator directly), with an honest fit
   residual; `discover_and_rank_diagonalization` composes that with
   `rank_by_diagonalization` — closes the gap where "operator discovery"
   previously meant only transporting an already-known operator.
   Structure vocabulary: `structure.py` (`check_sparsity`,
   `check_periodicity`, `check_low_rank`, `check_separability`, and later
   `check_conservation`/`check_duality`) makes six members of the original
   proposal's "Structure Language" genuinely checkable against real batch
   data; `check_conservation` (invariance under a permutation group's
   simultaneous row/column action, defaulting to the cyclic group) and
   `check_duality` (does a named representation diagonalize an operator,
   reusing `operator.py`'s transport machinery) were verified against real
   `rde_domains.tsp.circulant`-generated distance matrices, not fabricated
   data (`tests/rde_domains/test_tsp_circulant_structure.py`) — a genuinely
   circulant one is exactly conserved under cyclic shift and exactly
   diagonalized by `dft_full`; a symmetry-broken one degrades continuously
   on both checks, with an emergent finding that the two checks' scores are
   numerically identical on the same data (Frobenius-norm-preserving
   projection and DFT are both norm-preserving linear maps of the same
   underlying decomposition). The rest (compositionality, degeneracy,
   topology, correlation, entanglement) is still not implemented — no
   representation here instantiates them. Fixed a real design flaw found
   while testing `check_low_rank`: `effective_rank < full_rank` was too weak a bar
   (trivially true for an almost-full-rank matrix); it now requires
   `effective_rank <= full_rank // 2` by default.
   Holdout / anti-cheating audit: `holdout.audit_holdout` ranks with a
   primitive genuinely withheld (`grammar.build_primitive_representations`'s
   new `primitive_subset`, also threaded through `search.rank_representations`/
   `best_representation`) and reports whether the visible-only ranking
   falsely claimed compression it doesn't have (`leakage_ratio`) — a
   controlled ablation over the known, fixed grammar, not general
   representation synthesis (still unimplemented; see `search.py`).
   Pipeline wiring + full persistence: `rde repr-rank-run` ranks the
   grammar against an array field a real `run`/`campaign` already
   materialized (via `Store.read_instance_features`/`load_array`) and
   writes the result through `Store.append_representation_report`
   (`representation_reports.jsonl`, a new run-scoped file, not the
   campaign feature-table schema) — purely additive, verified not to
   modify `run`/`campaign`'s existing artifacts. Deliberately does NOT
   touch `hsp_functions/domain.py`'s `primitive_features()` or Mode 1's
   `correlate_with_target`/`assess_outcome` G0-G5 gates: adding a predictor
   column there is a larger change than a gap-closure pass should make
   without sign-off. `hsp_functions/preregistered_experiment.py`
   is the honest substitute — a standalone preregistered statistical check
   (Spearman correlation between `structure_strength` and best-achievable
   representation complexity) with a real, reported-as-found result: a
   detected but shallow, likely `domain_kind`-confounded correlation
   (`identity` won 100% of 120 instances — no primitive found real
   compression), not evidence of coset-structure discovery.
   Canonicalization (`pareto.canonical_representation`, one explicit
   disclosed selection rule over the Pareto frontier), depth-`K`
   composition *over the existing grammar* (`layered.py`/`program_search.py`
   — composing known primitives deeper, not inventing new ones; see
   `docs/representation-synthesis-theory.md`), and `DomainContract`/G0-G5
   integration (`repr.best_complexity` wired into both `hsp_functions` and
   `tsp_landscape_stats`' feature contracts, and into `gate.py`'s
   `STRUCTURAL_PREFIXES`) are now done.
   Novel primitive invention: `grammar.py` gained an eighth stage-1
   primitive, `dct` (orthonormal type-II Discrete Cosine Transform) —
   scoped to one well-known, textbook-standard transform rather than
   open-ended "invent a building block nobody has described." Genuinely
   different basis from `dft`/`dft_full` (half-integer- vs.
   integer-frequency cosines), exactly orthogonal (`cond == 1`, unlike
   `polynomial_vandermonde`'s ill-conditioned inverse), and verified with an
   honestly-disproven first attempt documented alongside the real result
   (see `docs/representation-synthesis-theory.md` §11): a smooth-ramp
   compression claim at a loose `eps=1e-2` threshold disappeared once
   checked at the grammar's real `eps=1e-6`; the real, verified win is on
   data built as an exact sparse combination of `dct` basis vectors
   (complexity `2.0` vs. `dft_full`'s `13.17`). Open-ended primitive
   invention in general (search for a building block with no concrete
   target algorithm in mind) remains out of scope.

The numbers above do not correspond to G0–G5.

## Implemented contracts

Every domain is expected to declare:

- a `Domain` implementation and independent instance generator;
- a `DomainContract` with its primary target and structural predictors;
- held-out families and honest stage sizes when it supports discovery;
- `prepare_instance` when materialization and primitive features share work;
- requested/effective backend and fallback provenance for accelerated paths.

The built-in `synthetic_poly` and `block_separable` domains are reference toys,
not scientific evidence about any external problem family.

## Remaining work

1. Complete lower Mode 2 compilation layers from mathematical operations through
   encodings, machine operations, primitives, and resource ledgers.
2. Expand independent reference domains and plugin contract documentation.
3. Keep NumPy parity tests alongside MLX/vectorized implementations.
4. Publish the standalone package only after a clean extraction rehearsal,
   focused suite, package build, and CLI smoke test.

No item here claims a polynomial algorithm for a hard problem.
