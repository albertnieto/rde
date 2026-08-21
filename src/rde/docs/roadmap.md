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
