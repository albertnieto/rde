# Agent Correction Playbook

Durable engineering memory for RDE work. `AGENTS.md` is the repository guide;
`CLAUDE.md` is the environment policy. Cursor rules in `.cursor/rules/` point
here.

## Before editing

1. Classify work as **RDE core** or **domain adapter** (HSP/TSP). Core must not
   import `rde_domains`.
2. Read `AGENTS.md`, `CLAUDE.md`, the relevant RDE docs, then inspect `git
   status` and the current diff. Preserve unrelated changes.
3. Use `.venv/bin/python3` for every project command.
4. Run `.venv/bin/python3 -m rde machine-profile` before backend/GPU work.
5. **Optimization-first:** new hot paths ship with MLX/vectorization/batching
   and worker-safe pipeline hooks. RDE experiments follow
   `src/rde/docs/experiment-playbook.md`.
6. Add or update regression tests with the implementation.

## Backend and device contracts

- RDE compute and expression evaluation are separate backend choices.
- Route through `rde.backends.resolve` and `rde.expression.batch.prepare_device_envs`.
- Detect the active machine with `rde machine-profile`. Never assume Intel vs
  Apple Silicon from static docs.
- On `intel_mac`, NumPy/CPU is default. On `apple_silicon_mac`, MLX is primary.
- Record requested backend, effective backend, and fallback reason.
- MLX work: batch, evaluate once, synchronize once, copy to NumPy at boundaries.
- NumPy remains the precision reference path.

## RDE performance rules

- Row dictionaries are an I/O boundary. Materialize once with
  `FeatureTable.from_rows`; do not rebuild arrays inside loops.
- Vectorize over rows, assignments, and candidates. Batch equal-shape inputs only.
- Avoid Python loops over \(2^N\) assignments unless a documented small-\(N\)
  exception.
- Reuse domain caches via `prepare_instance` for expensive shared primitives.
- Bound memory: chunked pools, streaming JSONL, observable sequential fallback.
- Non-trivial runs need live progress and durable structured logs.

## Numerical and data-integrity rules

- Mask non-finite pairs, guard denominators, preserve `NaN` for undefined stats.
- Persist configuration fingerprints, backend provenance, timings, and resume metadata.

## CLI, tests, and observability

- Test real CLI parsing and subprocess boundaries.
- An option is not implemented until it reaches execution layers and manifests.
- Run focused tests first, then the relevant full suite under `.venv`.
- Terminal output is not the experiment record.

## Pre-handoff checklist

- [ ] Existing user changes preserved.
- [ ] Backend choice and fallback behavior explicit and tested.
- [ ] No avoidable host/device transfer or Python array loop remains.
- [ ] Shapes, dtypes, masks, denominators, and NaN semantics tested.
- [ ] CLI/report/manifest/provenance wiring complete.
- [ ] Focused and relevant full tests ran under `.venv`.
