# RDE repository agent guide

Self-contained open-source distribution of the Representation Discovery Engine
(RDE) core and HSP/TSP domain adapters. No QPFA, QUBO-codesign, or quantum
simulation code lives in this repository.

## Package layout

- [`src/rde/`](src/rde/) — domain-agnostic core (`rde` on PyPI)
- [`src/rde_domains/`](src/rde_domains/) — HSP + TSP plugins (`rde-domains` on PyPI)

Read the in-package docs before changing science or pipeline behavior:

- [`src/rde/docs/methodology.md`](src/rde/docs/methodology.md) — PZXESO, G0–G5
- [`src/rde/docs/ARCHITECTURE.md`](src/rde/docs/ARCHITECTURE.md) — pipeline
- [`src/rde/docs/experiment-playbook.md`](src/rde/docs/experiment-playbook.md)
- [`src/rde/docs/README.md`](src/rde/docs/README.md) — CLI

Domain charters:

- [`docs/research/hidden-subgroup-function-discovery-charter.md`](docs/research/hidden-subgroup-function-discovery-charter.md)
- [`docs/research/tsp-novel-representation-discovery-charter.md`](docs/research/tsp-novel-representation-discovery-charter.md)

## Test boundary

- `tests/rde/` — core-only; must not import `rde_domains`
- `tests/rde_domains/` — HSP/TSP adapter tests

## Environment

- Work on `main`; do not create branches or PRs unless requested
- Preserve unrelated changes; inspect `git status` and diff before editing
- Run project code with `.venv/bin/python3`, never system Python
- Session start:

  ```bash
  .venv/bin/python3 -m rde machine-profile --json
  ```

  Bind backend claims to `profile_id`: MLX on `apple_silicon_mac`, NumPy on
  `intel_mac`.

## Implementation bar

- New hot paths: vectorize, batch, cache shared primitives, MLX on Apple Silicon
- NumPy parity tests for every fast path; record requested/effective backend
- Mask non-finite data, guard denominators, preserve `NaN` for undefined stats
- Test CLI, manifest, resume, and plugin entry-point boundaries
- Non-trivial runs need live progress (work counter + elapsed + ETA) and durable
  logs; `NullProgress()` is the explicit silence opt-out
- Real experiments need `DomainContract`, leak audit, held-out families,
  discovery loop, `assess_outcome`, preregistered decision rule, and stop rule

## Research scope

RDE is a conjecture factory, not a theorem prover. G4/G5 results are scoped to
the declared generator family, resource budget, and validation stage. Do not
generalize finite-size search outcomes to universal claims.

For recurring engineering corrections see
[`docs/engineering/agent-correction-playbook.md`](docs/engineering/agent-correction-playbook.md).
