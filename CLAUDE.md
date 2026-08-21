# RDE monorepo instructions

## Git and environment

Work on `main` directly. Do not create branches, PRs, or commits unless the
user explicitly asks. Preserve unrelated changes and never use destructive
reset/checkout commands.

Use the repository environment for every command:

```bash
.venv/bin/python3 -m rde machine-profile --json
.venv/bin/python3 -m pytest tests/
```

Never debug project dependencies with system Python. Before backend or GPU
claims, use the detected `profile_id`: `apple_silicon_mac` uses MLX as the
primary RDE backend; `intel_mac` uses NumPy/CPU.

## RDE library

RDE is the domain-agnostic core under `src/rde/`. Read its shipped
documentation before RDE work:

1. [`src/rde/docs/methodology.md`](src/rde/docs/methodology.md)
2. [`src/rde/docs/ARCHITECTURE.md`](src/rde/docs/ARCHITECTURE.md)
3. [`src/rde/docs/experiment-playbook.md`](src/rde/docs/experiment-playbook.md)
4. [`src/rde/docs/README.md`](src/rde/docs/README.md)

The canonical science stack is PZXESO, with forward discovery and reverse
synthesis as the only modes and G0–G5 as the outcome vocabulary. Core RDE must
not import domain plugins; adapters load through `rde.domains` entry points.

`src/rde/pyproject.toml` and `src/rde_domains/pyproject.toml` are separate
PyPI distributions. The root `pyproject.toml` is the development umbrella.

Core tests: `tests/rde/`. Adapter tests: `tests/rde_domains/`. The core suite
must remain runnable without importing `rde_domains`.

## Domain adapters (this repo)

This repository ships only HSP and TSP adapters under `src/rde_domains/`:

- `hsp_functions`
- `tsp_clustered`, `tsp_uniform_control`, `tsp_circulant_symmetry`,
  `tsp_cost_landscape`, `tsp_landscape_stats`

There is no QPFA, QUBO synthesis, coined-walk, or instrument-codesign code here.

## Engineering rules

- Optimization-first: vectorize, batch, cache, workers where safe
- Backend/dtype/shape/memory/numerical semantics are contracts — add regression tests
- Mask non-finite data, guard denominators, reject constant inputs, keep `NaN`
- Test actual CLI/parser, plugin, report, manifest, and storage boundaries
- Non-trivial runs: live genuine progress + durable structured logs
- Real experiments: `DomainContract`, leak audit, held-out families, discovery
  loop, `assess_outcome`, preregistered decision rule, stop rule

For recurring corrections see
[`docs/engineering/agent-correction-playbook.md`](docs/engineering/agent-correction-playbook.md).
