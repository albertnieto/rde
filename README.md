# RDE — Representation Discovery Engine

Domain-agnostic experimental-mathematics library for forward discovery and
reverse synthesis, with optional **HSP** and **TSP** domain adapters.

This repository is self-contained: it ships the full RDE core (`rde`) and a
trimmed plugin distribution (`rde-domains`) with no QPFA or quantum-simulation
dependencies.

## Packages

| Package | Path | PyPI name (planned) |
|---------|------|---------------------|
| Core library | `src/rde/` | `rde` |
| Domain adapters | `src/rde_domains/` | `rde-domains` |

Core ships two reference toy domains (`synthetic_poly`, `block_separable`).
Adapters register through the `rde.domains` entry-point group:

- `hsp_functions` — hidden-subgroup function discovery (bounded-query model)
- `tsp_clustered`, `tsp_uniform_control`, `tsp_circulant_symmetry`,
  `tsp_cost_landscape`, `tsp_landscape_stats` — Euclidean TSP domains

## Quick start

```bash
git clone https://github.com/albertnieto/rde.git
cd rde
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .

# Verify machine profile (Apple Silicon → MLX; Intel Mac → NumPy)
python -m rde machine-profile --json

# List registered domains
python -m rde list domains

# Smoke run (toy domain)
python -m rde run --domain synthetic_poly --size 4 --n-instances 10 \
  --indices 0,1,2 --backend numpy --store-root rde_runs

# Run tests
python -m pytest -q tests/rde tests/rde_domains
```

## Documentation

In-package docs (authoritative for science and engineering contracts):

- [`src/rde/docs/README.md`](src/rde/docs/README.md) — CLI guide
- [`src/rde/docs/methodology.md`](src/rde/docs/methodology.md) — PZXESO, G0–G5
- [`src/rde/docs/ARCHITECTURE.md`](src/rde/docs/ARCHITECTURE.md) — pipeline
- [`src/rde/docs/experiment-playbook.md`](src/rde/docs/experiment-playbook.md)

Domain research charters:

- [`docs/research/hidden-subgroup-function-discovery-charter.md`](docs/research/hidden-subgroup-function-discovery-charter.md)
- [`docs/research/tsp-novel-representation-discovery-charter.md`](docs/research/tsp-novel-representation-discovery-charter.md)

Agent workflow:

- [`AGENTS.md`](AGENTS.md) — repository agent guide
- [`CLAUDE.md`](CLAUDE.md) — environment and session policy
- [`.claude/skills/`](.claude/skills/) — agent skills

## Publishing (PyPI)

Each package has its own `pyproject.toml` under `src/rde/` and
`src/rde_domains/`. Build and publish independently:

```bash
cd src/rde && python -m build && python -m twine upload dist/*
cd ../rde_domains && python -m build && python -m twine upload dist/*
```

The root `pyproject.toml` is the development umbrella for editable installs
and CI; it is not the PyPI artifact name.

## License

MIT — see [`LICENSE`](LICENSE) and per-package LICENSE files.
