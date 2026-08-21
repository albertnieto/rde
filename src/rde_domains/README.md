# RDE domain adapters (HSP + TSP)

Optional domain plugins for the domain-agnostic [`rde`](../rde/) library. This
distribution ships only:

- **HSP functions** (`hsp_functions`) — hidden-subgroup function discovery
  under a bounded-query access model;
- **TSP** — Euclidean TSP synthesis, cost-landscape, and landscape-statistics
  domains (`tsp_clustered`, `tsp_uniform_control`, `tsp_circulant_symmetry`,
  `tsp_cost_landscape`, `tsp_landscape_stats`).

Adapters register through the `rde.domains` entry-point group. The core library
never imports this package.

## Install

From the repository root (development umbrella):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q tests/rde tests/rde_domains
```

PyPI (when published):

```bash
pip install rde rde-domains
```

Plugin-only install from this directory:

```bash
pip install rde
pip install -e .
```

## Research charters

- [`docs/research/hidden-subgroup-function-discovery-charter.md`](../../docs/research/hidden-subgroup-function-discovery-charter.md)
- [`docs/research/tsp-novel-representation-discovery-charter.md`](../../docs/research/tsp-novel-representation-discovery-charter.md)

## Test boundary

- `tests/rde/` — core suite; must not import `rde_domains`.
- `tests/rde_domains/` — adapter suite for HSP and TSP entry points.
