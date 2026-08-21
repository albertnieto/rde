# RDE — Representation Discovery Engine

RDE is a domain-agnostic experimental-mathematics library. It generates
populations, measures structural descriptors, searches for representations and
protocols, persists reproducible artifacts, and grades the resulting evidence.
It does not require QUBO, quantum mechanics, or any particular domain.

The scientific vocabulary is one stack:

\[
P \longrightarrow Z \longrightarrow X \longrightarrow E \longrightarrow S
\longrightarrow O
\]

The two operating modes are forward discovery and reverse synthesis. Evidence
is graded G0–G5; these grades are not theorem claims.

## Install and run

From the standalone package root (or `src/rde/` in this repository):

```bash
cd src/rde  # omit when using the repo-root editable install
python -m pip install -e .
python -m rde list domains
python -m rde run --domain synthetic_poly --size 4 --n-instances 10 \
  --indices 0,1,2 --backend numpy --store-root rde_runs
```

The installed console command `rde` is equivalent to `python -m rde`.

## Minimal Python surface

```python
from rde import RunConfig, run_pipeline

config = RunConfig(
    domain_id="synthetic_poly",
    size=4,
    n_instances=10,
    indices=[0, 1, 2],
    store_root="rde_runs",
    compute_backend="numpy",
)
run_pipeline(config)
```

The public top-level API includes `Domain`, `Descriptor`, `Metric`,
`InstanceRecord`, `Registry`, `RunConfig`, `CampaignConfig`, `run_pipeline`,
`run_campaign`, `flatten_features`, and `summarize_run`. More specialized
search and analysis APIs live in their subpackages.

## Domains and plugins

The library ships only two small reference domains:

- `synthetic_poly`, for forward-pipeline examples and tests;
- `block_separable`, for Mode 2 reverse-synthesis examples.

External domains register through the `rde.domains` entry-point group. The core
package never imports a domain plugin package. This repository ships HSP and TSP
adapters as the sibling `rde-domains` distribution under `src/rde_domains/`.
Install the root editable package or `rde-domains` when an adapter is needed.

The standalone core test suite is `tests/rde/`. Adapter tests are kept
separately in `tests/rde_domains/` and are not part of the core distribution.

## Documentation

- [CLI and user guide](docs/README.md)
- [Methodology](docs/methodology.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Reverse synthesis](docs/hierarchical-synthesis.md)
- [Experiment playbook](docs/experiment-playbook.md)
- [Implementation chronology](docs/roadmap.md)

## Non-claims

RDE is a conjecture factory, not a theorem prover and not an automatic
polynomial-algorithm generator. A G4 or G5 result is still subject to the
declared family, resource budget, validation stage, and independent human
review.
