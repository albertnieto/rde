# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m rde machine-profile --json
```

## Tests

```bash
python -m pytest tests/rde tests/rde_domains -q
python -m pytest tests/rde -m slow   # optional stress tests
```

## Packages

- **rde** — `src/rde/` (core library)
- **rde-domains** — `src/rde_domains/` (HSP + TSP plugins)

Build for PyPI from each package directory:

```bash
cd src/rde && python -m pip install build && python -m build
cd ../rde_domains && python -m pip install build && python -m build
```

## Agent workflow

Read `AGENTS.md` and `CLAUDE.md` before making changes. Skills live under
`.claude/skills/`.

## Code style

- Python 3.10+
- Line length 100 (ruff)
- Optimization-first for new hot paths
- Regression test with every behavior change
