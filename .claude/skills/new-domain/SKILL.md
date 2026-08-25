---
name: new-domain
description: Add a new domain adapter to rde_domains — entry point, contract, metrics, tests, and README updates.
---

# New RDE domain adapter

Use when adding a domain beyond HSP/TSP to this repository.

## Checklist

1. **Package** — create `src/rde_domains/<slug>/` with `domain.py`, metrics,
   generators as needed. No imports from other domain packages unless shared
   utilities are extracted deliberately.
2. **Contract** — add `DomainContract` to `src/rde_domains/contracts.py` with
   leak-audited `FeatureSpec` entries (access model, predictors, targets,
   held-out families — in the contract itself, not a separate document).
3. **Entry point** — register in `src/rde_domains/plugins.py` and both
   `pyproject.toml` files (`project.entry-points."rde.domains"`).
4. **Tests** — add `tests/rde_domains/test_<slug>_*.py`; verify core suite still
   passes without importing the adapter.
5. **Optimization-first** — vectorize hot paths; MLX batching on Apple Silicon;
   NumPy reference parity tests.

## Verify

```bash
.venv/bin/python3 -m rde list domains
.venv/bin/python3 -m pytest tests/rde_domains/test_<slug>_*.py -q
.venv/bin/python3 -m pytest tests/rde/integration/test_no_rde_domains_import.py -q
```

## PyPI

Update `src/rde_domains/pyproject.toml` `packages` list and root entry points
when publishing the expanded adapter distribution.
