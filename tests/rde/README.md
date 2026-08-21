# Core RDE tests

This suite belongs to the standalone `rde` distribution. It must run without
importing `rde_domains` or any QPFA implementation module.

Run it from the repository root with:

```bash
.venv/bin/python3 -m pytest -q tests/rde
```

When `rde` is extracted, copy this directory with `src/rde/`. Adapter tests
belong in `tests/rde_domains/` and are intentionally excluded.
