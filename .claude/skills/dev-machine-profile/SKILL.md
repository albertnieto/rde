---
name: dev-machine-profile
description: Run at the start of every RDE session before backend, GPU, MLX, workers, or performance claims. Detects Apple Silicon vs Intel Mac via rde machine-profile.
---

# Dev machine profile (session bootstrap)

## When to use

**First executable step** in any new agent session when work may touch RDE
backends, experiment runtime, or performance.

Do **not** infer the machine from docs or conversation — detect every session.

## Required command

From repo root:

```bash
.venv/bin/python3 -m rde machine-profile --json
```

## Interpretation

| `profile_id` | Machine | Default compute |
|---|---|---|
| `apple_silicon_mac` | arm64 + Metal | `mlx` or `auto` |
| `intel_mac` | x86_64 | `numpy` |
| `other` | non-mac / unknown | follow JSON |

## Hard rules

1. Never claim MLX unavailable on `apple_silicon_mac` with `mlx_usable: true`
   without re-running the command.
2. Never default to `--backend numpy` on Apple Silicon unless benchmarking CPU
   reference or Metal is actually unavailable.
3. If sandbox blocks Metal on arm64, re-run outside sandbox before concluding.
