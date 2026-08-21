---
name: new-experiment
description: Scaffold a new RDE experiment directory (EXP-NNN_<slug>) with README, run.py stub, PREREGISTRATION, and results.md. Enforces experiment-playbook gates for discovery experiments.
---

# New RDE experiment scaffolding

Use this skill to create `experiments/EXP-NNN_<slug>/` following
`src/rde/docs/experiment-playbook.md`.

## When to use

- Starting a new HSP or TSP discovery/validation experiment
- Any run that will call `run_discovery`, `assess_outcome`, or claim a G0–G5 grade

## Workflow

### 1. Determine next experiment number

Scan `experiments/EXP-*` and increment the three-digit ID.

### 2. Branch: RDE discovery experiment?

If the experiment discovers, predicts, or rules out structure, scaffold from
templates:

```bash
cp .claude/skills/new-experiment/templates/rde_discovery_run.py \
   experiments/EXP-NNN_<slug>/run.py
cp .claude/skills/new-experiment/templates/PREREGISTRATION.md \
   experiments/EXP-NNN_<slug>/PREREGISTRATION.md
```

Fill `DOMAIN_ID`, `TARGET`, `DEFAULT_SIZES`, and complete `PREREGISTRATION.md`
**before** running. The gate requires a `DomainContract`, held-out families, leak
audit, and full discovery loop.

### 3. Scaffold directory

Create `experiments/EXP-NNN_<slug>/` with:

- **README.md** — question, domain, charter link (`docs/research/`)
- **run.py** — executable; use RDE live progress + durable logs
- **results.md** — What was tested / Results / Interpretation
- **PREREGISTRATION.md** — for discovery experiments (from template)

### 4. Verify

```bash
ls -la experiments/EXP-NNN_<slug>/
.venv/bin/python3 -m pytest tests/rde/experiment/ -q  # if touching gate wiring
```

## Notes

- Slug is kebab-case from the experiment topic.
- A `run_pipeline`-only harness is a regression test, not a discovery experiment.
- Read the domain charter under `docs/research/` when working on HSP or TSP.
