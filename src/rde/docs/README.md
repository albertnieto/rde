# RDE CLI guide

The CLI is the reproducible command-line surface for the RDE library. Run it
as `python -m rde` or, after installation, as `rde`.

## Reference domains

The standalone library includes two small reference domains:

```bash
python -m rde list domains
```

- `synthetic_poly` is a forward-discovery toy.
- `block_separable` is a reverse-synthesis toy.

Additional domains are optional `rde.domains` plugins. They are not imported
by the core package.

## Fill a store

```bash
python -m rde run \
  --domain synthetic_poly \
  --size 4 \
  --n-instances 20 \
  --indices 0,1,2 \
  --backend numpy \
  --store-root rde_runs
```

For a multi-size population:

```bash
python -m rde campaign \
  --domain synthetic_poly \
  --sizes 3,4,5 \
  --n-per-size 20 \
  --workers 1 \
  --backend numpy \
  --store-root rde_runs
```

Use `--backend auto` for the size-aware default. On Apple Silicon with a
working Metal runtime, auto may select MLX; NumPy remains the reference path.
Long runs support Rich progress on a TTY and newline progress with
`--log-progress` or `RDE_LOG_PROGRESS=1`.

## Inspect and search

```bash
python -m rde summary --run-id <run_id> --store-root rde_runs
python -m rde export --run-id <run_id> --output features.csv --store-root rde_runs
python -m rde discover --run-id <run_id> --store-root rde_runs
python -m rde latent --run-id <run_id> --store-root rde_runs
python -m rde represent --run-id <run_id> --store-root rde_runs
python -m rde rank-metrics --run-id <run_id> --target metric.target --store-root rde_runs
python -m rde rank-desc --run-id <run_id> --target metric.target --store-root rde_runs
```

Search reuses durable measurements. It does not silently regenerate a
campaign or treat a failed stage as a null result; reports contain stage
errors and checkpoint metadata.

## Additional commands

Not exhaustive prose above every flag — `--help` on any of these is
authoritative. Grouped by what they act on:

Discovery-adjacent (Phase 2b/4/5, `src/rde/discovery/`,
`src/rde/expression/`):

```bash
python -m rde discover-symbolic --run-id <run_id> --store-root rde_runs   # symbolic regression + latent interpretation
python -m rde rank-expr --run-id <run_id> --target metric.target --store-root rde_runs  # rank expression-DSL candidates
python -m rde retain-topk --run-id <run_id> --store-root rde_runs         # keep targets + ranked columns only
```

Outcome gating (G0–G5, `src/rde/analyze/outcome.py`):

```bash
python -m rde outcome --run-id <run_id> --store-root rde_runs   # assess pre-registered G0 vs G1 outcome
python -m rde audit --run-id <run_id> --store-root rde_runs     # leak/tautology audit on promoted conjectures
python -m rde obstruct --run-id <run_id> --store-root rde_runs  # obstruction witness panel + lower-bound conjectures
python -m rde certify --run-id <run_id> --store-root rde_runs   # G5 candidate resource/query pre-check
```

Representation Core (`src/rde/representation/`, not campaign features —
see [Architecture](ARCHITECTURE.md)):

```bash
python -m rde repr-rank --n 8 --samples 8 --pattern random          # rank the fixed grammar against a demo batch
python -m rde repr-rank --input batch.npy                           # ...or a real (samples, n) .npy batch
python -m rde repr-rank-run --run-id <run_id> --array-key <key> --store-root rde_runs  # rank against a stored array field
```

Store/hardware/schema utilities:

```bash
python -m rde analyze --run-id <run_id> --store-root rde_runs         # feature-table analysis
python -m rde calibrate --run-id <run_id> --store-root rde_runs       # calibrate complexity metric across instance groups
python -m rde validate --run-id <run_id> --store-root rde_runs        # validate features.jsonl schema
python -m rde stress-store --store-root rde_runs                      # synthetic JSONL write benchmark
python -m rde power-plan --campaign-id <campaign_id> --store-root rde_runs   # prospective power planning from campaign data
python -m rde calibrate-hardware                                      # benchmark a declared RDE resource profile
```

## Reverse synthesis

Mode 2 starts from a resource target and searches algorithm skeletons:

```bash
python -m rde synthesize \
  --domain block_separable \
  --size 6 \
  --n-instances 10 \
  --target-degree 2
```

The reference toy should accept a decomposition-and-merge skeleton and reject
whole-instance brute force on complexity grounds. This is a machinery/V1 toy
result, not a theorem or a general optimization algorithm.

## Storage and reproducibility

Each run writes a manifest, instance records, feature rows, and optional array
sidecars under `rde_runs/`. Use `seal` to create a verified compact snapshot:

```bash
python -m rde seal --campaign-id <campaign_id> --store-root rde_runs
```

The store records requested/effective backends, configuration fingerprints,
progress, errors, and provenance. Preserve the store with the experiment
record when making a scientific claim.

## Backend diagnostics

```bash
python -m rde machine-profile --json
python -m rde backends
```

`torch_cpu` and `torch_mps` are explicit compatibility backends, not defaults.
Optional symbolic engines can be installed with the package's symbolic extra.

## Domain contract

A domain implements `rde.core.protocols.Domain`, which declares:

- `domain_id`;
- `generate(n, size, seed)`;
- `materialize(instance, index)`;
- `primitive_features(instance)`.

Two more hooks are runtime conventions, not part of the `Domain` `Protocol`
class itself — `runtime/worker.py` checks for them with `hasattr`/
`inspect.signature` rather than the type system:

- optional `prepare_instance(instance, indices=...)`, called once when
  `materialize` and `primitive_features` would otherwise recompute the same
  expensive primitive;
- when `prepare_instance` exists, its result is passed as `cache=...` to
  `materialize`/`primitive_features` if their own signature accepts it.

Plugins register a callable in the `rde.domains` entry-point group. The
callable receives a `Registry` and registers its domain, descriptors, metrics,
and any domain generators.

## Science and engineering

- [Methodology](methodology.md) defines PZXESO, the two modes, G0–G5, and V1–V4.
- [Architecture](ARCHITECTURE.md) maps the methodology onto code and the cache contract.
- [Reverse synthesis](hierarchical-synthesis.md) documents Mode 2 compilation depth.
- [Experiment playbook](experiment-playbook.md) defines population, leakage, gating,
  holdout, stop-rule, progress, and reproducibility requirements.
- [Roadmap](roadmap.md) records implementation chronology.

RDE generates conjectures and scoped negatives. It does not prove theorems or
automatically establish polynomial algorithms.
