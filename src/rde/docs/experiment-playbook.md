# RDE Experiment Playbook

**How to run a *real* RDE experiment — and never ship a trivial one again.**

This is the durable, authoritative guide for coordinating an experiment with
the Representation Discovery Engine. It exists because a "discovery campaign"
was once shipped that reused **one landscape per size**, used the thing under
study as the `generator` label, never called RDE's discovery or gating loop,
and finished in three seconds. That is a **parity/regression harness wearing
an experiment's name**, not a discovery. Do not do that again.

Read this **before** writing any `experiments/EXP-NNN_*/run.py` that claims to
*discover*, *rediscover*, *predict*, or *rule out* anything with RDE.

Related, mandatory reading:

- [`methodology.md`](methodology.md) —
  **canonical science vocabulary** (\(P\to Z\to X\to E\to S\to O\), Mode 1 /
  Mode 2, grades G0–G5). Read this first.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) —
  **code pipeline** that implements that stack (worker cache, fill vs search,
  folder map). Skipping both and going straight to a single CLI verb is how
  Direction E (TSP) spent a full session on Mode 2 `rde synthesize` while
  Mode 1 (descriptors → ranker → symbolic → representation search) never ran,
  because `primitive_features()` exposed three mechanism scalars instead of
  raw arrays.
- [`../../../docs/engineering/agent-correction-playbook.md`](../../../docs/engineering/agent-correction-playbook.md) — recurring
  engineering corrections.
- [`.cursor/rules/optimization-first-architecture.mdc`](../../../.cursor/rules/optimization-first-architecture.mdc)
  — the performance design bar (GPU/vectorize/batch/workers from commit one).
- [`.cursor/rules/rde-performance-guardrails.mdc`](../../../.cursor/rules/rde-performance-guardrails.mdc)
  — backend routing and numerical contracts.
- [`README.md`](README.md) — CLI commands and
  flags only.
- [`.claude/skills/dev-machine-profile/SKILL.md`](../../../.claude/skills/dev-machine-profile/SKILL.md)
  — session bootstrap.

The worked reference implementation for everything below is
**EXP-050** (`experiments/EXP-050_coin_shift_grammar_rediscovery/`): its
`PREREGISTRATION.md`, `run.py`, and `results.md` are the template.

---

## 0. Session bootstrap (do this first, every session)

```bash
.venv/bin/python3 -m rde machine-profile --json
```

Bind backend policy to the reported `profile_id`:

- `apple_silicon_mac` → **MLX** (`--backend mlx` or `auto`; aliases
  `mps`/`gpu`/`metal` → MLX, never PyTorch MPS).
- `intel_mac` → **numpy/CPU** (`mlx` is excluded by platform marker; expected).

Never assume which of the two dev Macs you are on from prose in the docs. Never
claim "MLX doesn't apply here" without having run the command above.

---

## 1. What an RDE experiment *is* (and what it is not)

An RDE experiment tests whether some **structure** predicts, separates, or
generates a **target** across a **population**, and does so under controls that
make a false positive hard and a null informative.

| A real RDE experiment has… | A joke has instead… |
|---|---|
| A **population**: many independent random instances per size | One landscape reused across a hand-picked grid |
| A **domain contract**: primary target, predictors, held-out groups | No contract; ad-hoc columns |
| **Leak-audited predictors** (structure only) | Outcome columns that reconstruct the target at \(r=1\) |
| A **generator label** = the mechanism under test, with **held-out families** | The thing under study used as the label with nothing held out |
| The **discovery loop** (`run_discovery` or the ranker + latent + symbolic) | `run_pipeline` only, then eyeballing |
| **Gated outcome** (`assess_outcome`) against a **pre-registered** rule | A verdict invented after seeing the numbers |
| A **stop rule** | "let's try more specs/seeds/sizes until something trips" |
| **Cross-\(N\)** and **held-out** generalization checks | A single-size discovery-split correlation |

If your `run.py` does not import from `rde.analyze.outcome`,
`rde.core.domain_contract`, and `rde.analyze.splits`, it is almost certainly a
joke. Stop and fix it.

---

## 1b. This is enforced in code, not by good intentions

Guidance did not work: this playbook was written, and the very next run still
skipped the discovery loop. So the bar is now **mechanical**.

- **`rde.experiment.ExperimentGate`** raises unless the run meets the bar. It
  checks the contract, the pre-registration, size/instance counts, population
  variety (distinct structural feature vectors per size — the one thing a
  reused landscape cannot fake), held-out rows, the leak audit, and that the
  required discovery phases actually ran. Only then does `finalize()` write
  `receipt.json`.
- **`tests/rde/test_experiment_receipts.py`** fails the suite when an
  experiment ships a `results.md` claiming a verdict without a valid receipt.
  Shortcutting is a red test, not a self-report.
- **The `new-experiment` scaffold** emits a `run.py` with the gate and the full
  `run_discovery` already wired, so the correct path is the default one and
  shortcutting requires deleting code.
- **Receipts record work done** — instances, distinct instances, ranked
  candidates, phases — and never wall-clock, so neither a fast run nor a padded
  one can be mistaken for rigor.

Non-discovery experiments (analytic/parity profiles) opt out with a `NO_RECEIPT`
file stating why. That file is for work the gate genuinely does not apply to —
not an escape hatch for a discovery run.

## 2. Definition of Done (the checklist)

An experiment is **not done** until every box is checked. The gate enforces most
of them; the rest are on you. "Done" claimed without these is the exact failure
this document exists to prevent.

- [ ] `rde machine-profile` was run; backend chosen accordingly.
- [ ] A `DomainContract` exists for the domain (in
      `src/rde/core/domain_contract.py`), with `primary_target`, structural
      `feature_specs`, `held_out_generator_groups`, and staged sizes.
- [ ] The domain generates a **distribution** — many independent random
      instances per size (not one shared landscape), each with its own seed.
- [ ] The domain exposes its **input primitives** (e.g. `Q`) from
      `primitive_features` so `matrix.*`/`graph.*` structural descriptors fire.
- [ ] `generator` labels the **mechanism under test**; at least one family is
      **held out** of discovery.
- [ ] **Leak audit**: only structural predictors reach the target. Confirm the
      raw rows leak (\(r \approx 1\) via an outcome column) and the cleaned rows
      do not. Outcome-derived columns are dropped.
- [ ] A `PREREGISTRATION.md` with a **fixed** NULL/SIGNAL decision rule and a
      **stop rule**, committed **before** the confirmatory run.
- [ ] The **discovery loop** runs (expression ranker at minimum; `run_discovery`
      for the full path) and feeds `assess_outcome`.
- [ ] **Cross-\(N\)** (sizes span ≥ 3 values) and **held-out-family**
      generalization are both reported.
- [ ] **Optimization-first**: hot paths are vectorized / MLX-batched; no Python
      loop over \(2^N\), rows, or candidates; instance batching is wired.
- [ ] **Live progress + durable logs**: RDE core's default TTY UI or flushed
      newline reporter must show overall and per-task work counters with
      elapsed time and ETA, including under `tee`/CI, and persist JSONL logs.
      A size-only `tqdm` bar or end-of-stage print is not sufficient.
- [ ] `results.md` states the verdict against the pre-registered rule, including
      a **negative** if that is what happened.
- [ ] Tests: parity/regression, leak-clean assertion, contract presence,
      population distinctness. Focused suite green under `.venv`.

---

## 3. Build it, step by step

### 3.1 Domain contract

Add a `DomainContract` (see `coined_walk_contract()` in
`src/rde/core/domain_contract.py`). It declares:

- `primary_target` (an outcome metric, e.g. `metric.E_peak`),
- `feature_specs` — the **structural** predictors (`matrix.Q.*`, `graph.Q.*`);
  mark them `POLYNOMIAL_INPUT`, and mark metrics `OUTCOME`,
- `held_out_generator_groups` — mechanism families withheld from discovery,
- `StageSizePolicy` — calibration / discovery / validation / confirmatory sizes
  (respect the brute-force cap \(N \le 14\)),
- `recurrence_applicable` / `representation_applicable` — set honestly.

Register it in the `_CONTRACTS` dict.

### 3.2 A real population (a distribution, not a grid)

Generate **one independent random instance per row**, varied by seed, and
assign the mechanism under test round-robin. Reusing a single landscape across
a categorical grid gives you \(N=1\) statistics — useless for discovery.

```python
def _population_instances(self, n, size, seed):
    grid = self.specs or all_valid_specs()
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)          # distinct draw per row
        Q = random_qubo(size, rng)
        spec = grid[i % len(grid)]                       # mechanism, round-robin
        out.append(InstanceRecord(
            domain_id=self.domain_id, size=size, seed=seed + i,
            params={"Q": Q.tolist(), "generator": grammar_family_group(spec),
                    **spec.to_params()},
        ))
    return out
```

### 3.3 Expose input primitives for descriptors

Return the raw input array from `primitive_features` so RDE's instance
descriptors compute `matrix.<name>.*` / `graph.<name>.*` automatically (see
`src/rde/runtime/instance_descriptors.py`):

```python
def primitive_features(self, instance, *, cache=None):
    ctx = self._build_cache(instance, cache)
    out = {"E_peak": ctx["E_peak"], ...}
    Q = np.asarray(ctx["Q"], float)
    if Q.ndim == 2 and Q.shape[0] == Q.shape[1] and np.any(Q):
        out["Q"] = Q          # 2-D square array -> matrix + graph descriptors
    return out
```

### 3.4 Generator labels + held-out families

The `generator` field must label the **mechanism you are testing**, grouped
coarsely enough that generator-separation and held-out generalization are
meaningful (not 29 singleton points). Hold out ≥ 1 family in the contract; a
discovered relationship must generalize to families it never saw.

### 3.5 Leak audit — the step that makes a null mean something

Any column that is a function of the target will reconstruct it at \(r=1\)
(e.g. \(\text{margin}=D-E_{\text{peak}}\)). Before correlating or ranking,
**drop every outcome-derived column** and keep only leak-clean structural
predictors + target + metadata. Prove it in a test:

```python
raw  = correlate_with_target(rows,        target, min_abs_r=0.0)
clean = correlate_with_target(clean_rows, target, min_abs_r=0.0)
assert max(abs(h["pearson_r"]) for h in raw)   > 0.99   # raw leaks
assert max(abs(h["pearson_r"]) for h in clean) < 0.99   # cleaned does not
```

Watch for **covariate leaks** too: `size`, `n_nodes`, `avg_degree`, and
`laplacian_*` are all proxies for \(N\). Report the top correlated column so a
size covariate cannot masquerade as a structural discovery.

### 3.6 Pre-registration + stop rule

Write `PREREGISTRATION.md` **before** the confirmatory run, fixing the decision
rule to RDE's gate thresholds (`assess_outcome`):

- **NULL (grade 0)** — no null-language trigger: best \(|r|<0.35\), best expr
  \(R^2<0.40\), max latent corr \(<0.50\), cross-\(N\) sign stability \(<0.60\),
  generator separation \(<2.0\).
- **SIGNAL (level ≥ 1)** — predictor with expr \(R^2 \ge 0.75\), \(>\) NR
  baseline \(+0.15\), extrapolation \(R^2 \ge 0.65\); or a cross-\(N\)-stable
  descriptor.
- **HIDDEN CLASS (level 2)** — bimodal target separation \(\ge 0.75\) or
  generator separation \(\ge 2.0\).

**Stop rule:** one confirmatory run. A null that clears the gate is a valid,
reportable result — **do not** re-run with more specs/seeds/sizes to chase a
threshold. A further run requires a *new* pre-registered hypothesis.

### 3.7 Discovery loop + gated outcome

Run the real machinery, not eyeballing:

```python
from rde.analyze.outcome import assess_outcome
from rde.analyze.ranker import ConjectureRanker
from rde.expression.generators import enumerate_metric_candidates, metric_variable_columns

vars_ = metric_variable_columns(discovery_rows, target, max_vars=16)
cands = enumerate_metric_candidates(vars_, max_depth=2, max_candidates=20_000)
ranker = ConjectureRanker(target_column=target, min_abs_r=0.20, max_results=20)
conjectures = ranker.rank_expressions(discovery_rows, list(cands), variables=vars_)

assessment = assess_outcome(clean_rows, target,
                            metric_candidates=[...],   # from conjectures
                            domain_contract=contract)
verdict = "SIGNAL" if assessment.grade >= 1 else "NULL"
```

For the full staged path (calibration → confirmatory with checkpoints, latent,
symbolic, phase-6, leak audit, replication), use `rde science-ledger` /
`run_discovery` once the contract is in place, rather than reimplementing it.

Assign splits from the contract so held-out families are excluded from
discovery (`rde.analyze.splits.SplitPolicy` / `assign_instance_fold`), and
report held-out-family correlation separately.

---

## 4. Optimization-first — non-negotiable, from the first commit

New domains, evaluators, and campaign hot paths are **performance-designed on
day one**, not "optimized later". This is why the last joke was doubly bad: it
was trivial *and* it round-tripped MLX through NumPy every step.

- **Machine profile** picks the backend; MLX on `apple_silicon_mac`.
- **Vectorize.** No Python loop over \(2^N\), state dimensions, rows, or
  candidates. Small-\(N\) setup, symbolic trees, and dependency-ordered work are
  the only allowed exceptions, and must be labelled as such.
- **Batch on device.** Group compatible instances so `prepare_instances_batch`
  and pipeline `batch_size` reuse work. Cache device tensors (Q, coin blocks,
  flip indices, costs) — see the `_MlxWalkCache` pattern in
  `src/rde_domains/coined_walk/evolution.py`. **No MLX↔NumPy round-trip inside a
  step loop.**
- **Workers** where process pools are safe; pre-batch GPU work in the parent.
- **Parity tests** for every fast path (NumPy reference + MLX when available),
  and record requested-vs-effective backend when fallback occurs.

Route enumeration and expression evaluation through `rde.backends.resolve` and
`rde.expression.batch.prepare_device_envs`. Reuse cached \(2^N\) values; never
re-enumerate them through a NumPy helper.

If a prototype is intentionally scalar, label it `theoretical-only` and keep it
off the campaign path until the optimized contract exists.

---

## 5. Progress + durable logging (a correctness requirement)

Any run longer than a few seconds must show **real live** progress and persist
**durable** logs — so a slow-but-working run is distinguishable from a hung
one, and status survives the terminal.

- "Real" = a genuine counter over completed work (instances, seeds, sizes,
  candidates), with elapsed time and ETA. **Never** a time-based or fake bar.
- RDE core automatically attaches its sticky Rich TTY UI or flushed newline
  reporter when callers omit `progress=` / `on_progress=`. Explicit
  `NullProgress()` is the only silence opt-out.
- Experiment scripts must use the core nested campaign/task progress helper,
  plus JSONL per-stage summaries and a `tee`'d log:

```bash
.venv/bin/python3 experiments/EXP-050_.../run.py --backend auto \
  2>&1 | tee experiments/EXP-050_.../runs/confirmatory_$(date +%Y%m%d_%H%M%S).log
```

---

## 6. Worked example — EXP-050

`experiments/EXP-050_coin_shift_grammar_rediscovery/` is the canonical
template. It:

1. Verifies the grammar↔harness parity (regression floor).
2. Builds a **480-instance random-QUBO population** across \(N\in\{6,8,10,12\}\),
   labels each by operator-mechanism family, holds out `coin_noninvolutory` and
   `shift_holonomy`, and exposes \(Q\) to structural descriptors.
3. **Leak-cleans** the rows (raw rows reconstruct the target at \(r=1\) via
   `margin_T1`; cleaned rows keep only `matrix.Q.*`/`graph.Q.*`).
4. Runs the expression ranker on the discovery split and `assess_outcome` with
   the pre-registered gate.
5. Emits a verdict, top correlations (to expose size covariates), held-out-family
   generalization, `tqdm` progress, and durable JSONL + `tee` logs.

Its confirmatory result was an honest **documented negative** (grade 0): the
only correlations were the trivial size covariate; the best structural
descriptor reached \(r=0.26\), below the 0.35 gate, with no held-out
generalization. That is a legitimate outcome — and, per the stop rule, it is
**not** re-run with more knobs.

---

## 7. Anti-patterns that produce a "joke" (memorize these)

1. **One landscape per size.** \(N=1\) statistics. Use a distribution.
2. **The label *is* the thing under study, nothing held out.** No
   generalization test is possible.
3. **`run_pipeline` only.** No discovery, no gate — cannot discover or reject.
4. **No contract.** Incompatible with science-ledger / gates; targets undefined.
5. **Outcome columns as predictors.** Trivial \(r=1\); the "result" is an
   identity. Always leak-audit.
6. **Size covariate mistaken for structure.** `n_nodes`/`avg_degree`/`size` are
   \(N\). Report top correlations and check held-out + cross-\(N\).
7. **Verdict invented after seeing numbers.** Pre-register the rule.
8. **Re-running until a gate trips.** Forbidden by the stop rule.
9. **Scalar / NumPy round-tripping hot loop on Apple Silicon.** Vectorize and
   cache device tensors; batch.
10. **"Done" with no durable log or real progress.** Not done.

When in doubt: it is better to report an honest, well-controlled **negative**
than to ship a fast, trivial "success". A negative that clears the design bar is
a result. A three-second "campaign" is not.
