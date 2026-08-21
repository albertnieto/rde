# EXP-NNN Pre-registration — <title>

**Status:** pre-registered before the confirmatory run. The decision rules below
are fixed. A negative result that satisfies the null gate is a valid,
reportable outcome — not a reason to add knobs and re-run.

Algorithm card: ALGO-NNN. Domain contract: `<domain_id>` in
`src/rde/core/domain_contract.py`.

## Scientific question

<State the question so that both a positive and a null answer are informative.
Name the structure, the target, and what "generalizes" means here.>

## Population (a distribution, not a grid)

- **Instances:** one **distinct random instance per row**, varied by seed.
- **Sizes:** discovery <..>, validation <..>, confirmatory <..> (≥ 3 sizes).
- **Instances per size:** <N ≥ 50>.
- **Operator/mechanism label (`generator`):** <coarse groups>.
- **Held-out families:** <names> — a discovered relationship must generalize
  to these.
- **Target:** `metric.<...>`. **Predictors:** structural descriptors only
  (`matrix.*`, `graph.*`); all outcome-derived columns are dropped by the leak
  filter and this is enforced by the gate.

## Fixed decision rule (RDE gated outcomes)

Verdict is `assess_outcome(...)` on the merged multi-\(N\), leak-clean
population:

| Outcome | Condition | Interpretation |
|---|---|---|
| **NULL (grade 0)** | no null-language trigger: best \(|r|<0.35\), best expr \(R^2<0.40\), max latent corr \(<0.50\), cross-\(N\) stability \(<0.60\), generator separation \(<2.0\) | No predictability beyond known bounds. **Reportable negative. Stop.** |
| **SIGNAL (level ≥ 1)** | expr \(R^2\ge0.75\), \(>\) NR baseline \(+0.15\), extrapolation \(R^2\ge0.65\); or a cross-\(N\)-stable descriptor | Candidate law. Freeze candidate, then replicate. |
| **HIDDEN CLASS (level 2)** | bimodal separation \(\ge0.75\) or generator separation \(\ge2.0\) | Mechanism-family structure. Identify the family. |

## Stop rule

- **One** confirmatory run at the sizes above, with the full `run_discovery`
  loop (narrowing it requires stating why here, before the run).
- If **NULL**: record it in `results.md`; **do not** re-run with more
  instances, seeds, or sizes to chase a threshold. A new run requires a new
  pre-registered hypothesis.
- If **SIGNAL / HIDDEN CLASS**: freeze the top candidate and run the held-out
  family and replication checks before any claim.

## What would make this wrong

- A predictor that leaks the target (blocked by the gate's leak audit).
- A size covariate mistaken for structure — report top correlations.
- Re-running until a gate trips (forbidden above).
