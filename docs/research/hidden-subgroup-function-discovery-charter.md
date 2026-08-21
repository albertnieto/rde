# Hidden-subgroup function-family discovery charter (Direction F)

**Status date:** August 2026.

**Status tag:** `active-research-framing` — not a theorem, algorithm card, or
experiment. Like [the Direction E charter](tsp-novel-representation-discovery-charter.md),
this is a **methodology guardrail document first, research plan second**: §1
exists because the adjacent Direction E pivot (TSP + planted abelian
symmetry) was tried, formalized, and then **closed negative** for a specific,
precise reason — and that reason is exactly the trap this charter is
written to keep Direction F out of.

**Relation to Direction E:** independent, not a continuation. Direction E's
abelian/dihedral-HSP pivot is closed for *standard* TSP (see
[roadmap.md](../roadmap.md) Direction E, item 7): once the full distance
matrix is handed to you, recognizing it as a relabeled circulant matrix is
already solvable in polynomial time **classically** (Muzychuk's theorem,
CLAIM-203), so there is no computational bottleneck left for a quantum
algorithm to beat. Direction E item 8(c) names the one thing that closure
does *not* foreclose: a genuinely **query-limited** reformulation. Direction
F takes that seriously but does not tie it to TSP — it generalizes the
question to its natural home, the hidden subgroup problem (HSP) itself, and
asks it directly instead of routing through a specific NP-hard optimization
problem that happens to have an unrelated closed-form classical answer.

**Related docs:** [roadmap.md](../roadmap.md) Direction E (the closed
precedent and why),
[literature/audits/2026-08-19_abelian-hsp-vs-nonabelian-tsp-symmetry-novelty-check.md](../../literature/audits/2026-08-19_abelian-hsp-vs-nonabelian-tsp-symmetry-novelty-check.md),
[literature/audits/2026-08-19_tsp-hsp-reduction-and-noisy-hsp-robustness.md](../../literature/audits/2026-08-19_tsp-hsp-reduction-and-noisy-hsp-robustness.md)
(CLAIM-188–203 — the abelian/non-abelian HSP hardness landscape, noisy-HSP
lineage, and DCP degradation curve this charter builds on),
[src/rde/docs/ARCHITECTURE.md](../../src/rde/docs/ARCHITECTURE.md),
[experiment-playbook.md](../../src/rde/docs/experiment-playbook.md).

---

## 1. The rule this charter exists to enforce

**RDE's job on this track is to flag candidate structural signatures of an
oracle function that correlate with a genuine exponential quantum/classical
*query-complexity* gap — across a wide, systematically-generated family of
functions — not to prove a new quantum algorithm exists, and not to search
inside an access model where the "hardness" it is chasing has already been
shown not to exist.**

Concretely, two failure modes to not repeat:

1. **Don't reuse the closed access model.** Direction E's closure turned
   entirely on *access model*: full, explicit data (the distance matrix) in
   hand up front. The entire reason Simon's, Shor's, and Kuperberg's
   algorithms beat classical computation is that they operate in a
   **query/oracle model** — the input is a black box you may query a bounded
   number of times, and the classical lower bound is a lower bound on
   *queries*, not on time given the whole truth table. THEOREM-014 already
   proved the general version of this lesson for QPFA's own setting: an
   \(N\)-variable QUBO's exact-value-query complexity is \(O(N^2)\) — full
   value-access kills exponential query lower bounds outright. Direction F's
   **fixed contract** (§4) makes the oracle/query model non-negotiable for
   exactly this reason — it is not a stylistic choice, it is the one
   precondition every known exponential HSP-style separation depends on.
2. **Don't report a rediscovery as a finding.** Simon (\(G=\mathbb Z_2^n\)),
   Shor/period-finding (\(G=\mathbb Z_N\)), and Kuperberg's dihedral
   algorithm (\(G=D_N\)) are **known, held-out validation cases** (§5) — used
   to confirm the pipeline finds real signal, never reported as this track's
   output. A discovered structural predictor that turns out to just be
   re-detecting "this function is abelian" is a successful pipeline check,
   not a result.

RDE itself cannot prove a new quantum algorithm correct — that is a
`THEOREM-NNN`/`ALGO-NNN` job for a human afterward, gated the same way every
other original result in this repo is gated (see §6's outcome-level mapping).
What RDE *can* legitimately do is what it already does elsewhere in this
repo: search a large space of parametrized instance families for structural
descriptors that predict a numeric target, flag the strongest and most
novel-looking hits, and force a literature novelty check before anything is
called new. That is the actual, honest scope of this charter.

---

## 2. Why this is decoupled from TSP (and from any single NP-hard problem)

Direction E asked "does TSP have hidden abelian structure a quantum
algorithm could exploit?" — a single-problem question that turned out to
have a single-problem answer (no, for the reason in §1). The right level to
ask the underlying question is the HSP itself: **for which groups \(G\), and
which structural signatures of an oracle function \(f\) hiding a subgroup
\(K\le G\), does Fourier sampling (or a Kuperberg-style non-Fourier-sampling
algorithm) recover \(K\) in \(\mathrm{poly}(\log|G|)\) queries, when neither
\(G\) nor \(K\) is told to the algorithm in advance?**

The literature audits already on record (CLAIM-188–203, done 2026-08-19)
establish the known landscape this charter treats as ground truth, not as a
search target:

- **All finite abelian \(G\):** Fourier sampling works, uniformly, in
  \(O(\log|G|)\) queries (Kitaev 1995 / CLAIM-188; Mosca–Ekert 1999 /
  CLAIM-189). Simon (\(\mathbb Z_2^n\)) and Shor's period-finding
  (\(\mathbb Z_N\)) are the two textbook special cases of one mechanism. The
  standard explanation is representation-theoretic and not attributable to a
  single citation (flagged the same way in CLAIM-189's own Notes): abelian
  groups have only 1-dimensional irreducible representations, so a coset
  state's Fourier-sampled measurement outcome is already enough information
  to pin down \(K\) with polynomially many samples.
- **\(S_n\) (graph-isomorphism-relevant):** weak Fourier sampling
  (Grigni–Schulman–Vazirani–Vazirani 2001 / CLAIM-190) and even strong
  Fourier sampling with arbitrary, jointly-entangled POVMs across multiple
  coset-state copies (Moore–Russell–Schulman 2005, Parts I/II / CLAIM-191,
  CLAIM-192) **provably fail** — exponentially little information, no matter
  the measurement strategy. \(S_n\)'s irreps grow to dimension
  \(\sim\sqrt{n!}\); this is the standard explanation for why the mechanism
  above breaks down.
- **\(D_N\) (dihedral):** genuinely intermediate. Non-abelian, so plain
  Fourier sampling does not apply, but Kuperberg's algorithm (2003 /
  CLAIM-193) — a different, non-Fourier-sampling mechanism (a sieve over the
  hidden reflection) — solves it in \(2^{O(\sqrt{\log N})}\) queries:
  subexponential, not polynomial, but a real, nontrivial win over the
  classical query complexity. Non-abelian HSP hardness is therefore **not
  uniform** — the group's representation-theoretic structure (specifically,
  the dimension of its irreducible representations, small for \(D_N\), huge
  for \(S_n\)) is the actual lever, not "abelian vs. non-abelian" as a bare
  dichotomy.
- **Noise/partial-oracle degradation is a separate, already-charted axis:**
  the abelian case degrades gracefully (May–Schlieper–Schwinger 2019,
  Noisy Simon / LPN reduction, CLAIM-196 — the best known algorithm stays
  strictly below classical brute force across the whole noise range, no
  cliff); the dihedral case has its own noise-parameter lineage (Regev 2003
  / CLAIM-197 through Bai et al. 2025 / CLAIM-199), tracing a real
  degradation curve from subexponential exact DCP to quasi-polynomial
  relaxed EDCP.

None of this is Direction F's discovery target — it is the **calibration
grid**. The open question this charter actually poses is: outside these
three named points (abelian, \(S_n\), \(D_N\)) and their already-studied
noise variants, is there a wider, systematically-searchable space of
(group-like structure, oracle-function family) pairs — including ones with
no clean textbook group label at all, described purely by an oracle
function's *behavioral* properties — where the same query-complexity gap
shows up, and where nobody has looked yet because nobody enumerated the
space?

---

## 3. North star

> Given only black-box query access to a function \(f:X\to Y\) on finite
> sets (no group label handed in), what **structural, purely functional**
> descriptors of \(f\) — computable from a polynomial number of queries and
> without presupposing which group or subgroup, if any, \(f\) hides — predict
> that \(f\) admits a \(\mathrm{poly}(\log|X|)\)-query quantum recovery
> algorithm for its hidden structure, while any classical algorithm needs
> \(\Omega(|X|^{c})\) queries for some \(c>0\); and does that predictor fire
> on any family that is not already one of Simon/Shor/Kuperberg-dihedral (or
> a direct noise/product variant of them)?

This is explicitly **not** "prove \(P\ne\) something" or "construct a new
circuit." A positive hit is a **lead**: a structural signature, a
parametrized family, and quantitative evidence the gap is real and
size-robust — exactly the outcome-level-2/3 "hidden class" result this
project already knows how to produce and gate honestly (§6). Turning a lead
into a genuine new algorithm (a poly-query quantum recovery procedure with a
correctness proof) is explicitly out of scope for RDE itself and would be a
separate, human-driven `ALGO-NNN`/`THEOREM-NNN` follow-on.

---

## 4. Fixed contract vs. variable design

### 4.1 Fixed — the problem contract

1. **Access model: oracle/query only, non-negotiable.** Every generated
   instance is defined as a function \(f\) that can be *queried* at chosen
   points; RDE's descriptors and any complexity estimate must be computable
   from a bounded, recorded number of queries to \(f\) (or, for closed-form
   theoretical bounds on the calibration grid, from the known formula, not
   from being handed \(f\)'s full truth table as raw data the way Direction
   E's TSP distance matrix was). This is the one thing that must never be
   relaxed without re-deriving the argument in §1 from scratch.
2. **Target quantity, defined without simulating a quantum computer.**
   Standard HSP theory already gives closed-form quantum query complexity for
   every family on the calibration grid (abelian: \(O(\log|G|)\); dihedral:
   \(2^{O(\sqrt{\log N})}\)), so Direction F does not need — and does not
   attempt — real statevector quantum circuit simulation to get a quantum
   query count for structured families. The **primary target** is a
   \(\log\)-ratio of query complexities:
   \[
   \Delta(f) \;=\; \log_2\!\big(Q_{\text{classical}}(f)\big) \;-\; \log_2\!\big(Q_{\text{quantum}}(f)\big),
   \]
   where \(Q_{\text{quantum}}(f)\) is the closed-form bound for the family
   \(f\) was drawn from (only computable/known for the calibration grid and
   any family a human has since proven a bound for — an **unknown** quantum
   bound is recorded as missing, never guessed), and \(Q_{\text{classical}}(f)\)
   is a concrete, computed value: either a proven classical query lower
   bound where one is known (e.g. the generic collision/birthday bound
   \(\Theta(\sqrt{|X|/|K|})\) for finding a period/shift by querying a
   black-box function with no further structure), or the empirical query
   count of an explicit, fixed classical baseline algorithm (random sampling
   + collision detection) actually run against the instance and recorded.
   \(\Delta(f)\) large and growing with size is the signature of a real
   query-complexity gap; RDE's discovery loop asks which **structural**
   descriptors of \(f\) (not \(\Delta\) itself, not anything derived from
   \(Q_{\text{quantum}}\)/\(Q_{\text{classical}}\) — see leak audit, §5)
   predict it.
3. **Output guarantee, stated per instance:** exact recovery of the hidden
   subgroup/shift \(K\) with probability \(\ge 2/3\) (the standard HSP
   convention), boosted to any constant confidence by \(O(1)\) repetitions —
   stated explicitly in the domain contract rather than left implicit, per
   the same discipline the Direction E and Direction A co-design charters
   already enforce for their own output guarantees.
4. **Descriptors must be computable from \(f\) alone, not from \(G\)/\(K\).**
   Any descriptor that requires already knowing which group or subgroup
   generated the instance is a label, not a structural predictor, and is
   excluded from the leak-clean predictor set (§5) the same way a QUBO
   experiment excludes outcome-derived columns.

### 4.2 Variable — everything RDE is free to search over

- The generating group/structure itself: abelian (\(\mathbb Z_2^n\),
  \(\mathbb Z_N\), general finite abelian products), dihedral \(D_N\),
  bounded-size non-abelian factors combined with abelian ones (product/
  semidirect constructions), and a continuously-tunable "structure-break"
  axis that interpolates between an exactly-abelian hidden-subgroup function
  and a generic function with no exploitable structure at all — the same
  planted-and-tunable-degradation pattern already used successfully in this
  repo for `tsp_circulant_symmetry` (`src/rde_domains/tsp/circulant.py`),
  generalized here from geometric symmetry-breaking to algebraic
  (commutator-norm / irrep-dimension) symmetry-breaking.
- The noise/partial-oracle axis (independent of the structure axis):
  corrupted or missing oracle answers at a tunable rate, connecting directly
  to the noisy-Simon (CLAIM-196) and noisy/extrapolated-DCP (CLAIM-197–199)
  literature already on record.
- Which structural descriptors end up predictive: Walsh–Hadamard spectrum
  sparsity/energy concentration, algebraic degree and correlation immunity
  (standard Boolean-function cryptanalysis quantities — see the literature
  gap-check in §5, Step 0), empirical collision/autocorrelation statistics
  from bounded random queries, and any GF(2)/GF(p)-linear-algebra structure
  detectable from query samples (e.g. approximate null-space rank of a
  sampled difference table) — RDE's descriptor catalog, ranker, symbolic
  regression, and Phase 6 representation search are the actual search
  machinery, per `ARCHITECTURE.md`'s three forward modalities, not a
  hand-picked shortlist decided in advance.

---

## 5. Work order

### Step 0 — Literature gap-check (targeted, not a fresh saturation pass)

CLAIM-188–203 already cover the abelian/non-abelian HSP hardness landscape,
Kuperberg's dihedral algorithm, and the noisy/approximate-HSP lineage in
depth (two saturated passes, 2026-08-19). What is **not yet checked**: (a)
whether cryptographic Boolean-function descriptors (Walsh spectrum,
correlation immunity, algebraic degree, nonlinearity) have already been
connected to HSP-style query advantage in the literature; (b) the broader
landscape of which specific groups beyond \(\mathbb Z_2^n\)/\(\mathbb
Z_N\)/\(D_N\)/\(S_n\) have known efficient or known-hard HSP algorithms —
so the discovery search targets genuinely unclassified territory, not a gap
in this project's reading; (c) whether anyone has already framed "search
over function/group families for HSP-style speedups" as a systematic
discovery problem the way this charter proposes. Track via a short targeted
gap-check audit, registering only genuinely new CLAIM-NNN cards (do not
re-register CLAIM-188–203's territory).

### Step 1 — `hsp_functions` RDE domain (implemented)

Package `src/rde_domains/hsp_functions/`: World-layer generators for the
calibration grid (Simon, cyclic period-finding, dihedral coset) plus the
variable families (§4.2), all under the fixed query-only contract (§4.1).
Observe-layer: `prepare_instance` builds the oracle and the bounded-query
descriptor bundle **once**; `materialize` and `primitive_features` consume
that cache (one `sample_difference_estimates` draw, not a second independent
profile). Raw arrays (`diff_profile`) are exposed so Mode 1's catalog can
run — the exact gap Direction E's history (`roadmap.md` item 6) shows is
easy to get wrong. Boolean-function descriptors live in ALGO-061
(`src/rde/features/boolean.py` + `sampling.py`).

### Step 2 — Gate 0: mechanism validation on the calibration grid

Before any discovery campaign: confirm the domain's \(\Delta(f)\) target and
descriptors reproduce the *known* answers — \(\Delta\) large and
\(\log|G|\)-scaling for Simon/Shor, present but smaller for dihedral,
correctly computed (even if not usefully large) for \(S_n\)-like generic
non-abelian cases, and near-zero for the pure-noise/no-structure control.
This is rediscovery-as-pipeline-check (§1, rule 2), not a result — do not
report it as one.

### Step 3 — Domain contract, held-out families, discovery campaign

Register a `DomainContract` (`src/rde/core/domain_contract.py`) with Simon,
Shor-cyclic, and dihedral held out of discovery entirely (they are
validation, not search space — consistent with, but stricter than, the
usual held-out-family discipline: held out **permanently**, not just from
one split). Run the real recipe per
[experiment-playbook.md](../../src/rde/docs/experiment-playbook.md): a
real population (many random instances per family per size, not one
landscape), leak audit (\(\Delta(f)\) and anything derived from
\(Q_\text{quantum}\)/\(Q_\text{classical}\) excluded from the predictor set),
the discovery loop (ranker + latent/symbolic + Phase 6 representation
search), `assess_outcome` gate, pre-registration with a fixed decision rule
and stop rule, cross-\(N\) and held-out-family generalization reporting.
This is the multi-day, multi-size, many-family campaign — sized for
unattended MLX execution on Apple Silicon, checkpointed and resumable, with
real progress and durable logs per CLAUDE.md's requirement.

### Step 4 — Novelty check on every surviving signal

Any structural predictor that clears the gate and generalizes to a held-out
size/family, on a family that is not itself just "detectably abelian" or
"detectably dihedral" (§1, rule 2), gets a targeted literature-swarm pass
before it is described as new anywhere in this repo — same discipline as
Direction E §5 and the project-wide novelty-check norm.

---

## 6. Honest scope and non-claims

- RDE does not simulate a real quantum computer here and does not need to:
  \(Q_\text{quantum}\) is read from closed-form theory on the calibration
  grid, and is explicitly **undefined/missing**, not guessed, for any family
  outside it. A campaign result is never "this family has quantum query
  complexity \(X\)" unless a human has since derived that bound — RDE's
  output is a candidate, not a proof.
- A "signal" result here is an outcome-level 1–3 finding in this project's
  existing ladder (correlate/predict, at best a hidden-class separation) —
  never a level 4–5 "verified new algorithm." Turning a signal into an
  actual algorithm requires a human-derived construction and its own
  `ALGO-NNN`/`THEOREM-NNN` cards, gated exactly like every other original
  result in this repo.
- This charter does not claim, and Step 0/2 are explicitly designed to
  prevent claiming, that any named family here is new before a literature
  check says so.
- If Step 3's campaign clears the gate only on the calibration grid and
  finds nothing on the variable families, that is a legitimate, reportable
  **negative** — per this project's stop rule, not grounds to keep sweeping
  more families until something trips.

---

## 8. What EXP-064–066 actually measured (and why that cannot produce an algorithm)

EXP-064–066 ran **Mode 1 Pearson screens** against the wrong objects:

| Receipt | Target | Gated verdict | What it actually asked |
|---|---|---|---|
| EXP-066 `receipt.json` | collision rate vs planted pairing | SIGNAL / grade 1 | Do repeated labels exist in a \(B=20n^2\) sample? |
| EXP-066 `receipt_recipe_catalog.json` | collision rate vs `structure_strength` | NULL / grade 0 | Does birthday physics track a planted [0,1] knob? |
| EXP-066 `receipt_recipe_kinds.json` | span+period vs literature class | NULL / grade 0 | Can two ALGO-061 statistics name Simon / Shor / dihedral through \(N=24\)? |

None of those targets is \(\Delta(f)\), and none is recovery of \(K\). Mode 1
cannot emit \(O\). The kinds screen failing at \(N=20,24\) is a classifier
scaling result, not “RDE found nothing and we stop.”

\(\Delta\) was never the gated target: `complexity.py` is an interpretive
table for the three textbooks and is **not** wired as a per-row outcome,
because a closed-form \(Q_{\text{quantum}}\) exists only on the calibration
grid (tautological with family identity) and is NaN on Heisenberg / Q8 /
blends. Substituting `structure_strength` / `algorithm_class` did not escape
that trap — it replaced the charter question with a weaker one.

**Do not** open another EXP-NNN whose decision rule is \(|r|\ge 0.35\) on a
label, collision rate, or planted strength. That loop cannot produce an
algorithm.

---

## 9. Next work order — recovery-protocol search (Mode 2, not another ranker)

This is still Direction F and still two modes only. The missing executable
surface is **`RecoveryDomain`** (ALGO-063): same Mode 2 question as
ALGO-057, but HSP algorithms are query-tape extractors, not divide/combine
QUBO recurrences.

### P (fixed)

Black-box \(f:X\to Y\). Output: a candidate for the planted hidden object
\(K\) (Simon string, cyclic period, dihedral fold, Heisenberg generator,
…). Success: exact match with probability \(\ge 2/3\) on a pre-registered
population, boosted by \(O(1)\) independent tapes if needed.

### \(C_{\text{target}}\) (fixed)

\(B=20n^2\) classical queries (same ALGO-061 budget already frozen; do not
retune to chase a rate) plus \(\mathrm{poly}(n)\) classical post-processing.
No quantum statevector. No guessed \(Q_{\text{quantum}}\).

### \(O\) (the search object)

A catalog of extractors that see **only** a `QueryTape` \((x_i,f(x_i))\).
They do not receive the family name, generator, or planted \(K\). The
initial catalog (implemented) is three family-agnostic collision moves:

- XOR-mode of colliding pairs
- modular-sum mode of colliding pairs
- GCD of modular differences of colliding pairs

Returning `None` is abstain (correct on structureless junk).

### Pipeline check (not a finding)

On exact Simon / Shor-cyclic / dihedral, the matching move must recover
the planted secret at the confirmatory sizes. That is rediscovery of known
classical post-processing. Report it as Gate 0, never as a result.

### Discovery (the only thing that can be a lead)

On families **with no dedicated extractor in the catalog** (Heisenberg,
quaternion, blends, unlabeled recipes): does any catalog move, or a later
composition of the same verb set, recover planted \(K\) at a rate that
holds through the confirmatory horizon?

- **Yes, and literature does not already own that move on that family** →
  candidate algorithm. Human `ALGO-NNN` / `THEOREM-NNN` next. RDE still
  does not prove it.
- **No** → reportable negative *under this verb set and this budget*. Stop
  or enlarge the verb set with a written reason. Do not add a
  Heisenberg-named extractor and then call its success a discovery.

### What this is not

Not a behavior-only / drop-the-group reformulation. Planted \(K\) remains
the scoring object. Not a Pearson gate. Not a new EXP folder until this
recovery loop exists, is tested, and has a pre-registered decision rule on
the recovery matrix.

---

## 10. Deliverables (tracking)

| Deliverable | Status |
|---|---|
| This charter | **Done** (this file) |
| Literature gap-check (Boolean-function descriptors × HSP query advantage) | Open — egress blocked 2026-08-19; see `literature/audits/2026-08-19_hsp-function-descriptor-gap-check-blocked.md` |
| `src/rde_domains/hsp_functions/` domain (calibration grid + variable families) | **Done** — Observe cache via `prepare_instance` |
| ALGO-061 / ALGO-062 | **Done** |
| `DomainContract` + Gate 0 mechanism validation | **Done** (contract registered; Gate 0 interactive, artifacts discarded) |
| Pre-registered multi-day discovery campaign | **Formulated** — [EXP-064](../../experiments/EXP-064_hsp_functions_structure_strength_discovery/) (not the confirmatory run) |
| Novelty check on any surviving signal | Open — EXP-067 SIGNAL is `xor_mode_high_half` on this Heisenberg bit packing; not yet a THEOREM/ALGO promotion |
| Mode 2 `RecoveryDomain` + collision extractors (ALGO-063) | **Done** for the enumerated catalog |
| Pre-registered recovery-protocol campaign | **Run** — [EXP-067](../../experiments/EXP-067_hsp_recovery_protocol_search/) SIGNAL / grade 1 |
