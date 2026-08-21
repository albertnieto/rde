# TSP novel-representation discovery charter (Direction E)

**Status date:** August 2026.

**Status tag:** `active-research-framing` — not a theorem, algorithm card, or
experiment. This is a **methodology guardrail document first, research plan
second** — the guardrail in §1 exists because it was violated once already in
this same conversation thread and must not be violated again.

**Supersedes:** the earlier framing (this session) of "search for
representations within TSP/MaxCut/3SAT's *known* structure (graph Laplacian,
clause hypergraph, permutation symmetry)." That framing was corrected by the
user and is wrong — see §1.

**Related docs:** [roadmap.md](../roadmap.md) Direction E,
[methodology.md](../../src/rde/docs/methodology.md) (PZXESO stack — the
vocabulary this charter uses throughout),
[hierarchical-synthesis.md](../../src/rde/docs/hierarchical-synthesis.md),
[ALGO-057](../algorithms/ALGO-057_rde-hierarchical-synthesis-backward-target-search.md)
(backward target-first synthesis — the search engine),
[ALGO-058](../algorithms/ALGO-058_qubo-interaction-component-synthesisdomain-stage-a.md)
(Stage A QUBO SynthesisDomain — the closest existing template, see §4),
[generic-qubo-instrument-co-design-charter.md](generic-qubo-instrument-co-design-charter.md)
(sibling charter for Direction A's co-design track; same document shape).

---

## 1. The rule this charter exists to enforce

**RDE's job on this track is to find a mathematical structure/representation
for TSP that has not been previously named or published — not to rediscover
and then exploit a structure that is already known.**

Concretely:

- Permutation-group symmetry \(S_n\), the graph/metric structure of a TSP
  instance, Held–Karp's DP recurrence, LP/SDP relaxations, Christofides-style
  approximation structure — these are **known objects**. RDE is allowed to
  *rediscover* one of them as a sanity check that the search pipeline works
  at all (a positive rediscovery is evidence the pipeline is sound), but a
  rediscovery is **not a result** for this track and must not be reported as
  one.
- Do **not** hand RDE a known structure and ask it to search *within* that
  structure for a poly(\(N\)) construction (e.g. "search for a good
  permutation-matrix embedding," "search the Laplacian eigenbasis"). That
  presupposes the answer's shape and forecloses exactly the kind of finding
  this track exists to make.
- Instead, point RDE's generative search (ALGO-057's backward synthesis,
  extended with a new TSP `SynthesisDomain`) at the **raw combinatorial
  object** — the TSP instance's distance/cost data itself, with no
  pre-imposed decomposition — and let it propose candidate \(Z\)
  (structure), following [ALGO-058](../algorithms/ALGO-058_qubo-interaction-component-synthesisdomain-stage-a.md)'s
  own stated principle: *"structure must be found from the instance itself,"*
  not pre-split or handed in.
- Every candidate \(Z\) that survives Gate 0 gets a **novelty check**
  (`literature-swarm`, CLAIM-NNN search) before it is described as new
  anywhere in this repo. "We don't know a name for this" is not the same as
  "this is new" — say "unidentified in the literature search performed"
  until an honest saturation pass says otherwise.

If a future session (agent or human) proposes "let's search for a
representation inside TSP's known permutation/metric structure," that is the
same mistake this section documents — read this section again before
proceeding, not the roadmap summary alone.

---

## 2. Why TSP, not MaxCut or 3SAT (recorded so this isn't re-litigated)

The goal (per [AGENTS.md](../../AGENTS.md) and `docs/roadmap.md`'s Summary) is
a conditional **NP ⊆ BQP** chain: *one* NP-hard problem with a genuine
poly(\(N\))-resource quantum construction is sufficient. Problem choice is a
**structure × impact** trade-off, not a convenience trade-off (existing
partial wiring, e.g. EXP-060's sparse-MaxCut arm or `coined_walk`'s MaxCut
domain, is explicitly **not** a selection criterion here).

| Candidate | Structure | Impact | Verdict |
|---|---|---|---|
| **MaxCut** | High (spectral graph theory) but MaxCut *is already* a native QUBO instance family — \(Q\) *is* the graph, no encoding/representation step exists to search over. | Canonical NP-hard benchmark, but QPFA already runs directly on QUBO \(Q\) matrices, so there's no open representation question here to give RDE. | **Rejected** — nothing for this track to discover; already fully explored as a QUBO instance family (EXP-036/059/060's sparse-MaxCut arm). |
| **3SAT** | Low by design — worst-case CNF is built to have minimal exploitable regularity; the standard \(k\)-body→2-body QUBO reduction is well-studied and has seen little improvement, so the raw object is close to a blank canvas. | Maximal — the canonical NP-complete problem; every other NP problem reduces to it, so a positive result here is the broadest possible headline. | **Fallback candidate.** Highest ceiling, hardest floor — revisit if TSP stalls, or run in parallel with independent budget if resources allow. |
| **TSP** | High — permutation group \(S_n\), and (for metric/Euclidean instances) genuine geometric/triangle-inequality structure the search can key off. The standard one-hot permutation-matrix QUBO reduction is known to be inefficient (\(O(N^2)\) variables, brittle penalty weights), so there is real headroom for a better representation. | High — one of the most practically consequential NP-hard problems (logistics, routing, scheduling). | **Selected.** Best available structure-for-the-search × real-world-impact trade-off. |

---

## 3. North star

> Given TSP instance data (a distance/cost matrix, or the raw city-coordinate
> generator for Euclidean instances), what structure \(Z\), coordinates \(X\),
> encoding \(E\), state space \(S\), and operators \(O\) — in the PZXESO sense
> — make a poly(\(N\))-resource quantum construction achievable, where \(Z\)
> is not one of the known objects listed in §1?

This is explicitly **not** "run QPFA/ALGO-001 on a TSP-shaped QUBO." The
standard TSP→QUBO reduction is itself a candidate representation to be
searched *against*, not assumed. A finding that the standard reduction is
provably suboptimal on some measurable axis (ancilla count, conditioning,
gate count) — even without a positive replacement — is a legitimate, useful
negative result for this track, of the same shape as THEOREM-005/023's
scoped negatives on Direction A.

---

## 4. Fixed contract vs. variable design (mirrors the Direction A co-design
charter's §3 shape)

### 4.1 Fixed — the problem contract

1. **Instance class:** start with metric/Euclidean TSP (triangle inequality
   holds; not general asymmetric TSP) — the more structured sub-case, so the
   search has real geometric data to find regularity in. General TSP is a
   later widening, not the starting point.
2. **Access model:** explicit distance-matrix / coordinate access (uniform
   classical description of the instance), matching this project's existing
   access-model convention (no hidden-oracle framing — see THEOREM-014's
   correction of THEOREM-011–013 for why that distinction matters here too).
3. **Output guarantee:** state explicitly per experiment whether the target
   is exact optimal-tour recovery, threshold decision (tour cost \(\le T\)),
   or constant-probability \(\varepsilon\)-approximate sampling — do not
   leave this implicit (this project's own history under Direction A shows
   what happens when the output guarantee is left vague — see
   [generic-qubo-instrument-co-design-charter.md](generic-qubo-instrument-co-design-charter.md)
   §3.1 for the precedent).
4. **Compilation:** uniform classical poly(\(N\)) construction of whatever
   circuit/instrument \(Z/X/E/S/O\) implies, from the instance data.
5. **Resources:** polynomial total gates, ancillas, precision, and shots —
   the same bar as every other direction in this repo.

### 4.2 Variable — everything RDE is free to search over

- The representation \(Z\) itself: no committed choice between permutation
  matrices, edge-indicator variables, position encodings, or anything else.
- Coordinates \(X\): whatever basis/ordering makes a discovered \(Z\) cheap.
- Encoding \(E\): amplitude, register, or hybrid encodings of \(Z/X\).
- Whether the eventual construction connects to QPFA's phase-filtration
  machinery, Direction D's cost-register approach, or neither — **do not
  presuppose QPFA** (mirrors PZXESO's own "domain-generic, does not
  presuppose QUBO/QPFA" framing in
  [methodology.md](../../src/rde/docs/methodology.md) §1).

---

## 5. Work order

### Step 1 — Literature saturation pass (baseline, not target)

Run [literature-swarm](../../.claude/skills/literature-swarm/SKILL.md) on
quantum-TSP representations specifically (QUBO permutation encodings,
quantum annealing TSP formulations, QAOA-TSP variants, any non-standard
embeddings already published). Register CLAIM-NNN cards. This pass exists
**only** to make the eventual novelty check in §1 honest — it is a baseline
list of "already known," not a source of search targets.

### Step 2 — Build the TSP `SynthesisDomain`

New `src/rde_domains/tsp/` domain implementing RDE's `SynthesisDomain`
protocol (`rde.core.protocols.SynthesisDomain`), following
[ALGO-058](../algorithms/ALGO-058_qubo-interaction-component-synthesisdomain-stage-a.md)'s
shape but **do not** pre-inject the permutation-matrix encoding as a
`decompose_flat`/`decompose_divide` move — those moves must be derived from
the raw distance matrix (e.g. via genuinely-discovered sub-structure such as
clustering, nearest-neighbor graphs, or something RDE itself proposes), the
same "structure must be found, not handed in" discipline ALGO-058 already
enforces for QUBO.

### Step 3 — Gate 0 before any ALGO card

No ALGO card for a "representation — construction TBD." A card requires an
explicit \(Z/X/E\) and either a poly(\(N\)) compilation argument or an
explicit `poly_gates: no` with the blocking step named — same bar as
[generic-qubo-instrument-co-design-charter.md](generic-qubo-instrument-co-design-charter.md)
§4 and the project-wide ALGO-card requirement in
[new-algorithm](../../.claude/skills/new-algorithm/SKILL.md).

### Step 4 — First RDE campaign

Run ALGO-057's backward target-first search against the new domain with a
declared resource budget, per
  [experiment-playbook.md](../../src/rde/docs/experiment-playbook.md)'s
mandatory recipe (real population, domain contract, leak audit, held-out
families, discovery loop, `assess_outcome` gate, pre-registration, stop
rule — no exceptions for this being a "new direction"). Live progress and
durable logs are required per CLAUDE.md's progress/logging policy — this is
exactly the kind of non-trivial, potentially long-running run that
requirement is for.

### Step 5 — Novelty check on every surviving \(Z\)

Before any surviving structure is described as a finding anywhere in this
repo (roadmap, ALGO card, results.md), re-run a targeted literature check
against Step 1's saturation baseline. If it matches a known object, report it
as a **rediscovery** (useful pipeline validation, not a novel-structure
result) and keep searching. If it's a genuinely new combination or object,
that is the actual deliverable this track exists to produce.

---

## 6. Realistic expectation

Most early RDE output on a brand-new domain is noise or rediscovery — that is
expected and is not failure (§5 exists to catch and correctly label it, not
to prevent it). A negative result that the standard permutation-matrix
reduction is provably non-optimal on a stated resource axis is a legitimate
scoped-negative outcome. A single genuinely unnamed \(Z\) that both survives
Gate 0 and clears the novelty check would be the first concrete deliverable
of this track.

---

## 7. Open deliverables (tracking)

| Deliverable | Status |
|---|---|
| This charter | **Done** (this file) |
| Literature saturation pass (quantum-TSP representations) | **Done** — CLAIM-176–187, [audit](../../literature/audits/2026-08-18_tsp-novel-representation-discovery-saturation.md) |
| `src/rde_domains/tsp/` SynthesisDomain | **Done** — [ALGO-060](../algorithms/ALGO-060_tsp-interaction-cluster-synthesisdomain-direction-e.md) |
| Gate 0 mechanism confirmation | **Done** — [EXP-061](../../experiments/EXP-061_tsp_clustered_synthesis_stage_a/) |
| Confirmatory sweep of the one implemented mechanism | **Done** — [EXP-062](../../experiments/EXP-062_tsp_clustered_synthesis_confirmatory_sweep/): 4200 instances, zero combine mismatches; see §8 for why this is not yet the generative search below |
| Generative search over *multiple* candidate \(Z\) (the actual North star) | **Open** — needs the domain to expose more than one decomposition move; not started |
| Novelty-check pass on any surviving \(Z\) | **Open** — nothing to check yet; blocked on the item above, not on the confirmatory sweep |

## 8. Status note (Aug 2026): infrastructure done, generative search not yet started

Steps 1-3 of §5's work order are done, and step 4 ran — but as a
**confirmatory sweep of the one mechanism `src/rde_domains/tsp/` implements
(MST-gap clustering)**, not the generative search over *candidate*
mechanisms this charter's North star (§3) describes. That distinction
matters and should not be blurred in future summaries: MST-gap clustering
is itself a known technique (§2's literature baseline already covers it
implicitly as "ordinary clustering," and §1 explicitly says known
structures are not the target). EXP-062 confirmed that mechanism is exact
wherever it fires, at real scale (4200 instances, zero mismatches) — a
legitimate, useful result (it retires "is the joint-boundary combine
actually correct" as an open question) — but it is infrastructure
validation, not the search for an unnamed \(Z\).

**What "the generative search" actually requires, not yet built:** the
current `TspSynthesisDomain.decompose_divide` hardcodes one decomposition
rule. ALGO-057's skeleton search can only choose *whether* to apply that
one rule, at what recursion shape (flat/divide/subtract) — it cannot
propose a genuinely different rule, because the domain does not expose more
than one. A real search over candidate \(Z\) needs the domain (or a family
of domain variants) to expose *several* structurally-derived decomposition
moves — e.g., different clustering criteria, different boundary-selection
rules, entirely different combinatorial objects derived from the distance
matrix (not just MST gaps) — so that ALGO-057's search is actually choosing
among live candidates, not confirming the one it was given. Building that
wider move vocabulary, and only then running a real discovery-style
campaign over it, is the next concrete step for a future session — with its
own fresh Gate 0 and pre-registration, per this charter's own discipline.
