# RDE Methodology — Theory and Definitions

**Status date:** August 2026.

**Status tag:** `canonical-methodology` — this is the **single science vocabulary**
for the Representation Discovery Engine. Implementation status, CLI recipes, and
experiment gating live elsewhere (see Related docs). This page is **not** a
theorem, not an `EXP-NNN` finding, and not a claim that any particular problem
admits a polynomial algorithm.

**Related docs (supporting, not competing ladders):**

- [RDE experiment playbook](experiment-playbook.md) — how to
  claim results
- [RDE README / CLI](README.md) — entry and commands
- [RDE architecture](ARCHITECTURE.md) — code pipeline
- [Mode 2 compilation depth](hierarchical-synthesis.md) — TARGET → primitives
  staging for reverse synthesis

---

## 1. What RDE is

RDE is a **domain-generic conjecture factory**.

It does **not** presuppose QUBO, Hilbert space, or any one physical model.
A domain adapter declares what the objects are and what counts as a valid
execution. The methodology below is the shared language for every domain.

RDE asks one scientific question in two modes:

> Given a problem \(P\) and a resource budget \(C_{\mathrm{target}}\), what
> structure \(Z\), coordinates \(X\), encoding \(E\), state space \(S\), and
> operators \(O\) make the required queries or transformations achievable?

ML, symbolic search, and program synthesis are **hypothesis generators only**.
They never certify a theorem. Promotion still goes through `EXP-NNN` and, when
proven, `THEOREM-NNN`.

---

## 2. The method stack (canonical)

\[
\boxed{P \;\longrightarrow\; Z \;\longrightarrow\; X \;\longrightarrow\; E \;\longrightarrow\; S \;\longrightarrow\; O}
\]

Spoken **PZXESO**. Every letter is a different kind of claim. Higher letters do
not follow automatically from lower ones.

| Letter | Name | Definition |
|---|---|---|
| **P** | **Problem** | The scientific object and task: instance family, generators, allowed queries, success criterion, and the resource budget \(C_{\mathrm{target}}\). |
| **Z** | **Structure / representation** | The mathematical object that *exposes* exploitable regularity (decomposition, factorisation, algebra, graph class, spectral object, recurrence, …). **Not** the answer / optimum. |
| **X** | **Coordinates** | The basis, chart, ordering, or variable system in which that structure is computationally simple (block axes, low-rank factors, elimination order, Fourier modes, …). |
| **E** | **Encoding** | How \(Z/X\) information is written into machine-readable state so later operations are cheap (tables, registers, amplitudes, bitstrings, tensors, programs, …). |
| **S** | **State / execution space** | The **domain-supplied** space in which computation runs. See §3. |
| **O** | **Operator / protocol** | The transformation (classical or quantum, deterministic or randomised) that, given an efficiently preparable state under \(E\) on \(S\), yields an efficiently extractable answer under \(C_{\mathrm{target}}\). |

### Resource budget (first-class, not a letter)

\(C_{\mathrm{target}}\) sits **beside** the chain. Typical axes (domain chooses
which apply):

- classical time / memory;
- representation size;
- circuit depth / gate count / qubits / ancillas / shots;
- success probability floor;
- exact vs approximate.

Every status claim and every experiment must name the budget axis it satisfies.
Bare “done” is forbidden.

---

## 3. Domain-supplied state space \(S\)

**\(S\) is never fixed by the methodology to Hilbert space.**

RDE is universal *because* domains exist. The domain adapter supplies the
semantics of \(S\):

| Domain family (examples) | Possible \(S\) |
|---|---|
| Quantum / domain-specific | Hilbert space with a declared register layout |
| Classical combinatorial | Assignment space \(\{0,1\}^n\), paths, matchings, … |
| Linear / spectral | Vector or matrix space |
| Probabilistic | Distributions / measure spaces |
| Symbolic | Program or expression state |
| Hybrid | Product of several of the above |

### Contract a domain must declare for \(S\)

1. **Valid states** — what objects inhabit \(S\).
2. **Identity / equivalence** — when two states count as the same.
3. **Allowed operations** — what maps on \(S\) are primitive.
4. **Composition** — how operations compose.
5. **Query / observable interface** — what can be read out (evaluate, overlap,
   compare, …).
6. **Resource accounting** — how cost is charged against \(C_{\mathrm{target}}\).

Quantum domains may write \(S = \mathcal{H}\). Other domains must not be forced
into that reading. Core RDE (`src/rde/`) remains domain-agnostic and must not
import external domain adapters.

---

## 4. Two modes only

### Mode 1 — Forward discovery

\[
P \;\longrightarrow\; Z \;\longrightarrow\; X \;\longrightarrow\; E \;\longrightarrow\; S \;\longrightarrow\; O
\]

**Question:** what structure is hiding here?

Fix \(P\) (and usually \(S\)). Populate instances, search for \(Z/X/E/O\) that
explain complexity or enable the locked queries. Emit conjectures for human
proof or scoped negatives.

### Mode 2 — Reverse synthesis

\[
C_{\mathrm{target}} \;\Longrightarrow\; (Z,X,E,O)
\quad\text{with } P \text{ declared and } S \text{ supplied by the domain}
\]

**Question:** what must exist for this budget to be achievable?

Start from the budget and required success criterion. Search for skeletons and
representations that meet the budget, then (when built) compile downward. Never
derive \(Z/X/E/O\) from the hidden answer.

The two modes feed each other: forward finds candidate structure classes;
reverse asks whether those classes admit a budget-respecting protocol.

There is **no third mode**.

Mode 2 has two executable surfaces, both answering the same question:

- **`SynthesisDomain`** — divide / combine recurrences (block-separable
  and TSP-clustered skeletons).
- **`RecoveryDomain`** — query-tape extractors under a declared query
  budget (hidden-subgroup recovery). Pearson correlation of a
  descriptor against a family label is Mode 1 and is **not** a substitute
  for either surface.

---

## 5. Reverse-mode compilation depth

Inside Mode 2 only, search descends a constrained compilation stack. This is
**not** a second science ontology; it is how reverse mode constructs
\(Z/X/E/O\) under \(C_{\mathrm{target}}\).

```text
TARGET
  ↓
ALGORITHM          (skeletons + restricted verb vocabulary)
  ↓
MATHEMATICAL OPERATIONS
  ↓
COMPUTATIONAL REPRESENTATION
  ↓
ENCODING INTO S    (classical or quantum, domain-defined)
  ↓
REVERSIBLE / MACHINE OPERATIONS
  ↓
PRIMITIVE GATES / INSTRUCTIONS
  ↓
RESOURCE + COMPLEXITY ANALYSIS
```

| Compilation step | Usually lands in |
|---|---|
| TARGET + recurrence / complexity prune | \(C_{\mathrm{target}}\) |
| ALGORITHM skeletons | shape of \(O\), often of \(Z\) |
| Mathematical operations | concrete operations on \(Z\) |
| Computational representation | \(X\) and classical forms of \(E\) |
| Encoding into \(S\) | \(E\) on the domain’s \(S\) |
| Machine ops → primitives | implementable \(O\) |
| Resource analysis | verify \(C_{\mathrm{target}}\) |

**Ordering rule:** prune by complexity **before** domain execution and **before**
any machine-level compilation. Example:

\[
T(n)=2\,T(n/2)+\mathrm{poly}(n)
\quad\text{(accepted)}
\qquad\text{vs}\qquad
T(n)=2\,T(n-1)+\mathrm{poly}(n)
\quad\text{(rejected)}.
\]

**Implementation note (not methodology):** as of August 2026, code covers
TARGET + ALGORITHM + pre-domain recurrence pruning (see
[hierarchical-synthesis.md](hierarchical-synthesis.md)). Lower compilation
steps are future work. Status belongs in an implementation status doc, not
here.

---

## 6. Outcome grades (grading axis, not letter \(O\))

Letter \(O\) is the **operator/protocol**. Separately, every discovery or
synthesis result is graded by **strength**:

| Grade | Name | Meaning |
|---|---|---|
| **G0** | Nothing | No stable signal beyond noise; valuable negative evidence under an explicit search quantifier |
| **G1** | Predictor | An instance functional explains complexity / behaviour across held-outs |
| **G2** | Hidden classes | Stable partition into tractable vs intractable (or similar) types |
| **G3** | Recurrence / finite memory | Dynamics or solution process closes in poly-size memory |
| **G4** | New coordinates | Poly-size \(X\) with simple dynamics and an efficient decoder |
| **G5** | Constructive representation | Uniform \(Z/X/E/O\) supporting the locked queries inside \(C_{\mathrm{target}}\) |

Same grades apply to forward and reverse. Reaching G5 on a **restricted**
family is not G5 on the general family.

---

## 7. Validation stages (Mode 2 stress tests)

Empirical stages for reverse synthesis. Separate from outcome grades.

| Stage | Instance class | Intent |
|---|---|---|
| **V1** | Known polynomial problems, or toys with planted independent structure | Rediscover a known poly skeleton; reject exponential brute force on complexity alone |
| **V2** | Artificial hidden structure (planted, possibly scrambled) | Structure must be *detected*, not handed pre-split |
| **V3** | Named structured families (treewidth, low rank, band, …) with honest size accounting | Protocol extensions beyond simple separability |
| **V4** | General / dense instances of the declared \(P\) | Verified poly construction **or** scoped negative under an explicit catalog + budget quantifier |

Passing V1/V2 does **not** imply V3/V4, does **not** imply a poly quantum
algorithm, and does **not** close Direction A.

---

## 8. The central trap

RDE must **not** work backward from the answer.

**Invalid:**

\[
P \;\longrightarrow\; x^\ast \;\longrightarrow\; \text{“efficient representation of } x^\ast\text{”}.
\]

That smuggles the solution into state preparation or decomposition.

**Valid:**

> Find \(Z/X/E/O\) such that the optimum (or required query) is identifiable
> through polynomially many efficiently computable structural operations —
> without consulting an oracle for the answer while building those operations.

Operational rules:

1. Structural hooks (decompose, combine, descriptors, …) never peek at the
   reference optimum / `brute_force` output.
2. Costs are re-evaluated independently; carried costs are not trusted.
3. Reference oracles verify candidates; they do not choose partitions.
4. Size parameters must be honest (no hiding \(2^n\) behind a coarser “size”).

---

## 9. How a domain plugs in

A domain declares only:

1. **P** — generators, instance contract, held-out families, success criterion.
2. Allowed vocabularies for **Z / X / E** (descriptors, synthesis hooks, DSLs).
3. **S** — the state/execution contract (§3).
4. Candidate shapes or search space for **O** (forward metrics, reverse
   skeletons, …).
5. **\(C_{\mathrm{target}}\)** and which mode(s) apply.

No domain invents a new Layers / Levels / Phases / Tracks science ladder. Domain
charters are specialisations of PZXESO.

**Instance-cache code contract** (how those declarations hit the worker):

| Hook | Role |
|---|---|
| `generate` | Sample **P** |
| `prepare_instance` | Compute expensive shared primitives **once** (query sample, landscape, tour costs, …) |
| `materialize` / `primitive_features` | Consume that cache; expose raw arrays, not a handful of mechanism scalars |
| `DomainContract` | Primary target, leak-audited predictors, held-out families |

`prepare_instance` is required whenever `materialize` and `primitive_features`
would otherwise recompute the same expensive primitive. The worker calls it
once and passes `cache=` into both methods. This is an engineering contract
for honest **Z** measurement, not a new science letter. Details:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

Example (illustrative, not a claim of generality): `rde.testing.block_separable`
(a block-diagonal cost function with block boundaries given, not hidden) is
**Mode 2**, search at ALGORITHM / \(Z\)-skeleton depth, validation **V1**
(rediscovers a known-poly decomposition skeleton, rejecting exponential
brute force on complexity alone) — an *instance* of this methodology. The
planted-vs-control **V2** pair is `rde_domains.tsp`'s `tsp_clustered` /
`tsp_uniform_control`, not `block_separable`; see
[hierarchical-synthesis.md](hierarchical-synthesis.md) §6.

---

## 10. Atoms of structure (unnumbered)

Structure is built from **objects**, **distinctions**, **relations**, and
**operations**, described in a **language**, relative to a **task** and a
**resource model**. A short description is not automatic cheap computation.
“No structure” is almost always too strong; rigorous negatives quantify over an
explicit representation class, model, query set, family, and budget.

---

## 11. No parallel ladders

Do not introduce a second numbered science stack. Status claims use §12.
Engineering chronology belongs in `src/rde/docs/roadmap.md`, not here
(ship slices 0–6 are implementation order, not G0–G5).

---

## 12. Status claims — required axes

Every “done / partial / not built” line must state:

1. **Methodology depth** — which of \(P,Z,X,E,S,O\) are in scope;
2. **Compilation depth** — TARGET … primitives (Mode 2);
3. **Outcome grade** — G0–G5;
4. **Validation stage** — V1–V4 when Mode 2;
5. **Engineering** — code exists / tests pass.

Example of a correct claim:

> `block_separable`: Mode 2; compilation = TARGET+ALGORITHM; validation = V1;
> outcome = rediscovery of a flat decomposition skeleton against exponential
> brute force on a block-separable cost function (block boundaries given,
> not hidden — not V3/V4, not a general poly algorithm).

---

## 13. Non-claims

- This document does not prove that a polynomial algorithm exists for any hard
  family.
- Domain success at V1/V2 is not V4 and is not Direction A closure.
- Forward engineering apparatus existing is not G4/G5 scientific success.
- \(S=\mathcal{H}\) is optional, not universal.
- ALGO/EXP cards instantiate this stack; they must not redefine it.

---

## 14. Operating rule

**RDE has one stack (PZXESO), two modes (forward / reverse), one outcome
vocabulary (G0–G5), one reverse validation progression (V1–V4), domain-supplied
state spaces \(S\), and ALGO/EXP cards that instantiate — never redefine — that
stack.**
