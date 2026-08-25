# RDE hierarchical synthesis — reverse-engineering stack

**Status date:** August 2026.

**Status tag:** `mode-2-compilation-staging` — **not** a science ontology.
Canonical stack is [RDE Methodology — Theory and Definitions](methodology.md)
(PZXESO). This page is methodology §5 compilation depth for Mode 2 only
(TARGET → … → primitives). Letter \(\mathcal{H}\) as a fixed substrate is
retired; domains supply \(S\).

**Canonical methodology (preferred):**
[RDE Methodology — Theory and Definitions](methodology.md)
(\(P\to Z\to X\to E\to S\to O\)). This page is a **Mode 2 compilation-depth**
staging note; letter \(\mathcal{H}\) as a fixed substrate is retired in favor
of domain-supplied \(S\).

**Implements:** `src/rde/synthesis/` (TARGET + ALGORITHM + pre-domain
recurrence pruning); `src/rde_domains/tsp/domain.py` (`TspSynthesisDomain`,
a real `SynthesisDomain` over Euclidean TSP instances).

**Related docs:**

- [RDE Methodology — Theory and Definitions](methodology.md)
- [RDE implementation chronology](roadmap.md)

---

## 1. What this page is

RDE has two complementary search modes (methodology §4):

1. **Mode 1 — Forward discovery** — populate **P**, measure **Z**, search for
   \(Z/X/E/O\).
2. **Mode 2 — Reverse synthesis** — start from \(C_{\mathrm{target}}\), search
   for a skeleton, then compile downward.

This page names the **Mode 2 compilation stack**, states what `rde.synthesis`
implements today, and keeps non-claims explicit. It is **not** a second
science ladder.

---

## 2. Compilation depth (Mode 2)

End-to-end synthesis is intended to descend one constrained layer at a time.
Higher layers prune search so lower layers are never asked to invent arbitrary
circuits from scratch.

```text
TARGET
  ↓
ALGORITHM (skeletons + 10-verb vocabulary)
  ↓
MATHEMATICAL OPERATIONS (restricted DSL)
  ↓
COMPUTATIONAL REPRESENTATION
  ↓
QUANTUM ENCODING
  ↓
REVERSIBLE OPERATIONS
  ↓
GATES
  ↓
RESOURCE + COMPLEXITY ANALYSIS
```

| Layer | Intent |
|---|---|
| **TARGET** | Declared resource budget the search must meet (e.g. polynomial degree). |
| **ALGORITHM** | Named algorithm *skeletons* — decomposition + combine shapes, labeled from a fixed verb vocabulary — not yet arithmetic or circuits. |
| **MATHEMATICAL OPERATIONS** | Restricted classical / algebraic DSL implementing the skeleton (sums, comparisons, projections, …) — not free-form expressions. |
| **COMPUTATIONAL REPRESENTATION** | Data structures and intermediate encodings that make those operations efficient (tables, graphs, factors, …). |
| **QUANTUM ENCODING** | How computational state embeds into quantum registers / amplitudes / cost registers. |
| **REVERSIBLE OPERATIONS** | Reversible (or measured-and-uncomputed) realization of the math ops on that encoding. |
| **GATES** | Explicit gate sequences / circuit templates. |
| **RESOURCE + COMPLEXITY ANALYSIS** | Gate count \(G\), depth \(D\), qubits \(N_q\), ancillas \(N_{\mathrm{ancilla}}\), shots — and closed-form recurrence checks that reject exponential shapes *before* domain work. |

**Ordering rule (load-bearing):** complexity / budget pruning happens **before**
domain execution and **before** any quantum compilation. The contrasting
example is intentional:

\[
T(n)=2\,T(n/2)+\mathrm{poly}(n)
\quad\text{(accepted: polynomial)}
\qquad\text{vs}\qquad
T(n)=2\,T(n-1)+\mathrm{poly}(n)
\quad\text{(rejected: exponential)}.
\]

---

## 3. Implementation status (Aug 2026)

Verify claims against the pointers; do not treat this table as authority over
the code.

| Layer | Status | Pointer |
|---|---|---|
| **TARGET** | **Done** | CLI `--target-degree` / `target_degree` in `rde.synthesis.search.synthesize`; `meets_target` in `rde.synthesis.recurrence` |
| **ALGORITHM** | **Done (catalog + search)** | 10 verbs + skeleton catalog (`rde.synthesis.skeleton`); recurrence solver master / subtract / flat (`rde.synthesis.recurrence`); two-stage search (`rde.synthesis.search`) |
| **Complexity pruning (pre-domain)** | **Done** | Symbolic `solve_recurrence` **before** any domain call; tests in `tests/rde/test_synthesis_recurrence.py` |
| **Reference domain (toy)** | **Done** | `block_separable` — `rde.testing.block_separable.BlockSeparableDomain`; rediscovery tests in `tests/rde/test_synthesis_search.py` |
| **Real domain** | **Done** | `TspSynthesisDomain` (`tsp_clustered` planted / `tsp_uniform_control`) — `src/rde_domains/tsp/domain.py`; tests `tests/rde_domains/test_tsp_synthesis.py` |
| **MATHEMATICAL OPERATIONS DSL** | **Not built** | — |
| **COMPUTATIONAL REPRESENTATION** | **Not built** | — |
| **QUANTUM ENCODING** | **Not built** | — |
| **REVERSIBLE OPERATIONS** | **Not built** | — |
| **GATES** | **Not built** | — |
| **Gate-level resource accounting** (\(G\), \(D\), \(N_q\), \(N_{\mathrm{ancilla}}\)) | **Not built** | Recurrence-level \(\Theta(\cdot)\) classes exist; circuit resource ledgers for synthesized skeletons do not |
| **Protocol extensions** (treewidth / low-rank / …) | **Not built** | Domain-specific generators may support forward discovery; they are not synthesis-protocol extensions |
| **Query-tape recovery** (HSP extractors) | **Partial** | `RecoveryDomain` + collision-algebra catalog, plus depth-2 compositional chain search (`rde.recovery`, `search_space.search_recovery_chains`) over an independent discovery/confirmatory split; not divide/combine |

### Known gaps inside the implemented top

- **`subtract`-shaped skeletons** are symbolically solved and pruned, but there
  is no `SynthesisDomain` hook for size-minus-constant decomposition — they
  report `rejected_unsupported` even when they meet the complexity budget.
- Catalog stays deliberately small (`flat` / constant-branching `divide` /
  `subtract` / `base`); no search over arbitrary arithmetic expressions.
- Compilation below ALGORITHM is explicitly out of scope for the current
  implementation.

---

## 4. Two-mode architecture

### Mode 1 — Forward discovery

\[
\text{data / object} \;\longrightarrow\; \text{structure search} \;\longrightarrow\; \text{conjecture}
\]

Operationally: generate instances → materialize → descriptors/metrics → rank
→ symbolic / latent / program search → conjecture handoff. See
[methodology.md](methodology.md).

### Mode 2 — Reverse synthesis

\[
C_{\mathrm{target}} \;\longrightarrow\; \text{algorithm skeleton} \;\longrightarrow\; \text{(future: math → encoding → gates)}
\]

Operationally today: declare `target_degree` (or any polynomial) → prune
skeletons by recurrence → verify survivors against `SynthesisDomain.brute_force`.
CLI: `rde synthesize --domain <id> --size N --n-instances k [--target-degree D]`.

Use the methodology letter chain \(P\to Z\to X\to E\to S\to O\) (forward) and
\(C_{\mathrm{target}}\Rightarrow(Z,X,E,O)\) (reverse). Domain-supplied \(S\).

---

## 5. How this relates to PZXESO

Compilation depth is methodology §5, **inside Mode 2 only**. Outcome strength
is G0–G5. Reverse stress tests are V1–V4. The current implementation covers
TARGET + ALGORITHM + pre-domain recurrence pruning. Lower compilation steps
need their own design note before anything relies on them.

---

## 6. Validation stages (Mode 2)

This asks: *does backward search rediscover known structure without cheating?*
Grades G0–G5 are a different axis.

| Stage | Instance class | Intent | Status (Aug 2026) |
|---|---|---|---|
| **V1** | Known polynomial algorithms (matching, shortest path, …) **or** a toy with planted independent blocks | Rediscover a known poly skeleton; reject exponential brute force on complexity alone | **Partial:** `block_separable` toy rediscovery **done** (`tests/rde/test_synthesis_search.py`). Independent V1 on classical matching / shortest-path domains **not** built. |
| **V2** | Artificial hidden structure (planted, possibly permuted) | Structure must be *detected*, not handed pre-split | **Done:** `tsp_clustered` / `tsp_uniform_control`. Treatment accepts a decomposed skeleton when the planted cluster gap holds; control accepts nothing; a blind contiguous splitter is `rejected_incorrect`. |
| **V3** | Structured families (treewidth, low-rank, band, …) with honest size accounting | Protocol / catalog extensions beyond block-separability | **Not built** for synthesis |
| **V4** | General / dense instances | Either a verified poly skeleton or a scoped negative under an explicit catalog quantifier | **Not built** — and must not be claimed from V1 toy success |

**Non-claim:** Passing V1 on `block_separable` does **not** imply V3–V4
progress and does **not** imply a poly(\(N\)) quantum algorithm.

---

## 7. Trap: do not work backward from the answer

The hierarchy fails scientifically if the “discovered” skeleton is a disguised
copy of the optimum.

**Protocol rule** (`rde.core.protocols.SynthesisDomain`): every structural hook
(`decompose_flat`, `decompose_divide`, `combine`, `cost`, `size_of`) must be a
genuine operation on the instance — **never** derived from
`brute_force`'s own output. A domain that peeks at the oracle to choose a
partition smuggles the answer into the skeleton; acceptance then becomes a
tautology.

**Search-side guards already in place:**

1. Symbolic complexity prune **before** domain calls.
2. Independent `cost(instance, assignment)` re-evaluation — do not trust a
   carried cost from a sub-solution.
3. Cross-check against `brute_force` only as a *reference oracle* for
   verification, not as an input to decomposition.
4. Honest size parameters: reporting a coarser size than the true exponential
   base (e.g. hiding \(2^N\) behind “number of blocks”) is the same trap in
   accounting form.

Project-wide parallel: methodology §8 (do not work backward from the answer);
leak/tautology audits on forward conjectures.

---

## 8. What is already done (Mode 1 + Mode 2 top)

| Item | Status |
|---|---|
| Forward discovery apparatus | **Done** (engineering) |
| Mode 2 TARGET + ALGORITHM + recurrence engine | **Done** |
| `rde synthesize` CLI + `synthesis_conjectures.jsonl` | **Done** |
| `block_separable` reference `SynthesisDomain` | **Done** |
| Trap wording in `SynthesisDomain` protocol | **Done** |
| `tsp_clustered` / `tsp_uniform_control` treatment/control pair | **Done** |

---

## 9. Open next steps

Ordered roughly by dependency:

1. **Independent V1 domains** — matching / shortest-path (or similar known
   poly problems) as `SynthesisDomain` fixtures.
2. **MATHEMATICAL OPERATIONS DSL** — restricted language implementing accepted
   skeletons; parity tests against classical reference.
3. **COMPUTATIONAL REPRESENTATION** layer — explicit data structures for the DSL.
4. **QUANTUM ENCODING → REVERSIBLE OPS → GATES** — each step needs its own
   design note before anything relies on it; gate-level
   \((G,D,N_q,N_{\mathrm{ancilla}})\) accounting.
5. **Protocol extensions** — treewidth / low-rank / other structured
   families as synthesis hooks (V3), without weakening honest
   size / complexity reporting.
6. Keep PZXESO as the only letter vocabulary in docs; it is not a Python API.

---

## 10. Non-claims (read before citing)

- This page does **not** prove that a poly(\(N\))-gate / poly(\(N\))-shot
  quantum algorithm exists for any domain here.
- Mode 2 success on `block_separable` is a **machinery + V1 toy** result.
- Lower compilation steps (math DSL through gates) are **absent**; do not
  describe the current implementation as a circuit compiler.
- Mode 1 engineering being “done” means the **search apparatus** exists, not
  that G4–G5 outcomes were achieved on campaign science.
- PZXESO is the canonical methodology; it is not a Python API.

---

## 11. Quick command pointers

```bash
# Reference toy (Mode 2 demo)
.venv/bin/python3 -m rde synthesize --domain block_separable --size 6 --n-instances 10

# Cap accepted polynomial degree
.venv/bin/python3 -m rde synthesize --domain block_separable --size 6 --n-instances 10 --target-degree 2

# Real domain (planted vs uniform control)
.venv/bin/python3 -m rde synthesize --domain tsp_clustered --size 12 --n-instances 4
.venv/bin/python3 -m rde synthesize --domain tsp_uniform_control --size 12 --n-instances 4
```
