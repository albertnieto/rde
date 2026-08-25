# Representation synthesis: a theory of layered composition

**Status date:** August 2026. **Status tag:** `phase-1-implemented` — the
mechanism and both example compositions described here exist as code
(`rde.representation.layered`) and are tested
(`tests/rde/representation/test_layered.py`). This is Phase 1 of an
explicitly iterative plan (§5); it is not a claim that representation
synthesis is "solved."

## 1. The gap this closes

`rde.representation`'s grammar (`grammar.py`) ships seven fixed,
hand-written primitives and *ranks* them (`search.py`). That is not
synthesis — nothing about it produces a representation nobody wrote by
hand. `search.py`'s own docstring rules out the obvious next step (compose
two grammar primitives) as **mathematically vacuous**: every primitive
there has `object_type` as *both* its domain and (after `decode`) its
range — they all round-trip through the same raw vector — so

```
R_b.encode(R_a.decode(R_a.encode(v))) == R_b.encode(v)
```

for any two grammar primitives `R_a`, `R_b`. Composing two flat primitives
carries no information a direct ranking of `R_b` alone doesn't already
have.

That argument has a hole: it assumes both primitives share a domain **and**
a range. Several grammar primitives don't — `dft_full`'s carrier is `C^n`,
not `R^n`; `sorted_permutation`'s carrier is a `(values, permutation)`
pair, not a plain vector. A second stage that operates *on that carrier*
(not on the original vector) sees structure the flat, single-stage view
cannot, because it is a different mathematical object. This document
theorizes that gap and Phase 1 implements the smallest real version of it.

## 2. Formal model

A `Representation` (Phase 1 core type, `representation.py`) already has
almost everything needed: `encode`, `decode`, `distance`, `complexity`. This
theory adds exactly one thing: **typing** `encode`'s input and output by a
carrier *kind* —

```
encode : input_carrier_kind -> carrier_kind
decode : carrier_kind -> input_carrier_kind
```

`input_carrier_kind`/`carrier_kind` are free-form strings (not a closed
enum — new kinds are added by whoever writes a new primitive). Every
`grammar.py` primitive defaults to `input_carrier_kind = carrier_kind =
"real_vector"` except the three whose carrier genuinely differs:

| Primitive | `carrier_kind` |
|---|---|
| `dft`, `dft_full` | `"complex_vector"` |
| `matrix_reshape` | `"matrix"` |
| `sorted_permutation` | `"sorted_pair"` |

**Composition.** Given `stage1: Representation` and `stage2: Representation`,
`stage2` composes onto `stage1` *iff*

```
stage2.input_carrier_kind == stage1.carrier_kind
```

and the composed representation is

```
encode(x) = stage2.encode(stage1.encode(x))
decode(y) = stage1.decode(stage2.decode(y))
```

which is itself a plain `Representation` over `stage1.object_type` — the
composed thing is not a new kind of object, it is exactly as usable as any
Phase 1 representation (rankable, certifiable, reportable, holdout-testable)
because it *is* one. This is why the implementation
(`layered.compose_layers`) does not introduce a parallel class hierarchy.

**Why this composition is not vacuous.** Unlike the flat case, `stage2`
operates on `stage1`'s carrier, which is a genuinely different object (a
complex vector, a sorted-values/permutation pair) than the thing `stage1`
started from. There is no flat primitive in the grammar equivalent to
`stage2` applied to that carrier — `sort_by_magnitude` (defined only for
complex vectors) has no meaning as a flat primitive on the original real
vector at all.

## 3. What Phase 1 actually ships

Two stage-2 primitives, one per carrier kind the grammar currently produces
that isn't already terminal:

- **`sort_by_magnitude`** (`complex_vector -> sorted_complex_pair`): sort
  `dft`/`dft_full` coefficients by `|value|` descending, keeping the
  permutation. Separates *which* frequencies dominate from *how much*.
- **`sorted_then_difference`** (`sorted_pair -> sorted_pair`): delta-encode
  the already-sorted values from `sorted_permutation`, carrying the
  permutation through unchanged. Sorted data is monotonic, so consecutive
  differences are small within a cluster and large only at cluster
  boundaries.

`matrix_reshape`'s `"matrix"` carrier gained a stage-2 primitive later —
`row_dft`, a row-wise transform, see §9. `dft` (compact, non-square) shares `"complex_vector"` with
`dft_full` and composes with `sort_by_magnitude` too, for the same reason
`operator.py` avoids probing `dft`'s decode side: `sort_by_magnitude`'s
decode only needs `dft`/`dft_full`'s own `decode`, which is already exact
for both — no matrix-probing involved, so `dft`'s known compact-carrier
fragility (see `operator.py`) doesn't apply here.

`build_layered_representations(n, ...)` enumerates every valid
`(stage1, stage2)` pair exhaustively — with 7 grammar primitives and 2
stage-2 primitives each accepting one carrier kind, there are at most a
handful of valid pairs, the same "exhaustive is already optimal" reasoning
`search.py` gives for depth-1 ranking.

## 4. Honest empirical validation

Every claim below was checked numerically before being written down, using
the real `Representation.complexity` function (which charges for storing
*both* the values and the permutation — an earlier draft of this section
measured only the diffed values in isolation and overclaimed a win that
disappeared once the permutation's own storage cost was counted; that
mistake is documented here on purpose, not quietly fixed).

**Roundtrip exactness.** Both compositions round-trip to machine precision
(`~1e-16`–`1e-17` max error) across every tested scenario — clustered data,
skewed categorical data, multi-frequency signals. The typing contract in
§2 (`decode` returns to `input_carrier_kind`, not all the way to the raw
object) is what makes this compose correctly; getting it wrong (an early
draft had `sorted_then_difference.decode` fully undo the permutation itself)
silently corrupts the result when `stage1.decode` runs a second time on
already-restored data — caught by testing against real clustered data
before shipping, not by inspection.

**Complexity — the honest result.** Under `grammar.py`'s standard
sparsity-based complexity metric (`sparsity_fraction`, `eps=1e-6` for
values, `eps=0.5` for permutation indices):

- `sorted_permutation+sorted_then_difference` **beats
  `sorted_permutation` alone in every tested scenario** (clustered data,
  skewed categorical data, sparse binary data, at `n` from 4 to 64) —
  confirming the core mechanism: once you've committed to carrying a
  permutation, delta-encoding the sorted values is strictly better than
  not.
- It does **not** beat the permutation-free baseline `identity` in any
  tested scenario. Neither does `dft_full+sort_by_magnitude` beat plain
  `dft_full`. The reason is structural, not a tuning problem: a
  permutation of `n` elements has essentially no near-zero entries under
  `sparsity_fraction`'s per-element threshold (only index `0` can read as
  "insignificant"), so it is charged close to the full `n` regardless of
  whether the permutation itself has exploitable structure (e.g. is close
  to the identity permutation, or has long sorted runs).

**What this means.** The composition mechanism is correct and the theory's
core claim (layering exposes real structure a flat view cannot) is
verified — but a fair *comparison against permutation-free representations*
needs a complexity metric that can recognize a cheap-to-describe
permutation, which `sparsity_fraction` cannot. That is a real, scoped gap,
not a reason to distrust the mechanism — **closed in §7**, using the same
`_clustered_batch`-style data this section already validated against (the
negative result above still holds for genuinely i.i.d.-shuffled data; §7 is
about the cases it was previously unable to tell apart from that one).

## 5. What Phase 1 deliberately does not do (later iteration)

- **No permutation-aware complexity metric.** `sparsity_fraction` treats
  any non-trivial permutation as maximally complex. A metric based on
  inversion count, run-length structure, or `log2(n!)` scaled by how far
  the permutation is from sorted/identity would let `sorted_permutation`
  and its layered derivatives compete honestly against permutation-free
  representations. Not built — this is the most concrete, highest-value
  next step.
- **`"matrix"` stage-2 primitive: done, see §9.** `row_dft`, a per-row full
  FFT, exposes 2D-periodic structure a flat 1-D view of the same `n` values
  cannot.
- **Depth-3+ composition: done, see §6.** `program_search.py`'s
  `enumerate_chains`/`search_chains` generalize `compose_layers` to any
  `max_depth` via exhaustive DFS over the same carrier-kind compatibility
  graph — no new stage-3-specific primitive was needed, because a composed
  `Representation` is a plain `Representation` and can itself be `stage1`
  of a further composition (this is exactly what "no depth-3 stage-3
  primitive exists" above was wrong to treat as a blocker for).
- **Canonicalization: done.** `pareto.canonical_representation` picks one
  representative from the Pareto frontier via one explicit, disclosed rule
  (lowest complexity, then conversion cost, then roundtrip error, then id).
  No new machinery — `pareto_rank` already computed the frontier.
- **No genuinely novel primitive invention.** Every stage-1 and stage-2
  primitive is still hand-written. `program_search.py` closes the "compose
  known building blocks *deeper*" gap, not the "invent a building block
  nobody wrote" gap — that remains open-ended research, not a Phase 2 item
  with a clear scope.
- **Permutation-aware complexity metric: done, see §7.**
- **`"matrix"` and `"sorted_complex_pair"` stage-2 primitives: done, see §9.**
- **Wired into `search.py`'s ranking, `holdout.py`'s audit, and
  `report.py`'s persistence: done, see §10.**

## 6. Depth-K composition and mandatory generalization checking (`program_search.py`)

`enumerate_chains` exhaustively enumerates every valid typed chain up to a
caller-chosen `max_depth`, by repeatedly applying `compose_layers` — no
depth-specific code, since a chain's own `carrier_kind` is just another
carrier kind a further stage can match against. `search_chains` layers
mandatory holdout checking on top (same discipline as `holdout.py`'s
`leakage_ratio`): both `train_batch` and an independent `holdout_batch` are
required (not optional), a chain that fails to verify (exact roundtrip) on
*either* is dropped entirely, and surviving chains are ranked by *holdout*
complexity, never train — `generalization_ratio` reports how much of the
apparent compression held up.

**A genuine depth-3-only result**, verified numerically before being written
into `tests/rde/representation/test_program_search.py`: for data that is
piecewise-linear once sorted (two linear segments joined at one knot),
`sorted_permutation+sorted_then_difference+sorted_then_difference`
(second-order differencing of the sorted values) reaches a lower complexity
than the depth-2 chain `sorted_permutation+sorted_then_difference` — e.g.
`22.0 -> 14.0` for one tested batch. This is not a reproduction of anything
§4 already found; §4's depth-2 chains were checked against clustered,
skewed-categorical, and multi-frequency data, none of which has the
second-order-flat structure this result depends on.

**Exhaustive DFS, not genetic programming.** Checked before building this:
the registry's carrier-kind compatibility graph is small and sparse — most
carrier kinds (`"matrix"`, `"sorted_complex_pair"`) have zero registered
continuations, `"sorted_pair"` has exactly one (itself). Exhaustive
enumeration up to `max_depth` is already optimal at today's registry size;
a population/mutation/crossover search would be complexity for its own
sake here. Revisit if the registry grows enough that exhaustive search
stops being cheap.

**A real bug this surfaced and fixed.** `dft_full`'s `decode` (`ifft`)
always returns complex-dtyped data, even when the underlying values are
mathematically real (near-zero imaginary part) — correct and necessary for
`operator.py`'s circulant-diagonalization proof, which needs `decode` to
stay the genuine, unrestricted complex inverse-DFT matrix. `layered.py`'s
own depth-2 tests never exposed this because they only ever used `dft_full`
as a chain's *last* stage feeding `sort_by_magnitude` (which handles
complex data natively). `program_search.py`'s deeper enumeration produced
chains where `dft_full`'s complex-dtyped decode output flowed into an
earlier stage's decode (`difference`'s `cumsum`, `polynomial_vandermonde`'s
`matmul_shared`) that unconditionally cast to real dtype via
`np.asarray(x, dtype=float)` — NumPy's implicit warn-then-truncate on that
cast is what surfaced the bug (`ComplexWarning`). Fixed in
`array_backend.py` by making `cumsum`/`matmul_shared` (both backends)
explicitly discard the imaginary part via a named `_to_real` helper instead
of relying on an implicit cast — `dft_full.decode` itself was deliberately
left untouched, since changing it would have broken `operator.py`'s probe.
Any genuine roundtrip failure this coercion could mask is still caught by
`certify_roundtrip`'s distance check against the original object.

**Identity-padding is deduplicated, not just harmless.** `enumerate_chains`
now drops any chain whose deepest stage is `identity` when a shorter chain
already reaches the same `(carrier_kind, encode/decode behavior)` — see §8.

## 7. Permutation-aware complexity metric (`array_backend.permutation_complexity`)

§4 found a real limitation, not just an unflattering number: charging every
permutation `sparsity_fraction(perm, eps=0.5)` counted every entry except
value `0` as "significant" — since permutation entries are exactly the
integers `0..n-1`, this charged **every** permutation close to `n`,
identity included. It never actually measured the permutation's structure,
so `sorted_permutation`-based chains could never beat a permutation-free
baseline even when the underlying sort order genuinely was cheap to
describe (e.g. already close to sorted).

**The fix.** `permutation_complexity(perm)` (new `ArraySearchBackend`
method, both backends): counts positions where `perm[i+1] != perm[i] + 1`
— the number of maximal contiguous ascending-by-1 runs, minus one. Identity
costs `0`; a permutation made of `k` contiguous ascending blocks (e.g. two
blocks swapped) costs `k - 1`; a maximally scrambled permutation costs up
to `n - 1` — the same worst case as the old charge, but now correctly `0`
in the best case rather than ~`n - 1` always. `grammar.py`'s
`sorted_permutation` and `layered.py`'s two stage-2 primitives all switched
to it; `_sparsity_complexity`'s `eps=1e-6` handling of the *values* half is
unchanged.

**Verified effect, both positive and honest-negative.**

- §4's negative result (`_clustered_batch`, i.i.d.-shuffled values) is
  unchanged — the sort permutation for genuinely random-order data is
  close to uniformly random, which costs close to the maximum under *any*
  permutation-complexity metric (an information-theoretic floor, not an
  artifact of the metric being replaced). `sorted_permutation`-based chains
  still do not beat `identity` here, correctly.
- **New, genuine win**: for data whose original order is already close to
  sorted (e.g. two joined linear segments, ascending), the sort permutation
  costs ~`0`, and `sorted_permutation+sorted_then_difference+
  sorted_then_difference` (depth-3 second-order differencing of the sorted
  values) beats plain `identity` outright — `3.0` vs. `11.0` for one tested
  batch (`tests/rde/representation/test_program_search.py::
  test_depth_3_self_chained_difference_beats_identity_when_permutation_is_cheap`).
- **Honest limit found while looking for a stronger claim.** Tried several
  scenarios (already-sorted original order, a cheap two-block swap) looking
  for a case where a `sorted_permutation`-based chain *strictly* beats the
  permutation-free equivalent (`identity+difference+difference`, no sort at
  all) — every one tied, none won. The data property that makes sorting
  cheap (original order already close to sorted, or a simple block swap)
  is, in this grammar, also exactly the property that makes plain
  differencing on the *original* order already cheap — so carrying a
  permutation adds no value in the cases checked. Whether some other data
  shape breaks that pattern is open; not claimed either way.

## 8. Identity-padding dedup in `enumerate_chains`

`enumerate_chains` recursed through every registered continuation at each
depth, including `identity` — so `max_depth=4` produced chains like
`identity+identity+identity+matrix_reshape`, structurally distinct
`representation_id`s that behave identically to the depth-1 `matrix_reshape`
chain (an `identity` stage's `encode`/`decode` are both the identity
function, so composing it anywhere past the first stage changes nothing
observable). `identity` is now excluded from `continuations` entirely —
never appended as a non-first stage — while remaining a valid `start` (the
one legitimate `real_vector -> real_vector` no-op chain, depth 1). Verified
numerically: `enumerate_chains(16, max_depth=4)` drops from `324` to `168`
chains, zero of which contain `identity` as a non-first stage, and every
previously-verified result in this document (§4's, §6's, §7's) is
unchanged — `test_program_search.py`'s existing assertions on specific
`representation_id`s still pass unmodified.

(The `168` count is specific to the registry at the time this section was
written — two stage-2 primitives, `sort_by_magnitude` and `sorted_then_
difference`. §9 adds two more, which grows `enumerate_chains(16,
max_depth=4)` to `188`; §11 adds a new *stage-1* primitive (`dct`), which —
unlike a stage-2 addition — multiplies the count further, since `dct` is
both a new `start` and a new same-carrier-kind `continuation` for every
existing `real_vector` chain: `enumerate_chains(16, max_depth=4)` grows to
`458`. The dedup mechanism itself is unaffected either time, since it
excludes `identity` unconditionally regardless of what else is registered.)

## 9. `"matrix"` and `"sorted_complex_pair"` stage-2 primitives (`row_dft`, `sorted_complex_then_difference`)

§5 flagged `matrix_reshape`'s `"matrix"` carrier and `sort_by_magnitude`'s
`"sorted_complex_pair"` carrier as the two remaining dead ends in the
registry's carrier-kind graph. Both now have a stage-2 primitive.

**`row_dft`: `matrix -> complex_matrix`, a per-row full complex FFT.**
`fft`/`ifft` (`array_backend.py`) already operate on an array's last axis
regardless of rank, so applying them to `matrix_reshape`'s `(B, side,
side)` carrier is a genuine per-row transform with no new backend kernel —
only the `real_vector`-typed stages (`matmul_shared`/`cumsum`/`diff_with_
first`) assume a flat last axis of length `n`, and `row_dft` uses none of
those. Verified numerically: for `matrix_reshape`'s `(4, 4)` carrier with
every row an identical single-frequency signal (2D-periodic-by-construction
data), `matrix_reshape+row_dft` reaches complexity `7.5` against `identity`/
`matrix_reshape`'s `9.0` — an exact roundtrip (`~1e-32` error) and a real
compression win, not merely a non-worse tie
(`tests/rde/representation/test_layered.py::
test_row_dft_beats_flat_identity_on_row_periodic_matrix_data`). Consistent
with the rest of this document's honesty discipline: this is a claim about
data with genuine row-periodic structure, not a claim `row_dft` beats a
flat `dft`/`dft_full` on arbitrary data — tiling identical rows also makes
the *whole* `n`-vector periodic, so a flat `dft` picks up the same
structure too (checked while building this; not written into a test as a
positive claim since it is data-construction-dependent, not a property of
`row_dft` itself).

**`sorted_complex_then_difference`: `sorted_complex_pair -> sorted_complex_pair`.**
Same mechanism as `sorted_then_difference` one layer up, with one real
correction: `sort_by_magnitude`'s values are genuinely complex (a DFT
coefficient's phase, not a decode artifact like `dft_full.decode`'s
always-complex-dtype output — see `_to_real`'s docstring), so this uses two
new backend kernels, `diff_with_first_complex`/`cumsum_complex`, that do
*not* discard the imaginary part the way `diff_with_first`/`cumsum` do by
design. Using the real-only pair here would have silently corrupted the
roundtrip instead of just failing to compress — caught by reasoning about
the carrier's actual dtype before writing the primitive, not by a test
failure after the fact.

Verified numerically: zero-phase cosine components make each DFT
conjugate-pair bin `X[k]`/`X[n-k]` purely real and numerically *identical*
(`X[n-k] = conj(X[k]) = X[k]` when `X[k]` has no imaginary part) —
`sort_by_magnitude` places those equal-valued bins adjacent, so
differencing them is exactly zero. For such data (`n=16`, five equal-
amplitude zero-phase frequencies), `dft_full+sort_by_magnitude+
sorted_complex_then_difference` reaches complexity `17.0` against the
depth-2 `dft_full+sort_by_magnitude`'s `25.0` — exact roundtrip preserved
(`~1e-15`)
(`tests/rde/representation/test_program_search.py::
test_sorted_complex_then_difference_beats_depth_2_on_conjugate_paired_spectrum`).
A same-frequencies-but-random-phase variant of the same data does **not**
show this win (checked while building this) — the conjugate pairs are no
longer numerically equal once phase varies, confirming the win is a real
structural property of zero-phase data, not an artifact of the metric.

## 10. Wired into `search.py`'s ranking, `holdout.py`'s audit, and `report.py`'s persistence

§5 noted that `build_layered_representations`'s and `program_search
.enumerate_chains`'s output were plain `Representation` objects that *could*
flow through `search.py`/`holdout.py`/`report.py` if a caller mixed them in,
but no call site did. Three changes close that:

- **`search.py`.** `rank_representations`/`best_representation` gained a
  `chain_max_depth: int | None = None` parameter. Left `None` (the
  default), behavior is byte-for-byte unchanged from before this parameter
  existed — no `program_search` import even happens, so this is additive,
  not a behavior change for any existing caller. Set to an integer, it
  ranks `program_search.enumerate_chains`'s output (which already includes
  every flat-grammar primitive as its own depth-1 chain) instead of the
  flat grammar alone.
- **`holdout.py`.** `audit_holdout` gained the same `chain_max_depth`
  parameter, forwarded to both its visible-only and full rankings. This
  also surfaced a real gap in `discovered_held_out_structure`'s original
  definition (exact `full_best.representation_id == a held-out name` match)
  — meaningless for a composed id, which will essentially never equal a
  single primitive's name even when it genuinely uses one as a stage. Fixed
  to check whether any held-out primitive appears among `full_best.
  representation_id.split("+")` — a strict generalization that leaves
  depth-1 (flat grammar) behavior identical (a flat id has no `"+"` to
  split) while making the check meaningful for chains. Verified
  numerically: on already-ascending piecewise-linear data, `identity+
  difference+difference` and `sorted_permutation+sorted_then_difference+
  sorted_then_difference` tie at complexity `3.0` (a near-identity sort
  permutation costs ~0, so both paths end up equivalent) — withholding
  plain `difference` still lets the visible-only chain search match that
  complexity through the `sorted_permutation` route: a genuine, partial
  leak this audit reports as a number (`leakage_ratio ~0.27`), not a
  pass/fail flag hiding it
  (`tests/rde/representation/test_holdout.py::
  test_audit_holdout_chain_max_depth_discovers_held_out_structure_reached_only_via_a_chain`).
- **`cost.py`.** A real blocker, not a documentation gap: `computational_
  cost` raised `KeyError` for any `"+"`-joined composed id, since its
  primitive registry only ever listed `grammar.py`'s stage-1 names —
  `rank_representations` would have crashed the moment `chain_max_depth`
  produced its first composed candidate. Fixed by adding cost estimates for
  `layered.py`'s four stage-2 primitives and making `computational_cost`
  split a composed id on `"+"` and sum each stage's own cost — each stage's
  encode/decode genuinely runs in sequence, so the total operation count
  genuinely is additive, not a new estimate needing invention.
- **`program_search.py`.** `atomic_registry`/`enumerate_chains`/
  `search_chains` gained `primitive_subset` (previously only `search.py`'s
  flat-grammar ranking had it) — withholding a stage-2 primitive now
  genuinely removes every chain that would have used it at any depth, not
  just its own depth-1 entry, which is what makes `holdout.py`'s
  chain-aware audit above a real ablation rather than a partial one.
- **`report.py`.** Needed no changes at all — `search_report_payload`
  already operates generically on `SearchCandidate.representation_id`
  (a string) and `.complexity`/`.conversion_cost` (floats), so a composed
  chain candidate serializes exactly like a flat one once `cost.py` can
  price it
  (`tests/rde/representation/test_representation_report.py::
  test_search_report_payload_serializes_composed_chain_candidates`).

## 11. A genuinely new stage-1 primitive: `dct` (type-II Discrete Cosine Transform)

Every primitive through §10 *composes* the same fixed seven-primitive
grammar; this section adds an eighth, standing on its own — the honest
scope agreed before building it: not "invent a building block nobody has
ever described" (open-ended, no finish line), but "add one well-known,
textbook-standard primitive the existing grammar structurally cannot
express, for a concrete reason." `grammar.py`'s existing basis-transform
primitives (`dft`, `dft_full`) both use integer-frequency sinusoids, exactly
periodic in the `n`-window; `dct` uses half-integer-frequency cosines,
which are not — a genuinely different linear basis, not a relabeling of one
already present.

**Implementation.** `_dct_matrix(n)` builds the orthonormal type-II DCT
basis matrix directly (`C[k, m] = alpha_k * cos(pi/n * (m + 0.5) * k)`),
verified to match `scipy.fft.dct(..., type=2, norm="ortho")` exactly before
being written
(`tests/rde/representation/test_grammar.py::test_dct_matches_scipy_dct_ii_ortho`).
Because `C` is orthogonal (`C @ C.T == I`, `cond(C) == 1` for every `n`
tested), `C.T` *is* the exact inverse — reusing `matmul_shared` the same way
`polynomial_vandermonde` does, but with no `np.linalg.inv` call and none of
that primitive's famous ill-conditioning (`cond(V) ~ 1e8` at `n=8`; `dct`'s
condition number is exactly `1` regardless of `n`, verified for `n=8, 16,
32`). Stays entirely real-valued — no `"complex_domain"` tag, unlike
`dft_full`.

**Honest empirical validation, including a disproven first attempt.** The
first claim written for this primitive was "beats `dft` on a smooth,
non-periodic ramp" at a loose `eps=1e-2` sparsity threshold (`dct`: `62.5%`
of coefficients significant; `dft`: `100%`). That claim quietly disappeared
once checked against the grammar's actual `eps=1e-6`: a ramp's DCT
coefficients decay but never hit exact (or near-`1e-6`) zero, so all `16`
remain "significant" under the real threshold for *both* bases — the same
mistake §4 documents on purpose rather than quietly fixing, not asserted
here either. The real, `eps=1e-6`-honest result: data built as an exact
2-of-16-sparse combination of `dct` basis vectors (`coeffs @ C` for a
2-hot `coeffs`) reaches `dct` complexity `2.0` against `identity`'s `16.0`,
`dft`'s (`rfft`) `7.0`, and `dft_full`'s `13.17` — exact roundtrip
preserved, and a real, decisive, structural difference (a DCT-basis
combination is not close to sparse in the DFT domain, and vice versa), not
a tuning artifact
(`tests/rde/representation/test_grammar.py::
test_dct_beats_dft_and_identity_on_dct_sparse_data`).

**Downstream cost.** `dct` shares `grammar.py`'s default `real_vector`
carrier kind, so it is automatically both a valid `enumerate_chains` `start`
*and* a valid same-carrier-kind `continuation` for every other
`real_vector`-carrier chain — unlike §9's stage-2 additions (which only
grew the registry additively), a new stage-1 primitive multiplies the chain
count: `enumerate_chains(16, max_depth=4)` grows from `188` to `458` (see
§8's addendum). `cost.py` needed one addition (`dct` costs `~2*n^2`, the
same dense-matmul estimate as `polynomial_vandermonde`, since this
implementation is a literal orthonormal-matrix matmul, not the `O(n log n)`
fast-DCT algorithm textbooks describe — the honest cost of the code that
actually runs).

## 12. A genuine topological homeomorphism: `(C\{0})/Z_n ≅ C\{0}` via `z ↦ z**n`

`equivalence_types.py` had explicitly declined to claim topological
homeomorphism: `RepresentationGraph` is a discrete labeled graph, not a
topological space, so "continuous bijection with continuous inverse" had
nothing real to check it against. Same honest-scope move as §11's `dct`:
not "prove homeomorphism in general" (open-ended, no finish line — checking
it on this package's own linear grammar primitives would be vacuous, since
every invertible linear map between finite-dimensional vector spaces is
automatically a homeomorphism), but one concrete, non-vacuous textbook fact.

**The claim.** `X = C \ {0}` (standard/Euclidean topology), `G = Z_n` acting
by multiplication by n-th roots of unity — the continuous analogue of
`structure._cyclic_group_permutations`'s discrete index-permutation action,
and literally the same rotation group `rde_domains.tsp.circulant`'s
`_points_on_circle` already uses to plant circulant symmetry
(`theta = 2*pi*k/n`). The map induced by `f(z) = z**n` on the orbit space
`X/G` is a homeomorphism onto `X` — a finite branched-covering / orbifold
quotient, standard in any topology text, implemented and checked here rather
than asserted.

**Numeric verification (`topology.py`, `HomeomorphismClaim`).**
`check_quotient_well_defined` confirms `f` is constant across sampled `Z_n`
orbits (max spread `~1.7e-19` for random complex samples at `n=6` —
floating-point roundoff, not exactly `0`, as expected for a numeric check of
an algebraic identity). `check_quotient_injective_sampled` finds `0`
collisions among `19900` distinct-orbit pairs from `200` random samples at
`n=6` — honest sample-based evidence, not a proof.
`check_map_continuity_sampled` confirms no blowup under finite-difference
perturbation against a local-Lipschitz bound.

**Honest disproven first attempt.** The obvious way to write "the inverse"
is a single-valued principal branch on a fundamental domain (angle in
`[0, 2*pi/n)`, principal n-th root). `check_naive_branch_inverse_discontinuous`
verifies this is *wrong*: two points `epsilon`-close to the domain seam have
images `f(z)` that are `~1.2e-8` apart (`f` correctly nearly-identifying
them), but the naive branch inverse recovers points `~1.0` apart — a real,
verified discontinuity, not a tuning artifact. This is exactly why the
homeomorphism claim must be stated on the *abstract* quotient space (orbits
as points, quotient topology), never by picking an explicit global inverse
on a literal fundamental-domain subset of `C` — the same distinction
`equivalence_types.py` had previously flagged as unimplemented.

**Formal verification (`symbolic.FormalCertificate`, reused, not
duplicated).** `prove_rotation_quotient_well_defined(n)` proves
`(zeta*z)**n == z**n` exactly via `sympy.simplify`, closing to `0` for every
`n` tried (`3` through at least `8`). `prove_rotation_quotient_injective(n)`
proves the classical factorization
`z1**n - z2**n == prod_{k=0}^{n-1}(z1 - zeta**k * z2)` via
`sympy.nsimplify(sympy.expand_complex(...), [sympy.pi], rational=False)`.
Closes exactly for `n = 3, 4, 5, 6, 8` — but **not** `n = 7`:
`cos(pi/7)`/`cos(2*pi/7)` are not expressible in the real-radical normal
form this `nsimplify` strategy searches for (a genuine Galois-theoretic
obstruction — 7 prime, no solvable radical tower of the kind `nsimplify`
looks for — not a numerical bug). Scoped per-`n`, `proved`/`disproved`, the
same honest limitation `prove_vandermonde_inverse(n)` already accepts by
only ever claiming one concrete `n` at a time; the `n=7` case is kept as a
disclosed limitation in the test suite rather than quietly excluded.

**What this does not claim.** Applying this same quotient/homeomorphism
machinery to the actual `tsp_circulant_symmetry` distance-matrix orbit space
— a *discrete* permutation group acting on `R^{n x n}`, not a continuous
rotation acting on `C` — is a harder, separate case and remains open.

## 13. A bounded instance of open-ended primitive invention: `adaptive.py`'s `klt`

§5 flagged "no genuinely novel primitive invention... that remains
open-ended research, not a Phase 2 item with a clear scope." §11's `dct`
closed one instance of that gap for *analytic* primitives; this section
closes an instance for *data-adapted* ones, same honest-scope move: not
"invent any building block" (no finish line), but the Karhunen-Loeve
theorem — a textbook fact — the covariance eigenbasis is the optimal
orthonormal linear basis for concentrating a distribution's variance into
the fewest coefficients, which is exactly what this grammar's
near-zero-coefficient `complexity` metric rewards.

**Architectural boundary, kept explicit.** Every `grammar.py` primitive
(`dft`, `dct`, ...) is analytic: constructible from `n` alone.
`adaptive.build_klt_representation` needs a training batch, so it is
deliberately *not* added to `grammar.py`'s `_PRIMITIVE_BUILDERS` registry —
`program_search.enumerate_chains` and every other caller that assumes "any
grammar primitive is buildable from `n` alone" would silently break
otherwise. It stays standalone, the way `operator_discovery.py` (recovers
an unknown operator from samples) stays standalone from `operator.py`
(transports an already-known one). Strictly linear (no mean subtraction —
the eigenbasis of the sample *second moment*, not the centered covariance),
so `equivalence_types.py`'s linear/isometry/isomorphism probes apply to it
unmodified, and matching every other primitive's `"linear"` tag.

**The preregistered comparison, decided before being run.** Does a `klt`
basis fit on a `train_batch` beat the best of the 7 grammar primitives that
verify their own roundtrip at `n=16` (`polynomial_vandermonde` fails its own
tolerance at this `n`, unrelated to this comparison) on an *independent*
`holdout_batch` from the same distribution, by at least `2x` lower
complexity? Target family: `x = z @ A.T`, `A` a fixed `(16, 3)` matrix with
orthonormal columns (QR of a Gaussian draw), `z ~ N(0, I_3)` — an exact
rank-3 Gaussian factor model no fixed analytic basis has any structural
relationship to, since `A`'s column space is random and shares no structure
with any basis's fixed coordinate/frequency axes. `train_seed=0`,
`holdout_seed=1`, `500` samples each, `margin_threshold=0.5`. One run, no
re-rolling to fish for a better margin — the same discipline
`rde_domains.hsp_functions.preregistered_experiment` already holds itself to.

**Actual result**, verified numerically before being written here: `klt`
reaches holdout complexity `3.0` — exactly `k`, the true subspace dimension
— against `dft`'s `9.0` (the best of the verified fixed primitives). Ratio
`0.333`, comfortably under the `0.5` margin
(`tests/rde/representation/test_adaptive.py::
test_preregistered_klt_holdout_comparison_beats_the_margin`). Every other
fixed primitive does no better than `dft`: `identity`, `matrix_reshape`,
`dft_full`, and `dct` all report the full `16.0` (a random 3-dimensional
subspace has no relationship to the identity basis, the full complex
Fourier basis, or the DCT basis), and `difference`/`sorted_permutation` are
worse still.

**Honest negative result, not swept under the rug.** The same comparison
with i.i.d. Gaussian noise added to `x` (`noise_scale=0.05`, still rank-3 in
expectation) shows *no* compression for `klt` under this grammar's real
`eps=1e-6` threshold: every coefficient, including the 13 "noise"
directions, exceeds `1e-6` in magnitude almost surely, so `klt` reports the
full `16.0` — identical to `identity`
(`tests/rde/representation/test_adaptive.py::
test_noisy_variant_shows_no_compression_for_anyone_honest_negative_result`).
The same class of mistake §11 documents on purpose for `dct`'s first
loose-`eps` claim: this grammar's complexity metric only rewards *exact*
near-zero coefficients, not *approximately* low rank under noise (a
genuinely different, harder notion — truncation with a disclosed nonzero
reconstruction error, which none of this grammar's `exact=True` primitives
attempt). The preregistered target family is deliberately noise-free
specifically to stay honest about what this metric can actually show.

**Characterizing the breakdown, not just one negative data point.**
`run_klt_noise_sensitivity` sweeps `noise_scale` across `(0, 1e-7, 5e-7,
1e-6, 5e-6, 1e-5, 1e-4, 1e-3, 1e-2)` — three orders of magnitude either side
of `eps=1e-6` — and finds a real, smooth transition, not a step function:
complexity stays pinned at the noise-free `3.0` through `1e-7` (an order of
magnitude below `eps`), rises to `~3.6` at `5e-7`, `~7.1` right at `eps`
itself (`1e-6`), `~14.9` at `1e-5`, and asymptotes toward (without exactly
reaching) `16.0` from `1e-4` on — exactly the shape Gaussian coefficient
magnitudes crossing a fixed bar should produce
(`tests/rde/representation/test_adaptive.py::
test_noise_sensitivity_sweep_is_flat_below_eps_and_saturates_above_it`). A
`train_count` sweep (`500` down to `3`, the minimum that spans a rank-3
subspace) was also checked, at `noise_scale=0.0`, and found completely
flat — an honest non-result kept in the record rather than presented as the
headline finding it was originally expected to be: with zero noise, any `k`
linearly independent training samples exactly span the true subspace, so
there is no estimation variance for a larger `train_count` to reduce
(`tests/rde/representation/test_adaptive.py::
test_train_count_does_not_affect_the_noise_free_result`).

**Robustness fixes made alongside this characterization.**
`_best_grammar_holdout_complexity` (factored out of
`run_klt_holdout_comparison`, now shared with `run_klt_noise_sensitivity`)
raises a `RuntimeError` naming the actual problem if no grammar primitive
ever verifies its own roundtrip against a holdout batch, rather than
`min()` on an empty dict raising `ValueError: min() arg is an empty
sequence` three frames deep — not reachable at `n=16` today (`7` of `8`
primitives verify), forced via monkeypatch in
`tests/rde/representation/test_adaptive.py::
test_best_grammar_holdout_complexity_raises_a_clear_error_when_nothing_verifies`
rather than left unexercised.

**What this does not claim.** A single bounded instance, not a general
"discover any useful data-adapted basis" capability — no other family,
covariance structure, or non-Gaussian distribution has been tried. `klt`
also cannot be composed via `program_search.enumerate_chains` (the
architectural-boundary paragraph above), so it does not participate in
depth-`K` composition search the way every `grammar.py` primitive does.
