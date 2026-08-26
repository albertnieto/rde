# EXP-001 — approximate-label abelian HSP recovery

## Question

Within the declared finite horizon, can a recovery grammar discover a
protocol that recovers the period of an additive-cyclic hidden subgroup when
coset labels are not equal but are Hamming-near?

This is a Mode 2 `RecoveryDomain` experiment.  It is a bounded-query V2 test
of a new recovery primitive in RDE; it is not evidence of a new general HSP
algorithm, a quantum algorithm, or a theorem.

## Frozen problem and resource model

- Groups: additive cyclic groups `Z/(2**n)Z`, with `n in {8, 10, 12, 16, 20,
  24}`.
- Query budget: `20*n**2` uniformly sampled oracle queries per instance.
- Positive discovery family: `approximate_cyclic_period`.  Its subgroup order
  is independently drawn from `{4, 8, 16}` and the planted object is the
  resulting period `r = 2**n / order`.
- Positive held-out family: `approximate_cyclic_period_alt`.  It has the same
  period distribution but a separately specified two-bit per-coset-member
  perturbation rather than the discovery family's one-bit perturbation.
- Calibration/pipeline families: exact `simon`, `shor_cyclic`, and
  `dihedral_kuperberg`.
- Additional discovery coverage: `approximate_xor_shift`.
- Negative control: `generic_random_control` with the same 63-bit label
  representation but no planted object.
- Labels are 63-bit opaque values.  The only non-exact observation available
  to the grammar is Hamming distance at most four.  The near-neighbor index
  is a five-block pigeonhole index, expected linear in the tape length for
  random labels and cached once per tape.

## Candidate grammar and split

The candidate set is frozen before execution: all depth-1 recovery programs
from `enumerate_recovery_chains(max_depth=1)`, including the generic
`near4_<bag>_<reducer>_<post>` programs.  No family name, planted value, or
family-specific candidate enters `RecoveryProtocol.extract`.

Even seeds are the discovery split.  A candidate is selected only when it has
at least 0.80 recall for `approximate_cyclic_period` at each discovery size
`{8, 10, 12}` and at least 0.95 control specificity at those sizes.  The
selected candidate identities are frozen before the held-out evaluation.

Odd seeds are the confirmatory split.  The held-out family is evaluated only
with those discovery-selected candidates, at every confirmatory size.

## Decision rule

The outcome is `SIGNAL` (grade G1) iff all of the following hold:

1. exact pipeline recall is at least 0.80 for each textbook family at each
   confirmatory size;
2. at least one non-textbook candidate is selected on the discovery family;
3. at least one selected candidate has at least 0.80 recall on the held-out
   perturbation family at every confirmatory size; and
4. control specificity is at least 0.95 on both splits at every tested size
   for that held-out candidate.

Otherwise the outcome is `NULL` (G0).  A run records all candidate rates,
including failures and abstentions.  No retries with changed labels, radius,
catalogue, seeds, sizes, or thresholds are permitted under this registration.

## Stop rule

Run the registered command once with 100 instances per family per size.  A
completed `NULL` is the result.  Any changed noise model, resource budget, or
candidate grammar requires a new experiment directory and registration.
