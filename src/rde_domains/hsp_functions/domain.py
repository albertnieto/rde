"""RDE domain: hidden-subgroup-style oracle-function family discovery (Direction F).

World layer: `functions.make_instance` generates one analytic,
query-evaluable oracle function per instance. The default roster is the
original six families — three exact calibration cases held out of discovery
permanently (Simon, cyclic/Shor-style period-finding, Kuperberg dihedral
coset), and three discovery-target families with a tunable, planted degree
of coset structure (`structure_break_abelian`, `abelian_dihedral_blend`,
`generic_random_control`). Callers may pass `families=` (or mutate
`.families` on a registered domain) for a different roster; EXP-066 uses
the Phase-3 non-abelian pairing families. Do not change the default: EXP-064
and EXP-065 regenerate against `ALL_FAMILIES`.

Observe layer: every predictor-eligible column comes from
`sampling.bounded_query_descriptors`, itself built from a poly(n_bits)
random-query sample -- never from the full 2**n_bits table. See
`docs/research/hidden-subgroup-function-discovery-charter.md` S4.1 for why
that is the one non-negotiable part of this domain's contract (it is the
precondition every known exponential HSP-style separation depends on, and
the precise thing whose absence closed Direction E's TSP pivot).

Mode 1 campaigns on this domain used `structure_strength` / collision
rate / `algorithm_class` as Pearson targets (EXP-064–066). That cannot
emit an algorithm. Mode 2 recovery of planted K from a query tape is
`HspFunctionRecovery` (ALGO-063). `complexity.py` remains an interpretive
reference, not a per-row Delta target.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice
from rde_domains.hsp_functions import sampling
from rde_domains.hsp_functions.functions import (
    ALL_FAMILIES,
    FAMILIES_HELD_OUT,
    RECIPE_FAMILY,
    FunctionInstance,
    make_instance,
)


class HspFunctionDomain:
    """Domain over analytic hidden-subgroup-style oracle-function instances."""

    # Analytic oracles never materialize a 2^n table on the campaign path.
    bruteforce_enumeration = False

    def __init__(
        self,
        *,
        domain_id: str = "hsp_functions",
        families: Sequence[str] | None = None,
        recipe_catalog_size: int = 0,
        exam_fraction: float = 0.15,
    ) -> None:
        self.domain_id = domain_id
        self.families = tuple(families) if families is not None else ALL_FAMILIES
        self.recipe_catalog_size = int(recipe_catalog_size)
        self.exam_fraction = float(exam_fraction)

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        if self.recipe_catalog_size > 0:
            return self._generate_recipe_catalog(n, size, seed)
        roster = self.families
        if not roster:
            raise ValueError("hsp_functions domain has an empty family roster")
        instances: list[InstanceRecord] = []
        for i in range(n):
            inst_seed = seed + i
            family = roster[i % len(roster)]
            fi = make_instance(family, n_bits=size, seed=inst_seed)
            instances.append(self._record(fi, inst_seed, size, family, family))
        return instances

    def _generate_recipe_catalog(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        from rde_domains.hsp_functions.recipes import draw_recipe_ids, make_recipe_instance

        n_exam = max(len(FAMILIES_HELD_OUT), int(round(self.exam_fraction * n)))
        n_exam = min(n_exam, n - 1) if n > 1 else 0
        recipe_ids = draw_recipe_ids(n - n_exam, seed + 0xA062)
        instances: list[InstanceRecord] = []
        recipe_cursor = 0
        for i in range(n):
            inst_seed = seed + i
            if i < n_exam:
                family = FAMILIES_HELD_OUT[i % len(FAMILIES_HELD_OUT)]
                fi = make_instance(family, n_bits=size, seed=inst_seed)
                instances.append(self._record(fi, inst_seed, size, family, family))
                continue
            recipe_id = recipe_ids[recipe_cursor]
            recipe_cursor += 1
            fi = make_recipe_instance(size, inst_seed, recipe_id)
            generator = str(fi.params.get("generator") or f"{RECIPE_FAMILY}.unknown")
            instances.append(
                self._record(
                    fi,
                    inst_seed,
                    size,
                    RECIPE_FAMILY,
                    generator,
                    extra={
                        "recipe_id": int(fi.params["recipe_id"]),
                        "pairing": fi.params.get("pairing"),
                        "hidden_rank": fi.params.get("hidden_rank"),
                        "usefulness_tier": fi.params.get("usefulness_tier"),
                    },
                )
            )
        return instances

    def _record(
        self,
        fi: FunctionInstance,
        inst_seed: int,
        size: int,
        family: str,
        generator: str,
        extra: dict[str, Any] | None = None,
    ) -> InstanceRecord:
        params: dict[str, Any] = {
            "family": family,
            "generator": generator,
            "domain_kind": fi.domain_kind,
            "n_bits": size,
            "structure_strength": fi.structure_strength,
        }
        if extra:
            params.update(extra)
        from rde_domains.hsp_functions.kind_screen import algorithm_class_for_generator

        klass = algorithm_class_for_generator(generator)
        if np.isfinite(klass):
            params["algorithm_class"] = float(klass)
        return InstanceRecord(
            domain_id=self.domain_id,
            size=size,
            seed=inst_seed,
            params=params,
        )

    def _function_instance(self, instance: InstanceRecord) -> FunctionInstance:
        family = instance.params["family"]
        n_bits = instance.params["n_bits"]
        if family == RECIPE_FAMILY:
            from rde_domains.hsp_functions.recipes import make_recipe_instance

            return make_recipe_instance(n_bits, instance.seed, int(instance.params["recipe_id"]))
        return make_instance(family, n_bits=n_bits, seed=instance.seed)

    def _build_cache(self, instance: InstanceRecord, cache: dict[str, Any] | None) -> dict[str, Any]:
        if cache is not None and "fi" in cache:
            return cache
        fi = self._function_instance(instance)
        rng = np.random.default_rng((instance.seed << 1) ^ 0x5A5A5A5A)
        diff_estimates = sampling.sample_difference_estimates(fi, rng)
        ctx = {"fi": fi, "diff_estimates": diff_estimates}
        if cache is not None:
            cache.update(ctx)
        return ctx

    def prepare_instance(self, instance: InstanceRecord, *, indices: list[int] | None = None) -> dict[str, Any]:
        del indices
        cache: dict[str, Any] = {}
        self._build_cache(instance, cache)
        fi: FunctionInstance = cache["fi"]
        rng = np.random.default_rng(instance.seed ^ 0x1234_5678)
        cache["sample_desc"] = sampling.bounded_query_descriptors(
            fi, rng, name="f", diff_estimates=cache["diff_estimates"]
        )
        return cache

    def materialize(
        self, instance: InstanceRecord, index: int, *, cache: dict[str, Any] | None = None
    ) -> SimpleFamilySlice:
        if index != 0:
            raise ValueError(f"hsp_functions has a single family slice (index=0); got {index}")
        ctx = self._build_cache(instance, cache)
        values = np.array(list(ctx["diff_estimates"].values()), dtype=float)
        return SimpleFamilySlice(values=values, index=0, kind="difference_profile")

    def primitive_features(
        self, instance: InstanceRecord, *, cache: dict[str, Any] | None = None
    ) -> dict[str, float | np.ndarray]:
        ctx = self._build_cache(instance, cache)
        fi: FunctionInstance = ctx["fi"]
        if cache is not None and "sample_desc" in cache:
            sample_desc = cache["sample_desc"]
        else:
            rng = np.random.default_rng(instance.seed ^ 0x1234_5678)
            sample_desc = sampling.bounded_query_descriptors(
                fi, rng, name="f", diff_estimates=ctx["diff_estimates"]
            )

        out: dict[str, float | np.ndarray] = {
            "n_bits": float(fi.n_bits),
            #: Raw bounded-query primitive (poly(n_bits)-query difference
            #: profile g(d) at O(n_bits) *fixed* candidate shifts -- see
            #: `sampling.sample_difference_estimates`). Exposed as a named
            #: 1-D array so RDE's generic instance-descriptor sweep
            #: auto-derives `landscape.diff_profile.*` descriptors from it.
            #: Note: since candidates are the n_bits standard-basis shifts
            #: while a random secret s is drawn from the full ~2^n_bits
            #: nonzero domain, this profile is a near-miss for almost
            #: every instance (g(d)=1 only in the rare case s happens to
            #: equal a tested candidate exactly) -- kept as a real,
            #: leak-free, honestly-weak structural probe, not the primary
            #: signal (see `hsp_sample.f.*` below for that).
            "diff_profile": np.array(list(ctx["diff_estimates"].values()), dtype=float),
            #: OUTCOME ground truth (leak-excluded from predictors by the
            #: domain contract) -- re-exposed here so it reliably flows
            #: into the row regardless of instance.params flattening.
            "structure_strength": fi.structure_strength,
        }
        from rde_domains.hsp_functions.kind_screen import algorithm_class_for_generator

        klass = algorithm_class_for_generator(str(instance.params.get("generator") or fi.family))
        if np.isfinite(klass):
            out["algorithm_class"] = float(klass)

        # Re-expose the genuinely-varying collision-search statistics
        # (real per-instance stochastic counts, confirmed empirically to
        # carry real row-to-row variance -- unlike the candidate-based
        # diff_profile above) under a `rde.experiment.gate`-recognized
        # STRUCTURAL_PREFIXES key ("landscape."), since a custom prefix
        # alone (`hsp_sample.*`) is not recognized by that hardcoded
        # whitelist and left the population-distinctness gate unable to
        # see this domain's real signal at all -- confirmed empirically
        # via EXP-064's Gate-0 run. Same underlying bounded-query data as
        # `hsp_sample.f.*`, not a second measurement.
        for key in ("collision_rate", "n_collisions_found", "difference_span_dim_fraction", "detected_period_divisor_fraction"):
            full_key = f"hsp_sample.f.{key}"
            if full_key in sample_desc:
                out[f"landscape.{key}"] = sample_desc[full_key]

        out.update(sample_desc)
        if fi.family == RECIPE_FAMILY:
            out["hsp_recipe.recipe_id"] = float(fi.params.get("recipe_id", -1))
            out["hsp_recipe.hidden_rank"] = float(fi.params.get("hidden_rank") or 0)
            out["hsp_recipe.structure_break"] = float(fi.params.get("structure_break") or 0.0)
        if fi.n_bits <= sampling.MAX_ORACLE_N_BITS:
            out.update(sampling.exact_oracle_audit_descriptors(fi, name="f"))
        return out


def hsp_functions_domain(**kwargs: Any) -> HspFunctionDomain:
    return HspFunctionDomain(**kwargs)
