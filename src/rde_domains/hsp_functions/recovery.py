"""HSP RecoveryDomain: score query-tape extractors against planted secrets.

Scoring-only. Extractors in ``rde.recovery.extractors`` never import this
module and never receive ``planted``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.core.protocols import QueryTape
from rde_domains.hsp_functions.functions import FunctionInstance
from rde_domains.hsp_functions.sampling import query_budget_for


@dataclass(frozen=True)
class AbelianDihedralSecret:
    """The two independent hidden shifts `abelian_dihedral_blend` mixes per query.

    Distinct type from a plain tuple so `HspFunctionRecovery.match` can tell
    this apart from `quaternion_coset`'s unordered coset tuple -- this one is
    ordered and two-part by construction, not a set.
    """

    s_abelian: int
    s_dihedral: int


class HspFunctionRecovery:
    """RecoveryDomain over analytic ``FunctionInstance`` objects."""

    domain_id = "hsp_functions"

    def size_of(self, instance: FunctionInstance) -> int:
        return int(instance.n_bits)

    def family_of(self, instance: FunctionInstance) -> str:
        return str(instance.family)

    def draw_tape(self, instance: FunctionInstance, rng: np.random.Generator) -> QueryTape:
        budget = query_budget_for(instance.n_bits)
        xs = rng.integers(0, instance.x_size, size=budget, dtype=np.int64)
        ys = instance.evaluate_batch(xs)
        return QueryTape(
            xs=xs,
            ys=ys,
            budget=budget,
            modulus=instance.x_size,
            n_bits=instance.n_bits,
        )

    def planted(self, instance: FunctionInstance) -> Any:
        family = instance.family
        params = instance.params
        if family == "simon":
            return int(params["s"])
        if family in {"shor_cyclic", "approximate_cyclic_period", "approximate_cyclic_period_alt"}:
            return int(params["r"])
        if family == "dihedral_kuperberg":
            return int(params["s"])
        if family in {"structure_break_abelian", "approximate_xor_shift"}:
            return int(params["s"])
        if family in {"heisenberg_noncentral", "heisenberg_v_low_register"}:
            return int(params["v"])
        if family == "quaternion_coset":
            from rde_domains.hsp_functions.functions import _Q8_HIDDEN

            return tuple(sorted(_Q8_HIDDEN))
        if family == "abelian_dihedral_blend":
            return AbelianDihedralSecret(
                s_abelian=int(params["s_abelian"]), s_dihedral=int(params["s_dihedral"])
            )
        if family == "multiplicative_fold":
            s, s2, s3 = int(params["s"]), int(params["s2"]), int(params["s3"])
            return tuple(sorted({1, s, s2, s3}))
        if family == "generic_random_control":
            return None
        if family == "hsp_recipe":
            pairing = params.get("pairing")
            if pairing == "xor":
                gens = params.get("xor_generators") or params.get("generators")
                if gens is None:
                    return None
                if isinstance(gens, (int, np.integer)):
                    return int(gens)
                values = [int(g) for g in gens if int(g)]
                return int(values[0]) if len(values) == 1 else tuple(sorted(values))
            if pairing == "cyclic":
                return int(params.get("period") or params.get("r") or 0) or None
            return None
        return None

    def match(self, recovered: Any, planted: Any) -> bool:
        if planted is None:
            return recovered is None
        if recovered is None:
            return False
        if isinstance(planted, AbelianDihedralSecret):
            if not isinstance(recovered, tuple) or len(recovered) != 2:
                return False
            a, b = recovered
            # A generic search can pair any two protocols, including ones
            # that themselves return tuples (e.g. GroupClosureProgram) --
            # such a pairing is simply not a match here, not a crash.
            if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)):
                return False
            a, b = int(a), int(b)
            # PairCombine's argument order is whichever position the two
            # atoms happened to land at in the flat depth-1 catalog, not
            # which secret each one actually recovers -- an order-sensitive
            # match would silently score a genuinely correct two-mechanism
            # recovery as a miss whenever the s_dihedral-recovering atom's
            # catalog index happens to precede the s_abelian one's.
            target = (planted.s_abelian, planted.s_dihedral)
            return (a, b) == target or (b, a) == target
        if isinstance(planted, tuple) and not isinstance(recovered, tuple):
            return False
        if isinstance(recovered, tuple) and not isinstance(planted, tuple):
            return len(recovered) == 1 and int(recovered[0]) == int(planted)
        if isinstance(planted, tuple):
            return tuple(sorted(recovered)) == tuple(sorted(planted))
        return int(recovered) == int(planted)
