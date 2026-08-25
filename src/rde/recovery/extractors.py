"""Family-agnostic collision extractors (ALGO-063).

Each extractor sees only a ``QueryTape``. Returning ``None`` is abstain.
These are the classical post-processing moves of Simon / period-finding /
dihedral pairing, written without those names so a recovery matrix can
show which move actually recovers which planted secret.
"""

from __future__ import annotations

from collections import Counter
from math import gcd

from rde.core.protocols import QueryTape
from rde.recovery.tape import collision_groups


def _pair_values(tape: QueryTape, op: str) -> list[int]:
    values: list[int] = []
    mod = int(tape.modulus)
    for group in collision_groups(tape):
        base = group[0]
        for other in group[1:]:
            if op == "xor":
                v = base ^ other
            elif op == "sum":
                v = (base + other) % mod
            elif op == "diff":
                v = (other - base) % mod
            elif op == "ratio":
                # The multiplicative counterpart of "diff": how a hidden
                # subgroup of the *multiplicative* group Z_mod^* (rather
                # than the additive/XOR group) relates two colliding
                # points. Undefined when `base` shares a factor with
                # `mod` (no inverse) -- skip that pair rather than guess.
                try:
                    v = (other * pow(base, -1, mod)) % mod
                except ValueError:
                    continue
            else:
                raise ValueError(op)
            if v:
                values.append(int(v))
    return values


def _mode_or_none(values: list[int]) -> int | None:
    if not values:
        return None
    return int(Counter(values).most_common(1)[0][0])


def _mode_or_none_confident(values: list[int], min_ratio: float) -> int | None:
    """Mode of ``values``, abstaining unless the winner leads the runner-up by ``min_ratio``.

    A plain mode always answers, even off a weak plurality; this prefers an
    honest abstain (``None``) to a low-confidence guess.
    """
    if not values:
        return None
    counts = Counter(values).most_common(2)
    top_val, top_freq = counts[0]
    if len(counts) > 1:
        runner_freq = counts[1][1]
        if runner_freq > 0 and top_freq < min_ratio * runner_freq:
            return None
    return int(top_val)


class XorCollisionExtractor:
    """Most common nonzero XOR of colliding query points."""

    protocol_id = "xor_collision_mode"

    def extract(self, tape: QueryTape) -> int | None:
        return _mode_or_none(_pair_values(tape, "xor"))


class AdditiveSumExtractor:
    """Most common nonzero sum (mod |X|) of colliding query points."""

    protocol_id = "additive_sum_mode"

    def extract(self, tape: QueryTape) -> int | None:
        return _mode_or_none(_pair_values(tape, "sum"))


class AdditiveGcdExtractor:
    """GCD of nonzero modular differences of colliding query points."""

    protocol_id = "additive_gcd"

    def extract(self, tape: QueryTape) -> int | None:
        diffs = _pair_values(tape, "diff")
        if not diffs:
            return None
        g = 0
        for d in diffs:
            g = gcd(g, d)
        return int(g) if g else None


def default_extractor_catalog() -> tuple[XorCollisionExtractor, AdditiveSumExtractor, AdditiveGcdExtractor]:
    return (XorCollisionExtractor(), AdditiveSumExtractor(), AdditiveGcdExtractor())
