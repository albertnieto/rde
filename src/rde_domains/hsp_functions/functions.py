"""Analytic, query-evaluable oracle-function families.

Every family below defines f: X -> label via a closed-form construction
evaluable at a single point x in O(1) time with no table materialization,
so the domain can be queried a bounded number of times regardless of how
large X is -- the query/oracle access model this domain's contract requires
non-negotiably. This also means N can be pushed far larger than this
project's usual brute-force-enumeration cap (`max_bruteforce_n`), since
nothing here ever needs to materialize a length-2^n table except the
small, explicitly-gated oracle-only calibration audit.

Each family carries a `structure_strength` in [0, 1], fixed at generation
time: 1.0 = an exact, uncorrupted hidden-coset structure; 0.0 = no
exploitable structure at all. This is the leak-excluded OUTCOME ground
truth the RDE domain's primary target regresses against -- the same
proven pattern already used successfully in this repo by
`tsp_circulant_symmetry` (`symmetry_break_param` -> `circulant_deviation`),
generalized here from geometric symmetry-breaking to algebraic
coset-structure-breaking. It must never be exposed to predictor-eligible
descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FAMILIES_HELD_OUT = ("simon", "shor_cyclic", "dihedral_kuperberg")
FAMILIES_DISCOVERY = ("structure_break_abelian", "abelian_dihedral_blend", "generic_random_control")
ALL_FAMILIES = FAMILIES_HELD_OUT + FAMILIES_DISCOVERY
# An alternative discovery roster: non-abelian pairings that are not
# dihedral, plus the same structureless control. Default ALL_FAMILIES stays
# unchanged so existing tests stay on the original six families.
FAMILIES_PHASE3_DISCOVERY = ("heisenberg_noncentral", "quaternion_coset", "generic_random_control")
PHASE3_POPULATION = FAMILIES_HELD_OUT + FAMILIES_PHASE3_DISCOVERY
# Path-B encoding control: same Heisenberg cosets but v XORs into the low
# register b instead of high register a. Not a discovery family.
PATH_B_HEISENBERG_FAMILY = "heisenberg_v_low_register"
KNOWN_FAMILIES = tuple(
    dict.fromkeys(
        ALL_FAMILIES + FAMILIES_PHASE3_DISCOVERY + ("hsp_recipe", PATH_B_HEISENBERG_FAMILY)
    )
)
RECIPE_FAMILY = "hsp_recipe"
N_RECIPES_DEFAULT = 10_000

_MASK64 = (1 << 64) - 1


def _label_hash(seed: int, coset_repr: int, salt: int) -> int:
    """Deterministic pseudorandom 63-bit label, injective on cosets whp.

    A fixed-seed splitmix64-style mix -- fast, dependency-free, and
    reproducible across a Python process without relying on hash()
    randomization (PYTHONHASHSEED). Collisions across distinct coset
    representatives are astronomically unlikely (63-bit range) and are not
    the mechanism this domain is testing.
    """
    z = (seed * 0x9E3779B97F4A7C15 + coset_repr * 0xBF58476D1CE4E5B9 + salt * 0x94D049BB133111EB) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z ^= z >> 31
    return z & ((1 << 63) - 1)


def _uniform_unit(seed: int, x: int, salt: int) -> float:
    return _label_hash(seed, x, salt) / float((1 << 63) - 1)


def _bit_noise(seed: int, x: int, salt: int, tau: float) -> bool:
    """Deterministic pseudorandom Bernoulli(tau) draw, keyed on (seed, x, salt)."""
    if tau <= 0.0:
        return False
    return _uniform_unit(seed, x, salt) < tau


# --- Vectorized (NumPy) counterparts of the scalar hash primitives above ---
#
# This workload (a fixed-depth integer hash mix over a batch of query
# points) is elementwise scalar-integer arithmetic, not dense tensor/linear
# algebra -- MLX offers nothing over plain NumPy here, so NumPy is the
# right vectorization primitive (see `docs/engineering/agent-correction-
# playbook.md`'s backend-contract rules: CPU/NumPy is always a legitimate,
# not merely fallback, choice when a workload isn't GPU-shaped). This
# replaces the Python-level per-query loop the bounded-query descriptor
# pipeline would otherwise need -- the actual optimization-first hot path
# for this domain, since a real campaign evaluates the oracle
# `O(population x sizes x n_bits^2)` times.
#
# `_MASK64` arithmetic (Python bigint-then-mask) and NumPy `uint64`
# arithmetic (silent wraparound mod 2**64) are algebraically identical --
# modular arithmetic is compatible with +/xor/* at every intermediate step,
# not just at the end -- verified by `test_functions_batch_matches_scalar`
# in `tests/rde/test_hsp_functions_domain.py`.

_U64 = np.uint64


def _label_hash_batch(seed: int, coset_repr: np.ndarray, salt: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        seed_u, salt_u = _U64(seed), _U64(salt)
        coset_u = coset_repr.astype(np.uint64)
        z = seed_u * _U64(0x9E3779B97F4A7C15) + coset_u * _U64(0xBF58476D1CE4E5B9) + salt_u * _U64(0x94D049BB133111EB)
        z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
        z ^= z >> _U64(31)
        return z & _U64((1 << 63) - 1)


def _uniform_unit_batch(seed: int, x: np.ndarray, salt: int) -> np.ndarray:
    return _label_hash_batch(seed, x, salt).astype(np.float64) / float((1 << 63) - 1)


def _bit_noise_batch(seed: int, x: np.ndarray, salt: int, tau: float) -> np.ndarray:
    if tau <= 0.0:
        return np.zeros(x.shape, dtype=bool)
    return _uniform_unit_batch(seed, x, salt) < tau


# Quaternion group Q8 encoded in 3 bits:
# 0=1, 1=i, 2=j, 3=k, 4=-1, 5=-i, 6=-j, 7=-k.
# Hidden K = {1, i, -1, -i} = {0, 1, 4, 5}; the other right coset is jK.
_Q8_BODY = np.array(
    [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ],
    dtype=np.int64,
)
_Q8_EXTRA_SIGN = np.array(
    [
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ],
    dtype=np.int64,
)
_Q8_MUL = np.array(
    [
        [
            ((_Q8_EXTRA_SIGN[a & 3, b & 3] ^ (a >> 2) ^ (b >> 2)) << 2)
            | int(_Q8_BODY[a & 3, b & 3])
            for b in range(8)
        ]
        for a in range(8)
    ],
    dtype=np.int64,
)
_Q8_COSET_REP = np.array([0 if q in (0, 1, 4, 5) else 2 for q in range(8)], dtype=np.int64)
_Q8_HIDDEN = (0, 1, 4, 5)


def _heisenberg_parts(x: int, n_bits: int) -> tuple[int, int, int, int, int]:
    m = n_bits // 2
    mask = (1 << m) - 1
    extra = x >> (2 * m)
    a = (x >> m) & mask
    b = x & mask
    return a, b, extra, m, mask


def _heisenberg_phi(a: int, v: int, a_bit: int = 0, v_bit: int = 1) -> int:
    """Non-symmetric bilinear form: φ(a,v) = a[a_bit] * v[v_bit], into b's LSB."""
    return ((a >> a_bit) & 1) * ((v >> v_bit) & 1)


def heisenberg_right_mul_k(
    x: int, n_bits: int, v: int, *, a_bit: int = 0, v_bit: int = 1
) -> int:
    """Right-multiply x by the order-2 element (v, 0) in the Heisenberg group."""
    a, b, extra, m, _mask = _heisenberg_parts(x, n_bits)
    a2 = a ^ v
    b2 = b ^ _heisenberg_phi(a, v, a_bit, v_bit)
    return (extra << (2 * m)) | (a2 << m) | b2


def _heisenberg_coset_repr(
    x: int | np.ndarray,
    n_bits: int,
    v: int,
    w: int,
    *,
    a_bit: int = 0,
    v_bit: int = 1,
) -> int | np.ndarray:
    """Canonical right-coset representative: force a·w = 0 by multiplying by k."""
    m = n_bits // 2
    mask = (1 << m) - 1
    extra_shift = 2 * m
    if isinstance(x, np.ndarray):
        xs = x.astype(np.int64, copy=False)
        extra = xs >> extra_shift
        a = (xs >> m) & mask
        b = xs & mask
        dot = (a & w) != 0
        phi = ((a >> a_bit) & 1) * ((v >> v_bit) & 1)
        a2 = np.where(dot, a ^ v, a)
        b2 = np.where(dot, b ^ phi, b)
        return (extra << extra_shift) | (a2 << m) | b2
    a, b, extra, m, _mask = _heisenberg_parts(int(x), n_bits)
    if (a & w) != 0:
        b ^= _heisenberg_phi(a, v, a_bit, v_bit)
        a ^= v
    return (extra << (2 * m)) | (a << m) | b


def _heisenberg_coset_repr_v_low(
    x: int | np.ndarray,
    n_bits: int,
    v: int,
    w: int,
) -> int | np.ndarray:
    """Path-B control: hidden v XORs into b (low half), not a (high half).

    Coset members differ by b ^= v with a unchanged, so collision XOR places
    v in the low m bits. The frozen ``xor_mode_high_half`` post should fail.
    """
    m = n_bits // 2
    mask = (1 << m) - 1
    extra_shift = 2 * m
    if isinstance(x, np.ndarray):
        xs = x.astype(np.int64, copy=False)
        extra = xs >> extra_shift
        a = (xs >> m) & mask
        b = xs & mask
        dot = (b & w) != 0
        b2 = np.where(dot, b ^ v, b)
        return (extra << extra_shift) | (a << m) | b2
    a, b, extra, m, _mask = _heisenberg_parts(int(x), n_bits)
    if (b & w) != 0:
        b ^= v
    return (extra << (2 * m)) | (a << m) | b


def quaternion_right_mul(x: int, k_q: int) -> int:
    """Right-multiply the Q8 register (low 3 bits) by k_q in {0,...,7}."""
    q = x & 7
    rest = x >> 3
    return (rest << 3) | int(_Q8_MUL[q, k_q & 7])


def _quaternion_coset_repr(x: int | np.ndarray) -> int | np.ndarray:
    if isinstance(x, np.ndarray):
        xs = x.astype(np.int64, copy=False)
        q = xs & 7
        rest = xs >> 3
        return (rest << 3) | _Q8_COSET_REP[q]
    q = int(x) & 7
    rest = int(x) >> 3
    return (rest << 3) | int(_Q8_COSET_REP[q])


@dataclass(frozen=True)
class FunctionInstance:
    """A single analytic oracle-function instance."""

    family: str
    domain_kind: str  # "gf2" or "cyclic"
    n_bits: int  # log2|X|
    x_size: int  # |X| = 2**n_bits
    seed: int
    params: dict[str, Any]  # family hyperparameters, incl. the true structure_strength

    def evaluate(self, x: int) -> int:
        return _evaluate(self, x)

    def evaluate_batch(self, xs: np.ndarray) -> np.ndarray:
        """Vectorized `evaluate` over an array of query points. See ALGO-062."""
        return _evaluate_batch(self, np.asarray(xs, dtype=np.int64) % self.x_size)

    @property
    def structure_strength(self) -> float:
        return float(self.params["structure_strength"])


def _evaluate(inst: FunctionInstance, x: int) -> int:
    x = x % inst.x_size
    fam = inst.family
    p = inst.params
    if fam == "simon":
        s = p["s"]
        coset = min(x, x ^ s)
        return _label_hash(inst.seed, coset, salt=1)
    if fam == "shor_cyclic":
        r = p["r"]
        coset = x % r
        return _label_hash(inst.seed, coset, salt=2)
    if fam == "dihedral_kuperberg":
        s = p["s"]
        coset = min(x, (s - x) % inst.x_size)
        return _label_hash(inst.seed, coset, salt=3)
    if fam == "structure_break_abelian":
        s = p["s"]
        break_frac = p["structure_break"]
        if _bit_noise(inst.seed, x, salt=11, tau=break_frac):
            # This point ignores the coset rule: an independent label, as
            # injective (collision-free) as the labeling hash itself.
            return _label_hash(inst.seed, x, salt=12)
        coset = min(x, x ^ s)
        return _label_hash(inst.seed, coset, salt=13)
    if fam == "abelian_dihedral_blend":
        s_ab = p["s_abelian"]
        s_di = p["s_dihedral"]
        w = p["blend_weight"]  # probability a point is routed through the abelian rule
        if _uniform_unit(inst.seed, x, salt=21) < w:
            coset = min(x, x ^ s_ab)
            return _label_hash(inst.seed, coset, salt=22)
        coset = min(x, (s_di - x) % inst.x_size)
        return _label_hash(inst.seed, coset, salt=23)
    if fam == "generic_random_control":
        return _label_hash(inst.seed, x, salt=31)
    if fam == "heisenberg_noncentral":
        break_frac = p["structure_break"]
        if _bit_noise(inst.seed, x, salt=41, tau=break_frac):
            return _label_hash(inst.seed, x, salt=42)
        coset = _heisenberg_coset_repr(
            x, inst.n_bits, p["v"], p["w"], a_bit=int(p.get("phi_a_bit", 0)), v_bit=int(p.get("phi_v_bit", 1))
        )
        return _label_hash(inst.seed, int(coset), salt=43)
    if fam == PATH_B_HEISENBERG_FAMILY:
        break_frac = p["structure_break"]
        if _bit_noise(inst.seed, x, salt=61, tau=break_frac):
            return _label_hash(inst.seed, x, salt=62)
        coset = _heisenberg_coset_repr_v_low(x, inst.n_bits, p["v"], p["w"])
        return _label_hash(inst.seed, int(coset), salt=63)
    if fam == "quaternion_coset":
        break_frac = p["structure_break"]
        if _bit_noise(inst.seed, x, salt=51, tau=break_frac):
            return _label_hash(inst.seed, x, salt=52)
        coset = _quaternion_coset_repr(x)
        return _label_hash(inst.seed, int(coset), salt=53)
    if fam == RECIPE_FAMILY:
        from rde_domains.hsp_functions.recipes import evaluate_recipe

        return evaluate_recipe(inst, x)
    raise ValueError(f"unknown hsp_functions family: {fam!r}")


def _evaluate_batch(inst: FunctionInstance, xs: np.ndarray) -> np.ndarray:
    fam = inst.family
    p = inst.params
    if fam == "simon":
        s = p["s"]
        coset = np.minimum(xs, xs ^ s)
        return _label_hash_batch(inst.seed, coset, salt=1)
    if fam == "shor_cyclic":
        r = p["r"]
        coset = xs % r
        return _label_hash_batch(inst.seed, coset, salt=2)
    if fam == "dihedral_kuperberg":
        s = p["s"]
        coset = np.minimum(xs, (s - xs) % inst.x_size)
        return _label_hash_batch(inst.seed, coset, salt=3)
    if fam == "structure_break_abelian":
        s = p["s"]
        break_frac = p["structure_break"]
        broken = _bit_noise_batch(inst.seed, xs, salt=11, tau=break_frac)
        coset = np.minimum(xs, xs ^ s)
        structured = _label_hash_batch(inst.seed, coset, salt=13)
        independent = _label_hash_batch(inst.seed, xs, salt=12)
        return np.where(broken, independent, structured)
    if fam == "abelian_dihedral_blend":
        s_ab, s_di, w = p["s_abelian"], p["s_dihedral"], p["blend_weight"]
        via_abelian = _uniform_unit_batch(inst.seed, xs, salt=21) < w
        coset_ab = np.minimum(xs, xs ^ s_ab)
        coset_di = np.minimum(xs, (s_di - xs) % inst.x_size)
        return np.where(via_abelian, _label_hash_batch(inst.seed, coset_ab, salt=22), _label_hash_batch(inst.seed, coset_di, salt=23))
    if fam == "generic_random_control":
        return _label_hash_batch(inst.seed, xs, salt=31)
    if fam == "heisenberg_noncentral":
        break_frac = p["structure_break"]
        broken = _bit_noise_batch(inst.seed, xs, salt=41, tau=break_frac)
        coset = np.asarray(
            _heisenberg_coset_repr(
                xs,
                inst.n_bits,
                p["v"],
                p["w"],
                a_bit=int(p.get("phi_a_bit", 0)),
                v_bit=int(p.get("phi_v_bit", 1)),
            ),
            dtype=np.int64,
        )
        structured = _label_hash_batch(inst.seed, coset, salt=43)
        independent = _label_hash_batch(inst.seed, xs, salt=42)
        return np.where(broken, independent, structured)
    if fam == PATH_B_HEISENBERG_FAMILY:
        break_frac = p["structure_break"]
        broken = _bit_noise_batch(inst.seed, xs, salt=61, tau=break_frac)
        coset = np.asarray(
            _heisenberg_coset_repr_v_low(xs, inst.n_bits, p["v"], p["w"]),
            dtype=np.int64,
        )
        structured = _label_hash_batch(inst.seed, coset, salt=63)
        independent = _label_hash_batch(inst.seed, xs, salt=62)
        return np.where(broken, independent, structured)
    if fam == "quaternion_coset":
        break_frac = p["structure_break"]
        broken = _bit_noise_batch(inst.seed, xs, salt=51, tau=break_frac)
        coset = np.asarray(_quaternion_coset_repr(xs), dtype=np.int64)
        structured = _label_hash_batch(inst.seed, coset, salt=53)
        independent = _label_hash_batch(inst.seed, xs, salt=52)
        return np.where(broken, independent, structured)
    if fam == RECIPE_FAMILY:
        from rde_domains.hsp_functions.recipes import evaluate_recipe_batch

        return evaluate_recipe_batch(inst, xs)
    raise ValueError(f"unknown hsp_functions family: {fam!r}")


def make_instance(family: str, n_bits: int, seed: int) -> FunctionInstance:
    """Draw one random instance of `family` at size `n_bits`, seeded deterministically."""
    if family not in KNOWN_FAMILIES:
        raise ValueError(f"unknown hsp_functions family: {family!r}")
    x_size = 1 << n_bits

    def _rand_bits(salt: int) -> int:
        # A nonzero pseudorandom element of {0,...,x_size-1}, deterministic in seed.
        v = _label_hash(seed, 0, salt) % (x_size - 1) + 1
        return v

    if family == "simon":
        s = _rand_bits(101)
        params: dict[str, Any] = {"s": s, "structure_strength": 1.0}
        return FunctionInstance(family, "gf2", n_bits, x_size, seed, params)
    if family == "shor_cyclic":
        r = max(2, x_size // 2)  # |K_true| = x_size // r = 2, matching simon's |K|=2
        params = {"r": r, "structure_strength": 1.0}
        return FunctionInstance(family, "cyclic", n_bits, x_size, seed, params)
    if family == "dihedral_kuperberg":
        s = _rand_bits(103)
        params = {"s": s, "structure_strength": 1.0}
        return FunctionInstance(family, "cyclic", n_bits, x_size, seed, params)
    if family == "structure_break_abelian":
        s = _rand_bits(111)
        # Uniform draw over [0, max_break]; max_break kept < 1 so some
        # structure always remains detectable in principle.
        max_break = 0.85
        structure_break = _uniform_unit(seed, 0, 201) * max_break
        params = {"s": s, "structure_break": structure_break, "structure_strength": 1.0 - structure_break}
        return FunctionInstance(family, "gf2", n_bits, x_size, seed, params)
    if family == "abelian_dihedral_blend":
        s_ab = _rand_bits(121)
        s_di = _rand_bits(123)
        blend_weight = _uniform_unit(seed, 0, 202)
        coherence = 2.0 * max(blend_weight, 1.0 - blend_weight) - 1.0  # 0 at w=0.5, 1 at w in {0,1}
        params = {
            "s_abelian": s_ab,
            "s_dihedral": s_di,
            "blend_weight": blend_weight,
            "structure_strength": coherence,
        }
        return FunctionInstance(family, "gf2", n_bits, x_size, seed, params)
    if family == "generic_random_control":
        params = {"structure_strength": 0.0}
        return FunctionInstance(family, "gf2", n_bits, x_size, seed, params)
    if family in {"heisenberg_noncentral", PATH_B_HEISENBERG_FAMILY}:
        if n_bits < 4:
            raise ValueError(f"{family} requires n_bits >= 4")
        m = n_bits // 2
        vm = 1 << m
        raw = _label_hash(seed, 0, 301) % vm
        # bit0=0 and bit1=1 ⇒ (v,0) has order 2 and is non-central.
        v = (raw & ~3) | 2
        w = 2
        max_break = 0.85
        structure_break = _uniform_unit(seed, 0, 401) * max_break
        params = {
            "v": int(v),
            "w": int(w),
            "phi_a_bit": 0,
            "phi_v_bit": 1,
            "structure_break": structure_break,
            "structure_strength": 1.0 - structure_break,
        }
        return FunctionInstance(family, "gf2", n_bits, x_size, seed, params)
    if family == "quaternion_coset":
        if n_bits < 8:
            raise ValueError("quaternion_coset requires n_bits >= 8 (Q8 factor plus abelian bits)")
        max_break = 0.85
        structure_break = _uniform_unit(seed, 0, 501) * max_break
        params = {
            "structure_break": structure_break,
            "structure_strength": 1.0 - structure_break,
        }
        return FunctionInstance(family, "gf2", n_bits, x_size, seed, params)
    if family == RECIPE_FAMILY:
        raise ValueError("hsp_recipe requires recipes.make_recipe_instance(n_bits, seed, recipe_id)")
    raise AssertionError("unreachable")
