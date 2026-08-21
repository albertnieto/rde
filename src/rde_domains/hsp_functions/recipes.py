"""ALGO-062 recipe catalog: many HSP-style oracles, biased toward useful structure.

This is not a list of 10k named families and not a draw from random truth
tables. It is a mixed-radix catalog of ``N_RECIPES_DEFAULT`` (10_000)
analytic hidden-pairing recipes:

- 80% structured HSP-style (XOR subspace, cyclic period, dihedral
  reflection, Heisenberg, quaternion) -- constructions with a
  literature or plausible query-gap (CLAIM-188--193). Most of these keep
  the exact-promise pairing; a small tail adds mild label noise.
- 10% blends of two structured laws
- 10% structureless control

Default ``ALL_FAMILIES`` is unchanged. Discovery campaigns opt in via
``HspFunctionDomain(recipe_catalog_size=10000)``. Callers must
**stratify** with :func:`draw_recipe_ids` -- walking consecutive ids from
0 never reaches the blend/control bands.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rde_domains.hsp_functions.functions import (
    N_RECIPES_DEFAULT,
    RECIPE_FAMILY,
    FunctionInstance,
    _bit_noise,
    _bit_noise_batch,
    _heisenberg_coset_repr,
    _label_hash,
    _label_hash_batch,
    _quaternion_coset_repr,
    _uniform_unit,
    _uniform_unit_batch,
)

PAIRINGS_STRUCTURED = ("xor", "cyclic", "dihedral", "heisenberg", "quaternion")
N_STRUCTURED = 8_000
N_BLEND = 1_000
N_RANDOM = 1_000
assert N_STRUCTURED + N_BLEND + N_RANDOM == N_RECIPES_DEFAULT

# Of 20 break codes, 16 are exact promise, 3 are mild noise, 1 is heavy.
N_BREAK_CODES = 20
N_EXACT_PROMISE_CODES = 16
MIX_STRUCTURED = 0.80
MIX_BLEND = 0.10
MIX_RANDOM = 0.10


def structure_break_from_code(break_code: int) -> float:
    """Map a 0..19 code onto a usefulness-biased noise level."""
    code = int(break_code) % N_BREAK_CODES
    if code < N_EXACT_PROMISE_CODES:
        return 0.0
    if code < N_EXACT_PROMISE_CODES + 3:
        return 0.05 * (code - N_EXACT_PROMISE_CODES + 1)
    return 0.40


def cyclic_period(n_bits: int, rank: int, period_class: int) -> int:
    """Period families: dyadic, odd, 3-smooth, near-half odd -- not one 2^{n-r}."""
    n_bits = int(n_bits)
    rank = max(1, min(int(rank), max(1, n_bits - 1)))
    dyadic = max(2, (1 << n_bits) >> rank)
    klass = int(period_class) % 4
    if klass == 0:
        return int(dyadic)
    if klass == 1:
        odd = dyadic - 1 if dyadic > 2 else 3
        return int(odd if odd % 2 else odd - 1)
    if klass == 2:
        return int(max(3, 3 * max(1, dyadic // 4)))
    return int(max(3, (dyadic // 2) * 2 + 1))


def decode_recipe(recipe_id: int) -> dict[str, Any]:
    """Deterministic catalog entry. ``recipe_id`` in ``[0, N_RECIPES_DEFAULT)``."""
    rid = int(recipe_id) % N_RECIPES_DEFAULT
    if rid >= N_STRUCTURED + N_BLEND:
        return {
            "recipe_id": rid,
            "pairing": "random",
            "hidden_rank": 0,
            "structure_break": 1.0,
            "usefulness_tier": "control",
            "generator": f"{RECIPE_FAMILY}.random",
        }
    if rid >= N_STRUCTURED:
        mix = rid - N_STRUCTURED
        a = PAIRINGS_STRUCTURED[mix % 5]
        b = PAIRINGS_STRUCTURED[(mix // 5) % 5]
        if a == b:
            b = PAIRINGS_STRUCTURED[(mix // 5 + 1) % 5]
        weight = ((mix // 25) % 20) / 19.0
        rank = 1 + (mix // 500) % 3
        return {
            "recipe_id": rid,
            "pairing": "blend",
            "blend_a": a,
            "blend_b": b,
            "blend_weight": float(weight),
            "hidden_rank": int(rank),
            "structure_break": 0.0,
            "period_class": int((mix // 100) % 4),
            "fold_mode": int((mix // 50) % 2),
            "xor_weight_class": int((mix // 10) % 3),
            "usefulness_tier": "blend",
            "generator": f"{RECIPE_FAMILY}.blend",
        }
    pairing = PAIRINGS_STRUCTURED[rid % 5]
    hidden_rank = 1 + ((rid // 5) % 3)
    break_code = (rid // 15) % N_BREAK_CODES
    form_id = rid // 300
    return {
        "recipe_id": rid,
        "pairing": pairing,
        "hidden_rank": int(hidden_rank),
        "structure_break": float(structure_break_from_code(break_code)),
        "form_id": int(form_id),
        "period_class": int(form_id % 4),
        "fold_mode": int(form_id % 2),
        "xor_weight_class": int(form_id % 3),
        "usefulness_tier": "hsp_gap",
        "generator": f"{RECIPE_FAMILY}.{pairing}",
    }


def catalog_slot_counts(n_recipe: int) -> tuple[int, int, int]:
    """How many structured / blend / random draws for ``n_recipe`` instances.

    Matches the 80/10/10 catalog mix. Small ``n`` still gets at least one
    blend and one control once there are three or more recipe rows.
    """
    n_recipe = int(n_recipe)
    if n_recipe <= 0:
        return 0, 0, 0
    if n_recipe == 1:
        return 1, 0, 0
    if n_recipe == 2:
        return 1, 0, 1
    n_random = max(1, int(round(MIX_RANDOM * n_recipe)))
    n_blend = max(1, int(round(MIX_BLEND * n_recipe)))
    if n_random + n_blend >= n_recipe:
        n_random = max(1, n_recipe // 10)
        n_blend = max(1, n_recipe // 10)
    n_struct = n_recipe - n_blend - n_random
    if n_struct < 1:
        n_struct = 1
        n_random = max(0, n_recipe - n_struct - n_blend)
    return int(n_struct), int(n_blend), int(n_random)


def draw_recipe_ids(n_recipe: int, seed: int, *, catalog: int = N_RECIPES_DEFAULT) -> list[int]:
    """Stratified sample from the 10k catalog (not consecutive ids from 0).

    Consecutive ``(seed + i) % catalog`` never reaches ids 8000--9999 unless
    the walk is thousands of steps. Every campaign size must draw from all
    three bands.
    """
    del catalog  # catalog cardinality is the 10k decode space, not a truncate
    n_struct, n_blend, n_random = catalog_slot_counts(n_recipe)
    rng = np.random.default_rng(int(seed) ^ 0xA062_10_00)
    struct_ids = _choice_ids(rng, 0, N_STRUCTURED, n_struct)
    blend_ids = _choice_ids(rng, N_STRUCTURED, N_STRUCTURED + N_BLEND, n_blend)
    random_ids = _choice_ids(rng, N_STRUCTURED + N_BLEND, N_RECIPES_DEFAULT, n_random)
    ids = struct_ids + blend_ids + random_ids
    rng.shuffle(ids)
    return [int(x) for x in ids]


def _choice_ids(rng: np.random.Generator, lo: int, hi: int, n: int) -> list[int]:
    if n <= 0:
        return []
    span = int(hi) - int(lo)
    if span <= 0:
        return []
    replace = n > span
    drawn = rng.choice(span, size=n, replace=replace)
    return [int(lo) + int(v) for v in drawn]


def _independent_gens(
    n_bits: int, rank: int, seed: int, salt: int, *, weight_class: int = 1
) -> tuple[int, ...]:
    """``rank`` linearly independent nonzero elements of GF(2)^n.

    ``weight_class`` 0 = sparse (low Hamming weight), 1 = mixed, 2 = dense.
    """
    n_bits = int(n_bits)
    rank = max(1, min(int(rank), n_bits))
    gens: list[int] = []
    used = 0
    t = 0
    x_size = 1 << n_bits
    klass = int(weight_class) % 3
    while len(gens) < rank and t < 512:
        v = _label_hash(seed, t, salt) % (x_size - 1) + 1
        if klass == 0:
            # Prefer weight-1/2/3 secrets (Simon-like sparse hidden strings).
            while v.bit_count() > 3 and t < 512:
                t += 1
                v = _label_hash(seed, t, salt) % (x_size - 1) + 1
        elif klass == 2:
            mask = _label_hash(seed, t, salt + 17) % x_size
            v = v | mask
            if v == 0:
                v = (1 << (t % n_bits))
        pivot = v & -v
        if pivot == 0 or (used & pivot):
            bit = len(gens) % n_bits
            if used & (1 << bit):
                bit = next((i for i in range(n_bits) if not (used & (1 << i))), 0)
            v = 1 << bit
            pivot = v
        gens.append(int(v))
        used |= pivot
        t += 1
    return _rref_gens(tuple(gens))


def _rref_gens(gens: tuple[int, ...]) -> tuple[int, ...]:
    """Reduced row-echelon GF(2) basis so a single pivot-clearing pass is exact."""
    reduced: list[int] = []
    pivots: list[int] = []
    for raw in gens:
        g = int(raw)
        for pivot, row in zip(pivots, reduced):
            if g & pivot:
                g ^= row
        if g == 0:
            continue
        pivot = g & -g
        for i, row in enumerate(reduced):
            if row & pivot:
                reduced[i] ^= g
        reduced.append(g)
        pivots.append(pivot)
    return tuple(int(g) for g in reduced)


def xor_subspace_rep(x: int | np.ndarray, gens: tuple[int, ...]) -> int | np.ndarray:
    """Canonical coset rep: clear each generator's pivot bit by XOR.

    One pass is enough when ``gens`` is RREF (``_rref_gens``). Extra passes
    catch a non-reduced basis so a later generator cannot re-set an earlier
    pivot (the rank-3 failure mode).
    """
    gens = tuple(int(g) for g in gens if int(g) != 0)
    rounds = max(1, len(gens) + 1)
    if isinstance(x, np.ndarray):
        out = x.astype(np.int64, copy=True)
        for _ in range(rounds):
            for g in gens:
                pivot = g & -g
                bit_set = (out & pivot) != 0
                out = np.where(bit_set, out ^ g, out)
        return out
    out = int(x)
    for _ in range(rounds):
        for g in gens:
            pivot = g & -g
            if out & pivot:
                out ^= g
    return out


def _heisenberg_form(n_bits: int, form_id: int, seed: int) -> tuple[int, int, int, int]:
    m = max(2, n_bits // 2)
    vm = 1 << m
    a_bit = int(form_id) % m
    v_bit = (int(form_id) // m + 1) % m
    if v_bit == a_bit:
        v_bit = (a_bit + 1) % m
    raw = _label_hash(seed, form_id, 301) % vm
    v = (raw & ~(1 << a_bit) & ~(1 << v_bit)) | (1 << v_bit)
    if v == 0:
        v = 1 << v_bit
    w = 1 << v_bit
    return int(v), int(w), int(a_bit), int(v_bit)


def _dihedral_fold(
    x: int | np.ndarray, s: int, x_size: int, fold_mode: int, translate: int
) -> int | np.ndarray:
    """Kuperberg-style involution, optionally after a cyclic translate."""
    s = int(s) % x_size
    t = int(translate) % x_size if int(fold_mode) % 2 else 0
    if isinstance(x, np.ndarray):
        y = (x + t) % x_size
        return np.minimum(y, (s - y) % x_size)
    y = (int(x) + t) % x_size
    return min(y, (s - y) % x_size)


def make_recipe_instance(n_bits: int, seed: int, recipe_id: int) -> FunctionInstance:
    """One analytic oracle from the 10k catalog. Secrets depend on ``seed``."""
    spec = decode_recipe(recipe_id)
    x_size = 1 << n_bits
    pairing = spec["pairing"]
    rank = int(spec.get("hidden_rank") or 1)
    break_frac = float(spec.get("structure_break") or 0.0)
    period_class = int(spec.get("period_class") or 0)
    fold_mode = int(spec.get("fold_mode") or 0)
    weight_class = int(spec.get("xor_weight_class") or 1)
    params: dict[str, Any] = {
        "recipe_id": spec["recipe_id"],
        "pairing": pairing,
        "hidden_rank": rank,
        "structure_break": break_frac,
        "usefulness_tier": spec["usefulness_tier"],
        "generator": spec["generator"],
        "period_class": period_class,
        "fold_mode": fold_mode,
        "xor_weight_class": weight_class,
    }
    if pairing == "random":
        params["structure_strength"] = 0.0
        return FunctionInstance(RECIPE_FAMILY, "gf2", n_bits, x_size, seed, params)
    if pairing == "xor":
        gens = _independent_gens(n_bits, rank, seed, 701, weight_class=weight_class)
        params["gens"] = list(gens)
        params["structure_strength"] = 1.0 - break_frac
        return FunctionInstance(RECIPE_FAMILY, "gf2", n_bits, x_size, seed, params)
    if pairing == "cyclic":
        params["r"] = cyclic_period(n_bits, rank, period_class)
        params["structure_strength"] = 1.0 - break_frac
        return FunctionInstance(RECIPE_FAMILY, "cyclic", n_bits, x_size, seed, params)
    if pairing == "dihedral":
        params["s"] = _label_hash(seed, 0, 103) % (x_size - 1) + 1
        params["translate"] = _label_hash(seed, 0, 109) % x_size
        params["structure_strength"] = 1.0 - break_frac
        return FunctionInstance(RECIPE_FAMILY, "cyclic", n_bits, x_size, seed, params)
    if pairing == "heisenberg":
        if n_bits < 4:
            gens = _independent_gens(n_bits, 1, seed, 701, weight_class=weight_class)
            params["pairing"] = "xor"
            params["gens"] = list(gens)
            params["generator"] = f"{RECIPE_FAMILY}.xor"
            params["structure_strength"] = 1.0 - break_frac
            return FunctionInstance(RECIPE_FAMILY, "gf2", n_bits, x_size, seed, params)
        v, w, a_bit, v_bit = _heisenberg_form(n_bits, int(spec.get("form_id", 0)), seed)
        params.update({"v": v, "w": w, "phi_a_bit": a_bit, "phi_v_bit": v_bit})
        params["structure_strength"] = 1.0 - break_frac
        return FunctionInstance(RECIPE_FAMILY, "gf2", n_bits, x_size, seed, params)
    if pairing == "quaternion":
        if n_bits < 8:
            gens = _independent_gens(n_bits, rank, seed, 701, weight_class=weight_class)
            params["pairing"] = "xor"
            params["gens"] = list(gens)
            params["generator"] = f"{RECIPE_FAMILY}.xor"
            params["structure_strength"] = 1.0 - break_frac
            return FunctionInstance(RECIPE_FAMILY, "gf2", n_bits, x_size, seed, params)
        extra_rank = max(0, rank - 1)
        if extra_rank:
            params["abelian_gens"] = list(
                _independent_gens(n_bits - 3, extra_rank, seed, 711, weight_class=weight_class)
            )
        else:
            params["abelian_gens"] = []
        params["structure_strength"] = 1.0 - break_frac
        return FunctionInstance(RECIPE_FAMILY, "gf2", n_bits, x_size, seed, params)
    if pairing == "blend":
        params["blend_a"] = spec["blend_a"]
        params["blend_b"] = spec["blend_b"]
        params["blend_weight"] = float(spec["blend_weight"])
        w = params["blend_weight"]
        coherence = 2.0 * max(w, 1.0 - w) - 1.0
        params["structure_strength"] = float(coherence)
        params["s"] = _label_hash(seed, 0, 121) % (x_size - 1) + 1
        params["r"] = cyclic_period(n_bits, rank, period_class)
        params["s_dihedral"] = _label_hash(seed, 0, 123) % (x_size - 1) + 1
        params["translate"] = _label_hash(seed, 0, 127) % x_size
        gens = _independent_gens(n_bits, rank, seed, 721, weight_class=weight_class)
        params["gens"] = list(gens)
        params["abelian_gens"] = (
            list(_independent_gens(max(1, n_bits - 3), 1, seed, 731, weight_class=weight_class))
            if n_bits >= 8
            else []
        )
        if n_bits >= 4:
            v, wbit, a_bit, v_bit = _heisenberg_form(n_bits, seed, seed)
            params.update({"v": v, "w": wbit, "phi_a_bit": a_bit, "phi_v_bit": v_bit})
        return FunctionInstance(RECIPE_FAMILY, "gf2", n_bits, x_size, seed, params)
    raise ValueError(f"unknown recipe pairing: {pairing!r}")


def _structured_coset(inst: FunctionInstance, pairing: str, x: int | np.ndarray) -> int | np.ndarray:
    p = inst.params
    if pairing == "xor":
        return xor_subspace_rep(x, tuple(int(g) for g in p["gens"]))
    if pairing == "cyclic":
        r = int(p["r"])
        return x % r if not isinstance(x, np.ndarray) else np.mod(x, r)
    if pairing == "dihedral":
        s = int(p.get("s_dihedral") or p["s"])
        return _dihedral_fold(
            x, s, inst.x_size, int(p.get("fold_mode") or 0), int(p.get("translate") or 0)
        )
    if pairing == "heisenberg":
        return _heisenberg_coset_repr(
            x,
            inst.n_bits,
            int(p["v"]),
            int(p["w"]),
            a_bit=int(p.get("phi_a_bit", 0)),
            v_bit=int(p.get("phi_v_bit", 1)),
        )
    if pairing == "quaternion":
        q_rep = _quaternion_coset_repr(x)
        gens = tuple(int(g) for g in p.get("abelian_gens") or [])
        if not gens:
            return q_rep
        if isinstance(q_rep, np.ndarray):
            q = q_rep & 7
            ab = q_rep >> 3
            ab_rep = xor_subspace_rep(ab, gens)
            return (ab_rep << 3) | q
        q = int(q_rep) & 7
        ab = int(q_rep) >> 3
        return (int(xor_subspace_rep(ab, gens)) << 3) | q
    raise ValueError(f"no coset rule for pairing {pairing!r}")


def evaluate_recipe(inst: FunctionInstance, x: int) -> int:
    x = x % inst.x_size
    p = inst.params
    pairing = p["pairing"]
    if pairing == "random":
        return _label_hash(inst.seed, x, salt=31)
    break_frac = float(p.get("structure_break") or 0.0)
    if pairing != "blend" and _bit_noise(inst.seed, x, salt=61, tau=break_frac):
        return _label_hash(inst.seed, x, salt=62)
    if pairing == "blend":
        if _uniform_unit(inst.seed, x, salt=21) < float(p["blend_weight"]):
            coset = _structured_coset(inst, str(p["blend_a"]), x)
        else:
            coset = _structured_coset(inst, str(p["blend_b"]), x)
        return _label_hash(inst.seed, int(coset), salt=63)
    coset = _structured_coset(inst, pairing, x)
    return _label_hash(inst.seed, int(coset), salt=63)


def evaluate_recipe_batch(inst: FunctionInstance, xs: np.ndarray) -> np.ndarray:
    p = inst.params
    pairing = p["pairing"]
    if pairing == "random":
        return _label_hash_batch(inst.seed, xs, salt=31)
    break_frac = float(p.get("structure_break") or 0.0)
    if pairing == "blend":
        via_a = _uniform_unit_batch(inst.seed, xs, salt=21) < float(p["blend_weight"])
        coset_a = np.asarray(_structured_coset(inst, str(p["blend_a"]), xs), dtype=np.int64)
        coset_b = np.asarray(_structured_coset(inst, str(p["blend_b"]), xs), dtype=np.int64)
        coset = np.where(via_a, coset_a, coset_b)
        return _label_hash_batch(inst.seed, coset, salt=63)
    broken = _bit_noise_batch(inst.seed, xs, salt=61, tau=break_frac)
    coset = np.asarray(_structured_coset(inst, pairing, xs), dtype=np.int64)
    structured = _label_hash_batch(inst.seed, coset, salt=63)
    independent = _label_hash_batch(inst.seed, xs, salt=62)
    return np.where(broken, independent, structured)


def required_recipe_generators() -> tuple[str, ...]:
    """Generator names a catalog population of decent size must contain."""
    return tuple(f"{RECIPE_FAMILY}.{name}" for name in (*PAIRINGS_STRUCTURED, "blend", "random"))
