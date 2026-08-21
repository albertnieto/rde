"""Boolean/finite-domain hidden-structure descriptors (Direction F).

Two tiers, matching this project's POLYNOMIAL_INPUT / ENUMERATED_ORACLE
split (see `rde.core.feature_contract`):

- **Bounded-query (predictor-eligible).** Computed from a poly(n)-sized
  random sample of oracle query pairs and a poly(n)-sized sample of
  candidate difference values -- never from the full 2^n table. This is
  the tier a genuine query-limited discovery campaign is allowed to use as
  a predictor. Keys under ``hsp_sample.*``.
- **Full-table (oracle-only, exponential cost, audit/calibration use
  only).** Exact Walsh-Hadamard spectrum of a full 0/1-valued
  difference/autocorrelation table, exact algebraic degree via the
  Mobius/Zhegalkin transform. Gated to small n by the caller (mirrors this
  project's existing ``max_bruteforce_n`` convention). Keys under
  ``hsp_oracle.*``.

Both tiers reuse `rde.features.fourier.walsh_hadamard` as the underlying
fast transform rather than reimplementing it. See
`docs/research/hidden-subgroup-function-discovery-charter.md` S4.1 for why
the bounded-query / full-table split is not optional here.
"""

from __future__ import annotations

import numpy as np

from rde.features.fourier import walsh_hadamard


def mobius_transform_gf2(table01: np.ndarray) -> np.ndarray:
    """Algebraic-normal-form (Mobius/Zhegalkin) transform over GF(2).

    Same butterfly shape as the fast Walsh-Hadamard transform, but the
    combine step is GF(2) XOR (bottom ^= top) instead of +/-. Input and
    output both have length 2^k; ``anf[mask]`` is the ANF coefficient of
    the monomial product_{i in mask} x_i. Oracle-only (needs the full
    table); the stage loop is logarithmic, each stage is a vectorized
    XOR over a reshaped view (no per-element Python loop).
    """
    arr = np.asarray(table01, dtype=np.uint8).ravel().copy()
    n = arr.size
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Mobius transform requires length 2^k")
    h = 1
    while h < n:
        pairs = arr.reshape(-1, 2, h)
        pairs[:, 1, :] ^= pairs[:, 0, :]
        h *= 2
    return arr


def exact_algebraic_degree(table01: np.ndarray) -> int:
    """Exact algebraic degree of a full 0/1-valued truth table (oracle-only, O(2^n))."""
    anf = mobius_transform_gf2(table01)
    nonzero = np.flatnonzero(anf)
    if nonzero.size == 0:
        return 0
    idx = nonzero.astype(np.uint64)
    popcounts = np.zeros(idx.shape, dtype=np.int64)
    while np.any(idx):
        popcounts += (idx & 1).astype(np.int64)
        idx >>= np.uint64(1)
    return int(popcounts.max())


def exact_spectral_descriptors(diff_table01: np.ndarray, name: str = "diff") -> dict[str, float]:
    """Exact Walsh spectrum of a full 0/1-valued difference/autocorrelation table.

    ``diff_table01[d]`` should be an exact collision-indicator value (in
    [0, 1]) for shift ``d``, over the full domain d in {0,...,2^k-1}.
    Oracle-only: requires the full length-2^k table -- gate the caller to
    small n before calling this.
    """
    arr = np.asarray(diff_table01, dtype=float).ravel()
    n = arr.size
    if n == 0 or (n & (n - 1)) != 0:
        return {}
    pm1 = 2.0 * arr - 1.0
    coeffs = walsh_hadamard(pm1)
    abs_c = np.abs(coeffs)
    total = float(np.sum(abs_c))
    p = f"hsp_oracle.{name}"
    n_bits = int(np.log2(n))
    if total <= 0:
        return {f"{p}.sparsity": 1.0, f"{p}.max_coeff_normalized": 0.0}
    threshold = 1e-8 * abs_c.max()
    sparsity = float(np.count_nonzero(abs_c <= threshold) / n)
    return {
        f"{p}.sparsity": sparsity,
        f"{p}.max_coeff_normalized": float(abs_c.max()) / np.sqrt(n),
        f"{p}.l1_norm_normalized": total / (n * np.sqrt(n)),
        f"{p}.algebraic_degree_fraction": exact_algebraic_degree(arr > 0.5) / max(1, n_bits),
    }


def sample_difference_profile(diff_estimates: dict[int, float], name: str = "diff") -> dict[str, float]:
    """Bounded-query (predictor-eligible) descriptors of a sampled difference profile.

    ``diff_estimates`` maps a small number (poly(n_bits)) of candidate
    shift values ``d`` to an empirically-estimated collision probability
    g(d) = Pr_x[f(x) == f(x XOR/+ d)] in [0, 1], each itself estimated
    from a bounded number of queries by the caller. Never touches the full
    2^n table -- entropy/concentration over this small sample is the whole
    point: it is well-defined for any sample size, unlike an exact FWHT.
    """
    if not diff_estimates:
        return {}
    g = np.asarray(list(diff_estimates.values()), dtype=float)
    g = np.clip(g, 1e-12, 1.0)
    total = float(g.sum())
    probs = g / total
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(g.size)) if g.size > 1 else 1.0
    p = f"hsp_sample.{name}"
    return {
        f"{p}.mean_collision_prob": float(np.mean(g)),
        f"{p}.max_collision_prob": float(np.max(g)),
        f"{p}.concentration_fraction": float(np.count_nonzero(g > 0.5) / g.size),
        f"{p}.normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
    }


def gf2_rank(vectors: np.ndarray) -> int:
    """Rank of a set of GF(2) row vectors via Gaussian elimination.

    ``vectors``: (k, n_bits) array of 0/1 rows, k expected small (poly(
    n_bits) -- the number of collisions found within a bounded query
    budget). A dependency-ordered sequential algorithm on a small matrix:
    the same kind of intentional non-vectorized exception this project
    already allows for small-N setup / OMP-style algorithms (see
    `docs/engineering/agent-correction-playbook.md`).
    """
    mat = np.asarray(vectors, dtype=np.uint8).copy()
    if mat.size == 0:
        return 0
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if mat[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        mat[[rank, pivot]] = mat[[pivot, rank]]
        for r in range(rows):
            if r != rank and mat[r, col]:
                mat[r] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return rank
