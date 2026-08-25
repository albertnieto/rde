"""Circulant-symmetry TSP domain.

Generates Euclidean TSP instances with a tunable, planted degree of broken
cyclic (Z_N) symmetry: at `symmetry_break=0`, cities sit exactly on a
regular N-gon, so the distance matrix D is exactly circulant (invariant
under simultaneous cyclic shift of rows and columns -- the signature of a
hidden abelian Z_N structure, the group Simon/Shor-style Fourier sampling
exploits). Increasing `symmetry_break` perturbs city positions with
Gaussian noise, degrading that symmetry continuously.

Purpose: give RDE's forward discovery loop a real, non-circular target --
`circulant_deviation`, computed directly from D by projecting onto the
circulant subspace, independent of any brute-force tour computation -- and
ask whether raw-matrix descriptors can detect/predict planted symmetry
breaking. This is Mode 1 measure-Z only: no SynthesisDomain / Mode 2 protocol,
no brute-force tour ground truth, so it scales past the N<=9 cap that
`TspSynthesisDomain` needs. This domain only characterizes detectability of
the symmetry itself; whether detectable circulant structure translates into
an actual algorithmic advantage is a separate, unresolved question.
"""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice
from rde_domains.tsp.components import distance_matrix


def _circulant_projection(D: np.ndarray) -> np.ndarray:
    """Average D over all N simultaneous cyclic row/column shifts.

    The result is the projection of D onto the subspace of circulant
    matrices under the regular representation of Z_N acting by conjugation
    -- exactly the matrices diagonalized by the discrete Fourier transform.
    """
    n = D.shape[0]
    idx = np.arange(n)
    acc = np.zeros_like(D)
    for s in range(n):
        shifted = (idx + s) % n
        acc += D[np.ix_(shifted, shifted)]
    return acc / n


def circulant_deviation(D: np.ndarray) -> float:
    """Fraction of D's Frobenius norm lying outside the circulant subspace.

    0.0 = exactly circulant (perfect hidden Z_N symmetry). Grows toward 1.0
    as the symmetry is broken. Computed directly from D (O(N^3), no brute
    force), so it is a real target independent of the mechanism being
    searched for -- not derived from any decomposition move.
    """
    d = np.asarray(D, dtype=float)
    proj = _circulant_projection(d)
    denom = np.linalg.norm(d)
    if denom < 1e-15:
        return 0.0
    return float(np.linalg.norm(d - proj) / denom)


def _points_on_circle(n: int, radius: float) -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


class TspCirculantSymmetryDomain:
    """Domain (Mode 1 measure-Z only) over circulant-symmetry-broken TSP instances."""

    def __init__(
        self,
        *,
        domain_id: str = "tsp_circulant_symmetry",
        max_symmetry_break: float = 0.6,
        radius: float = 1.0,
    ) -> None:
        self.domain_id = domain_id
        #: Upper bound on the per-instance uniform draw for the Gaussian
        #: perturbation scale (as a fraction of `radius`). 0.6 was chosen to
        #: span from near-exact-circulant to visibly generic without most
        #: draws landing in self-intersecting/degenerate point clouds.
        self.max_symmetry_break = max_symmetry_break
        self.radius = radius

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        instances: list[InstanceRecord] = []
        for i in range(n):
            inst_seed = seed + i
            rng = np.random.default_rng(inst_seed)
            symmetry_break = float(rng.uniform(0.0, self.max_symmetry_break))
            points = _points_on_circle(size, self.radius)
            noise_scale = symmetry_break * self.radius
            if noise_scale > 0:
                points = points + rng.normal(0.0, noise_scale, size=points.shape)
            d = distance_matrix(points)
            instances.append(
                InstanceRecord(
                    domain_id=self.domain_id,
                    size=size,
                    seed=inst_seed,
                    params={
                        "points": points.tolist(),
                        "D": d.tolist(),
                        #: Generation-time knob, kept for sanity-checking the
                        #: generator only -- never eligible as a predictor
                        #: (it would leak the answer since it's the noise
                        #: scale used to produce `circulant_deviation`, not
                        #: an independently-measured structural fact).
                        "symmetry_break_param": symmetry_break,
                    },
                )
            )
        return instances

    def materialize(self, instance: InstanceRecord, index: int) -> SimpleFamilySlice:
        d = np.asarray(instance.params["D"], dtype=float)
        return SimpleFamilySlice(values=d.sum(axis=1), index=index, kind="row_sums")

    def primitive_features(self, instance: InstanceRecord) -> dict[str, float | np.ndarray]:
        d = np.asarray(instance.params["D"], dtype=float)
        return {
            "n_cities": float(d.shape[0]),
            "circulant_deviation": circulant_deviation(d),
            "D": d,
        }


def circulant_symmetry_domain(**kwargs) -> TspCirculantSymmetryDomain:
    return TspCirculantSymmetryDomain(**kwargs)
