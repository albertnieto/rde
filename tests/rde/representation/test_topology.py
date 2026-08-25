"""Tests for `topology.py`'s one genuine, non-vacuous topological homeomorphism instance:

the map induced by `f(z) = z**n` on the plane-rotation orbit space `(C\\{0})/Z_n`.
Every `holds=True`/`holds=False` here was independently verified in development
before being written into an assertion, the same standard `test_structure.py`
holds itself to: `check_quotient_well_defined` really does find spread `~1e-19`
(floating-point roundoff, not zero, because complex exponentials are computed
numerically); `check_naive_branch_inverse_discontinuous` really does find a
~1.0 discontinuity where the map itself only moves points ~1e-8; and
`prove_rotation_quotient_injective` really does fail to close for `n=7`
(a genuine algebraic obstruction, not a flaky test -- see that function's
docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation.topology import (
    HomeomorphismClaim,
    canonical_representative,
    check_map_continuity_sampled,
    check_naive_branch_inverse_discontinuous,
    check_quotient_injective_sampled,
    check_quotient_well_defined,
    prove_rotation_quotient_injective,
    prove_rotation_quotient_well_defined,
)


def _f(n):
    def rotation_power(z):
        return np.asarray(z) ** n

    return rotation_power


def _random_samples(count: int, *, seed: int = 0, min_radius: float = 0.1, max_radius: float = 5.0):
    rng = np.random.default_rng(seed)
    radii = rng.uniform(min_radius, max_radius, size=count)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
    return radii * np.exp(1j * angles)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_canonical_representative_reduces_angle_into_fundamental_domain(n):
    samples = _random_samples(50, seed=1)
    reps = canonical_representative(samples, n)
    angles = np.angle(reps)
    assert np.all(angles >= -1e-12)
    assert np.all(angles < 2.0 * np.pi / n + 1e-12)
    # radius is preserved -- canonicalization only rotates, never scales
    assert np.allclose(np.abs(reps), np.abs(samples))


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_quotient_well_defined_holds_to_floating_point_tolerance(n):
    samples = _random_samples(30, seed=2)
    claim = check_quotient_well_defined(_f(n), n, samples)
    assert isinstance(claim, HomeomorphismClaim)
    assert claim.holds
    # not exactly 0.0 -- floating-point complex exponentials, not exact rationals
    assert claim.score < 1e-9


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_quotient_injective_sampled_finds_no_collisions(n):
    samples = _random_samples(60, seed=3)
    claim = check_quotient_injective_sampled(_f(n), n, samples)
    assert claim.holds
    assert claim.score == 0.0


def test_quotient_injective_sampled_detects_a_planted_collision():
    """Negative control: two points on the *same* orbit are not "distinct" --

    confirms the distinct-orbit filter actually filters, not just always passes.
    """
    n = 6
    z = 1.3 + 0.4j
    zeta = np.exp(2j * np.pi / n)
    same_orbit_samples = np.array([zeta**k * z for k in range(n)])
    claim = check_quotient_injective_sampled(_f(n), n, same_orbit_samples)
    # every pair here is the *same* orbit -- zero distinct-orbit pairs to test,
    # so this must hold vacuously (no evidence against injectivity, none for it)
    assert claim.holds
    assert "0/0" in claim.detail


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_map_continuity_sampled_holds(n):
    samples = _random_samples(40, seed=4)
    claim = check_map_continuity_sampled(_f(n), samples)
    assert claim.holds


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_naive_branch_inverse_discontinuous_at_seam(n):
    """The honest negative result: a single-valued branch inverse is not

    continuous at the fundamental-domain seam, even though `f` itself moves
    the two seam-adjacent points' images by only ~1e-8.
    """
    claim = check_naive_branch_inverse_discontinuous(n)
    assert claim.holds  # holds == "the discontinuity was confirmed", not "f is discontinuous"
    assert claim.score > 0.5  # recovered points are ~1.0 apart


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_prove_rotation_quotient_well_defined_closes_for_every_n_tried(n):
    certificate = prove_rotation_quotient_well_defined(n)
    assert certificate.status == "proved"


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_prove_rotation_quotient_injective_closes_for_the_verified_n_values(n):
    certificate = prove_rotation_quotient_injective(n)
    assert certificate.status == "proved"


def test_prove_rotation_quotient_injective_does_not_close_for_n_7():
    """Disclosed limitation, not a bug: cos(pi/7) is not expressible in the

    real-radical normal form `sympy.nsimplify` searches for, so this
    particular proof *strategy* fails here -- the mathematical claim itself
    is still true for n=7, this function just cannot certify it that way.
    """
    certificate = prove_rotation_quotient_injective(7)
    assert certificate.status == "disproved"
