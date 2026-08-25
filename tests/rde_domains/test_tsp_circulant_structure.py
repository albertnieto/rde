"""`rde.representation.structure`'s `check_conservation`/`check_duality`, verified against
real `tsp_circulant_symmetry`-generated distance matrices, not fabricated data.

Domain-integration gap closure: `structure.py`'s conservation/duality checks are
domain-agnostic core machinery (core must not import `rde_domains`), generalized from
`rde_domains.tsp.circulant`'s `circulant_deviation` idea. This module is the actual proof
that generalization is sound against the real domain object it was built to describe — a
genuinely circulant TSP distance matrix (`symmetry_break=0`) really is conserved under
cyclic shift and really is diagonalized by the full DFT, and a symmetry-broken one really
degrades continuously, not fabricated positive/negative examples.
"""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import check_conservation, check_duality
from rde_domains.tsp.circulant import TspCirculantSymmetryDomain, circulant_deviation


def test_check_conservation_holds_for_exactly_circulant_tsp_distance_matrix():
    dom = TspCirculantSymmetryDomain(max_symmetry_break=0.0)
    [inst] = dom.generate(n=1, size=8, seed=1)
    d = np.asarray(inst.params["D"], dtype=float)
    claim = check_conservation(d[None, :, :])
    assert claim.holds
    assert claim.score < 1e-9


def test_check_conservation_score_matches_domains_own_circulant_deviation():
    # check_conservation's default (cyclic-group permutations) is a direct
    # generalization of circulant_deviation's formula -- for the exact
    # cyclic-shift case they must agree bit-for-bit, not just "both small".
    dom = TspCirculantSymmetryDomain(max_symmetry_break=0.6)
    insts = dom.generate(n=6, size=8, seed=2)
    for inst in insts:
        d = np.asarray(inst.params["D"], dtype=float)
        domain_deviation = circulant_deviation(d)
        claim = check_conservation(d[None, :, :])
        assert claim.score == domain_deviation


def test_check_conservation_fails_for_visibly_symmetry_broken_instances():
    dom = TspCirculantSymmetryDomain(max_symmetry_break=0.6)
    insts = dom.generate(n=8, size=8, seed=3)
    for inst in insts:
        if inst.params["symmetry_break_param"] > 0.1:
            d = np.asarray(inst.params["D"], dtype=float)
            claim = check_conservation(d[None, :, :])
            assert not claim.holds


def test_check_conservation_score_correlates_with_symmetry_break_param():
    # symmetry_break_param sets the *scale* of a per-instance random
    # perturbation, so deviation is not a deterministic (strictly
    # monotonic) function of it for any single instance -- checked over
    # enough instances for the correlation to be the honest claim, not a
    # per-instance ordering guarantee.
    dom = TspCirculantSymmetryDomain(max_symmetry_break=0.6)
    insts = dom.generate(n=60, size=8, seed=3)
    symmetry_break = np.array([inst.params["symmetry_break_param"] for inst in insts])
    scores = np.array(
        [
            check_conservation(np.asarray(inst.params["D"], dtype=float)[None, :, :]).score
            for inst in insts
        ]
    )
    correlation = np.corrcoef(symmetry_break, scores)[0, 1]
    assert correlation > 0.8


def test_check_duality_holds_for_exactly_circulant_tsp_distance_matrix():
    dom = TspCirculantSymmetryDomain(max_symmetry_break=0.0)
    [inst] = dom.generate(n=1, size=8, seed=1)
    d = np.asarray(inst.params["D"], dtype=float)
    claim = check_duality(d[None, :, :], n=8)
    assert claim.holds
    assert claim.score < 1e-9


def test_check_duality_fails_for_symmetry_broken_tsp_distance_matrix():
    dom = TspCirculantSymmetryDomain(max_symmetry_break=0.6)
    [inst] = dom.generate(n=1, size=8, seed=4)
    d = np.asarray(inst.params["D"], dtype=float)
    assert inst.params["symmetry_break_param"] > 0.05  # sanity: this draw is not near-exact
    claim = check_duality(d[None, :, :], n=8)
    assert not claim.holds


def test_check_conservation_and_check_duality_agree_on_real_tsp_data():
    # The same numerical-equality finding test_structure.py verified on a
    # synthetic circulant matrix, confirmed here on genuine domain data too
    # -- not an artifact of the synthetic construction.
    dom = TspCirculantSymmetryDomain(max_symmetry_break=0.6)
    insts = dom.generate(n=8, size=8, seed=5)
    for inst in insts:
        d = np.asarray(inst.params["D"], dtype=float)
        conservation = check_conservation(d[None, :, :])
        duality = check_duality(d[None, :, :], n=8)
        assert conservation.score == pytest.approx(duality.score, rel=1e-6, abs=1e-9)
