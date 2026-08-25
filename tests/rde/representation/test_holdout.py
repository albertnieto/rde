"""Tests for the representation holdout / anti-cheating audit."""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import audit_holdout
from rde.representation.array_backend import NumpySearchBackend

BACKEND = NumpySearchBackend()


def test_audit_holdout_periodic_signal_is_honestly_uncertain_without_fourier_primitives():
    n = 8
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(0)
    periodic = np.stack([np.sin(3 * t + phase) for phase in rng.normal(size=8)])

    audit = audit_holdout(
        periodic,
        n=n,
        visible_primitives=["identity", "difference", "sorted_permutation"],
        held_out_primitives=["dft", "dft_full", "polynomial_vandermonde"],
        backend=BACKEND,
    )
    # No visible-only primitive should claim compression on this data.
    assert audit.leakage_ratio == pytest.approx(1.0)
    assert audit.honestly_uncertain
    # The full grammar must actually find the true (withheld) structure.
    assert audit.discovered_held_out_structure
    assert audit.full_best.representation_id in {"dft", "dft_full"}
    assert audit.visible_best.representation_id in {"identity", "difference"}


def test_audit_holdout_polynomial_batch_shows_partial_leakage_from_dft():
    # Real finding while building this module: a low-degree polynomial has
    # *some* spectral concentration too, so dft partially "leaks" even
    # though the true structure (polynomial_vandermonde) is withheld.
    n = 6
    nodes = np.arange(n, dtype=float)
    slopes = np.array([1.0, -2.0, 0.5, 3.0])
    intercepts = np.array([0.0, 5.0, -1.0, 2.0])
    batch = intercepts[:, None] + slopes[:, None] * nodes[None, :]

    audit = audit_holdout(
        batch,
        n=n,
        visible_primitives=["identity", "dft", "dft_full", "sorted_permutation"],
        held_out_primitives=["polynomial_vandermonde"],
        backend=BACKEND,
    )
    assert 0.0 < audit.leakage_ratio < 1.0
    assert audit.discovered_held_out_structure
    assert audit.full_best.representation_id == "polynomial_vandermonde"


def test_audit_holdout_uncertainty_threshold_changes_the_honest_flag():
    n = 6
    nodes = np.arange(n, dtype=float)
    slopes = np.array([1.0, -2.0, 0.5, 3.0])
    intercepts = np.array([0.0, 5.0, -1.0, 2.0])
    batch = intercepts[:, None] + slopes[:, None] * nodes[None, :]

    lenient = audit_holdout(
        batch,
        n=n,
        visible_primitives=["identity", "dft", "dft_full", "sorted_permutation"],
        held_out_primitives=["polynomial_vandermonde"],
        backend=BACKEND,
        uncertainty_threshold=0.1,
    )
    strict = audit_holdout(
        batch,
        n=n,
        visible_primitives=["identity", "dft", "dft_full", "sorted_permutation"],
        held_out_primitives=["polynomial_vandermonde"],
        backend=BACKEND,
        uncertainty_threshold=0.9,
    )
    assert lenient.honestly_uncertain
    assert not strict.honestly_uncertain
    assert lenient.leakage_ratio == strict.leakage_ratio


def test_audit_holdout_not_discovered_when_true_structure_also_withheld_from_full_grammar():
    # If held_out_primitives names something that isn't actually the winner
    # of the *full* grammar either, discovered_held_out_structure is False —
    # it checks the actual winner, not the caller's assumption.
    n = 6
    nodes = np.arange(n, dtype=float)
    slopes = np.array([1.0, -2.0, 0.5, 3.0])
    intercepts = np.array([0.0, 5.0, -1.0, 2.0])
    batch = intercepts[:, None] + slopes[:, None] * nodes[None, :]

    audit = audit_holdout(
        batch,
        n=n,
        visible_primitives=["identity", "difference"],
        held_out_primitives=["sorted_permutation"],  # not actually the true structure
        backend=BACKEND,
    )
    assert not audit.discovered_held_out_structure
    assert audit.full_best.representation_id == "polynomial_vandermonde"


def test_audit_holdout_chain_max_depth_never_leaks_withheld_primitive_into_visible_chains():
    # The genuine-withholding guarantee `program_search.atomic_registry`'s
    # primitive_subset docstring makes: a stage-2 primitive named in
    # held_out_primitives must not appear as any stage of any chain the
    # visible-only ranking considers, at any depth.
    n = 16
    t = np.arange(n)
    freqs = [1, 2, 3, 4, 5]
    batch = np.stack([sum(3.0 * np.cos(2 * np.pi * f * t / n) for f in freqs) for _ in range(4)])

    audit = audit_holdout(
        batch,
        n=n,
        visible_primitives=[
            "identity",
            "difference",
            "dft",
            "dft_full",
            "sorted_permutation",
            "sorted_then_difference",
        ],
        held_out_primitives=["sort_by_magnitude", "sorted_complex_then_difference"],
        backend=BACKEND,
        chain_max_depth=4,
    )
    visible_stages = set(audit.visible_best.representation_id.split("+"))
    assert not visible_stages & {"sort_by_magnitude", "sorted_complex_then_difference"}


def test_audit_holdout_chain_max_depth_discovers_held_out_structure_reached_only_via_a_chain():
    # Verified numerically before writing: on already-ascending piecewise-
    # linear data, identity+difference+difference and sorted_permutation+
    # sorted_then_difference+sorted_then_difference tie at complexity 3.0
    # (a near-identity sort permutation costs ~0, so both paths end up
    # equivalent) -- withholding plain `difference` still lets the
    # visible-only chain search match that complexity through the
    # sorted_permutation route, a genuine (if partial) leak this audit is
    # built to surface as a number, not hide behind a pass/fail flag.
    n = 12
    seg1 = np.linspace(0.0, 5.0, 7)
    seg2 = np.linspace(5.2, 20.0, 5)
    batch_row = np.concatenate([seg1, seg2])
    batch = np.stack([batch_row, batch_row.copy()])

    audit = audit_holdout(
        batch,
        n=n,
        visible_primitives=["identity", "sorted_permutation", "sorted_then_difference"],
        held_out_primitives=["difference"],
        backend=BACKEND,
        chain_max_depth=3,
    )
    assert audit.discovered_held_out_structure
    assert "difference" in audit.full_best.representation_id.split("+")
    assert "difference" not in audit.visible_best.representation_id.split("+")
    assert audit.visible_best.complexity == pytest.approx(audit.full_best.complexity)


def test_audit_holdout_rejects_unknown_visible_primitive_name():
    n = 6
    batch = np.zeros((3, n))
    with pytest.raises(ValueError):
        audit_holdout(
            batch, n=n, visible_primitives=["not_a_real_primitive"], held_out_primitives=[]
        )
