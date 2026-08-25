"""Tests for `adaptive.py`'s data-adapted KLT/PCA primitive and its preregistered

holdout comparison against `grammar.py`'s fixed 8-primitive grammar. Every
number asserted here was independently verified before being written, the
same standard `test_grammar.py`/`test_structure.py` hold themselves to: the
preregistered target family really does let `klt` reach complexity `3.0`
(exactly `k`) against `dft`'s `9.0`; the noisy variant really does show no
compression for anyone at this grammar's `eps=1e-6` threshold (documented as
an honest negative result in `adaptive.py`'s module docstring, not hidden).
"""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation.adaptive import (
    KltHoldoutComparison,
    KltNoiseSensitivityPoint,
    _best_grammar_holdout_complexity,
    build_klt_representation,
    fit_klt_basis,
    low_rank_factor_batch,
    run_klt_holdout_comparison,
    run_klt_noise_sensitivity,
)
from rde.representation.array_backend import NumpySearchBackend
from rde.representation.certificate import certify_roundtrip
from rde.representation.grammar import primitive_names

BACKEND = NumpySearchBackend()


def test_low_rank_factor_batch_is_exactly_rank_k_with_no_noise():
    batch = low_rank_factor_batch(16, 3, 200, seed=0)
    singular_values = np.linalg.svd(batch, compute_uv=False)
    assert np.sum(singular_values > 1e-8) == 3


def test_low_rank_factor_batch_shares_loading_matrix_across_seeds():
    """Train and holdout batches must be the same distribution (same `A`),

    not just the same shape -- otherwise a holdout "comparison" would be
    meaningless. `A` is recovered as the row space each batch spans.
    """
    train = low_rank_factor_batch(16, 3, 200, seed=0)
    holdout = low_rank_factor_batch(16, 3, 200, seed=1)
    _, _, vt_train = np.linalg.svd(train, full_matrices=False)
    _, _, vt_holdout = np.linalg.svd(holdout, full_matrices=False)
    train_subspace = vt_train[:3]
    holdout_subspace = vt_holdout[:3]
    # principal angles between the two rank-3 subspaces should be ~0
    overlap = np.linalg.svd(train_subspace @ holdout_subspace.T, compute_uv=False)
    assert np.allclose(overlap, 1.0, atol=1e-6)


def test_fit_klt_basis_is_orthonormal():
    batch = low_rank_factor_batch(16, 3, 200, seed=0)
    basis = fit_klt_basis(batch)
    assert np.allclose(basis.T @ basis, np.eye(16), atol=1e-9)


def test_fit_klt_basis_sorts_by_descending_eigenvalue():
    batch = low_rank_factor_batch(16, 3, 500, seed=0)
    basis = fit_klt_basis(batch)  # rows are basis vectors
    projected_variance = np.var(batch @ basis.T, axis=0)
    assert np.all(np.diff(projected_variance) <= 1e-9)  # non-increasing


def test_klt_representation_roundtrips_exactly():
    train = low_rank_factor_batch(16, 3, 500, seed=0)
    holdout = low_rank_factor_batch(16, 3, 200, seed=1)
    representation = build_klt_representation(train, backend=BACKEND)
    certificate = certify_roundtrip(representation, holdout, tolerance=1e-6)
    assert certificate.status == "verified"


def test_klt_is_not_in_grammars_fixed_primitive_registry():
    """Architectural boundary: `klt` needs data, every `grammar.py` primitive

    needs only `n` -- `klt` must never silently appear in `primitive_names()`.
    """
    assert "klt" not in primitive_names()


def test_preregistered_klt_holdout_comparison_beats_the_margin():
    """The actual preregistered result (see `adaptive.py` module docstring):

    klt reaches holdout complexity 3.0 (exactly k) against dft's 9.0 --
    ratio 0.333, under the 0.5 margin fixed in advance.
    """
    result = run_klt_holdout_comparison()
    assert isinstance(result, KltHoldoutComparison)
    assert result.klt_holdout_complexity == pytest.approx(3.0, abs=1e-6)
    assert result.best_grammar_representation_id == "dft"
    assert result.best_grammar_holdout_complexity == pytest.approx(9.0, abs=1e-6)
    assert result.klt_beats_grammar_by_margin


def test_noisy_variant_shows_no_compression_for_anyone_honest_negative_result():
    """Documented honest limitation: with noise added, every coefficient

    (including the 13 "noise" directions) exceeds this grammar's real
    eps=1e-6 threshold, so klt shows full density -- same as identity. This
    is exactly why the preregistered target family is noise-free.
    """
    result = run_klt_holdout_comparison(noise_scale=0.05)
    assert result.klt_holdout_complexity == pytest.approx(16.0, abs=1e-6)
    assert not result.klt_beats_grammar_by_margin


def test_best_grammar_holdout_complexity_raises_a_clear_error_when_nothing_verifies(monkeypatch):
    """Robustness gap closed: `min()` on an empty dict used to raise an opaque

    `ValueError` three frames deep -- now a `RuntimeError` that names the
    actual problem. Forced via monkeypatch (every primitive verifies its own
    roundtrip at every `n` this grammar actually supports, so there is no
    real batch that reaches this branch -- this only demonstrates the guard
    fires when it must).
    """
    import rde.representation.adaptive as adaptive_module

    def _always_refuted(*args, **kwargs):
        from rde.representation.certificate import Certificate

        return Certificate(
            representation_id="x", object_type="x", claim="x", status="refuted", error=1.0, tolerance=0.0
        )

    monkeypatch.setattr(adaptive_module, "certify_roundtrip", _always_refuted)
    with pytest.raises(RuntimeError, match="No grammar.py primitive verified"):
        _best_grammar_holdout_complexity(low_rank_factor_batch(16, 3, 10, seed=0), 16, BACKEND)


def test_noise_sensitivity_sweep_is_flat_below_eps_and_saturates_above_it():
    """The real result the train-count sweep didn't have to show: compression

    survives noise well below this grammar's eps=1e-6 threshold, degrades
    smoothly through it, and saturates at full density above it -- verified
    numerically before being written (see `adaptive.py`'s
    `run_klt_noise_sensitivity` docstring for the full curve).
    """
    points = run_klt_noise_sensitivity()
    by_noise = {p.noise_scale: p for p in points}
    assert isinstance(points[0], KltNoiseSensitivityPoint)
    # an order of magnitude below eps: compression untouched
    assert by_noise[1e-7].klt_holdout_complexity == pytest.approx(3.0, abs=1e-6)
    # well above eps: compression gone, same as identity
    assert by_noise[1e-2].klt_holdout_complexity == pytest.approx(16.0, abs=0.05)
    # monotonically non-decreasing across the sweep -- a real transition, not noise
    complexities = [p.klt_holdout_complexity for p in points]
    assert all(a <= b + 1e-9 for a, b in zip(complexities, complexities[1:]))


def test_train_count_does_not_affect_the_noise_free_result():
    """The honest reason a train-count sweep isn't a swept parameter of

    `run_klt_noise_sensitivity`: with zero noise, 3 training samples already
    exactly span the true rank-3 subspace, so there is no estimation
    variance left for more samples to reduce.
    """
    small = run_klt_holdout_comparison(train_count=3)
    large = run_klt_holdout_comparison(train_count=500)
    assert small.klt_holdout_complexity == pytest.approx(large.klt_holdout_complexity, abs=1e-6)
