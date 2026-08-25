"""Tests for `rde.representation` applied to real hsp_functions data."""

from __future__ import annotations

import numpy as np

from rde_domains.hsp_functions.functions import ALL_FAMILIES, FAMILIES_HELD_OUT, make_instance
from rde_domains.hsp_functions.representations import diff_profile_batch, rank_diff_profile_representations
from rde_domains.hsp_functions.sampling import sample_difference_estimates


def test_diff_profile_batch_shape_matches_families_times_instances():
    batch, families = diff_profile_batch(n_bits=8, instances_per_family=3, seed=0)
    assert batch.shape == (len(ALL_FAMILIES) * 3, 8)
    assert len(families) == len(ALL_FAMILIES) * 3
    assert set(families) == set(ALL_FAMILIES)


def test_diff_profile_batch_matches_direct_sampling_for_one_instance():
    instance = make_instance("simon", n_bits=6, seed=42)
    rng = np.random.default_rng(42 ^ 0x1234_5678)
    expected = np.array(list(sample_difference_estimates(instance, rng).values()), dtype=float)

    batch, families = diff_profile_batch(n_bits=6, families=["simon"], instances_per_family=1, seed=42)
    assert families == ["simon"]
    assert np.array_equal(batch[0], expected)


def test_diff_profile_batch_is_reproducible_for_fixed_seed():
    batch_a, _ = diff_profile_batch(n_bits=6, instances_per_family=2, seed=7)
    batch_b, _ = diff_profile_batch(n_bits=6, instances_per_family=2, seed=7)
    assert np.array_equal(batch_a, batch_b)


def test_diff_profile_batch_supports_held_out_families_only():
    batch, families = diff_profile_batch(
        n_bits=6, families=FAMILIES_HELD_OUT, instances_per_family=2, seed=0
    )
    assert batch.shape == (len(FAMILIES_HELD_OUT) * 2, 6)
    assert set(families) == set(FAMILIES_HELD_OUT)


def test_rank_diff_profile_representations_runs_on_real_data_and_verifies():
    ranked = rank_diff_profile_representations(n_bits=8, instances_per_family=3, seed=0)
    assert len(ranked) > 0
    # At least the exact primitives (identity, matrix_reshape if applicable,
    # difference, dft_full) must certify on real bounded-query data.
    verified_ids = {c.representation_id for c in ranked if c.certificate.status == "verified"}
    assert {"identity", "dft_full", "difference"}.issubset(verified_ids)


def test_rank_diff_profile_representations_is_sorted_by_complexity_among_verified():
    ranked = rank_diff_profile_representations(n_bits=8, instances_per_family=2, seed=1)
    verified = [c for c in ranked if c.certificate.status == "verified"]
    complexities = [c.complexity for c in verified]
    assert complexities == sorted(complexities)
