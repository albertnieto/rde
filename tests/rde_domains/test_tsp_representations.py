"""Tests for `rde.representation` applied to real TSP distance data."""

from __future__ import annotations

import numpy as np

from rde_domains.tsp.components import distance_matrix, random_euclidean_points
from rde_domains.tsp.representations import (
    distance_profile_batch,
    rank_distance_profile_representations,
    upper_triangular_distances,
)


def test_upper_triangular_distances_length_and_values():
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    profile = upper_triangular_distances(D)
    assert profile.shape == (3,)
    assert np.array_equal(profile, [1.0, 2.0, 3.0])


def test_upper_triangular_distances_matches_direct_generation():
    rng = np.random.default_rng(3)
    points = random_euclidean_points(rng, 5)
    D = distance_matrix(points)
    profile = upper_triangular_distances(D)
    n = D.shape[0]
    expected_length = n * (n - 1) // 2
    assert profile.shape == (expected_length,)
    rows, cols = np.triu_indices(n, k=1)
    assert np.array_equal(profile, D[rows, cols])


def test_distance_profile_batch_shape():
    rng = np.random.default_rng(0)
    batch = distance_profile_batch(rng, n_cities=6, batch_size=8)
    assert batch.shape == (8, 6 * 5 // 2)


def test_distance_profile_batch_rows_are_independent_instances():
    rng = np.random.default_rng(0)
    batch = distance_profile_batch(rng, n_cities=5, batch_size=4)
    # Independently generated random instances should not be identical rows.
    assert not np.allclose(batch[0], batch[1])


def test_distance_profile_entries_are_nonnegative():
    rng = np.random.default_rng(1)
    batch = distance_profile_batch(rng, n_cities=6, batch_size=5)
    assert np.all(batch >= 0.0)


def test_rank_distance_profile_representations_runs_on_real_geometry_and_verifies():
    rng = np.random.default_rng(2)
    ranked = rank_distance_profile_representations(rng, n_cities=6, batch_size=12)
    assert len(ranked) > 0
    verified_ids = {c.representation_id for c in ranked if c.certificate.status == "verified"}
    assert {"identity", "dft_full", "difference"}.issubset(verified_ids)


def test_rank_distance_profile_representations_is_sorted_by_complexity_among_verified():
    rng = np.random.default_rng(4)
    ranked = rank_distance_profile_representations(rng, n_cities=6, batch_size=12)
    verified = [c for c in ranked if c.certificate.status == "verified"]
    complexities = [c.complexity for c in verified]
    assert complexities == sorted(complexities)
