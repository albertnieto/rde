"""Tests for the MAP-Elites-style diversity archive (`rde.discovery.archive`)."""

from __future__ import annotations

import pytest

from rde.discovery.archive import EliteArchive, archive_candidates


def test_rejects_non_positive_resolution():
    with pytest.raises(ValueError):
        EliteArchive(resolution=(1.0, 0.0))


def test_rejects_empty_resolution():
    with pytest.raises(ValueError):
        EliteArchive(resolution=())


def test_descriptor_axis_count_must_match_resolution():
    archive = EliteArchive(resolution=(1.0, 1.0))
    with pytest.raises(ValueError):
        archive.add("candidate", descriptor=(1.0,), fitness=0.5)


def test_add_returns_true_for_first_occupant_of_a_bucket():
    archive = EliteArchive(resolution=(1.0,))
    assert archive.add("a", descriptor=(0.5,), fitness=1.0) is True
    assert len(archive) == 1


def test_higher_fitness_replaces_incumbent_in_same_bucket():
    archive = EliteArchive(resolution=(1.0,))
    archive.add("low", descriptor=(0.1,), fitness=1.0)
    replaced = archive.add("high", descriptor=(0.2,), fitness=5.0)
    assert replaced is True
    assert len(archive) == 1
    assert archive.elites[0].candidate == "high"
    assert archive.elites[0].fitness == 5.0


def test_lower_or_equal_fitness_does_not_replace_incumbent():
    archive = EliteArchive(resolution=(1.0,))
    archive.add("first", descriptor=(0.1,), fitness=5.0)
    replaced = archive.add("second", descriptor=(0.2,), fitness=5.0)
    assert replaced is False
    assert archive.elites[0].candidate == "first"


def test_different_buckets_coexist():
    archive = EliteArchive(resolution=(1.0,))
    archive.add("a", descriptor=(0.5,), fitness=1.0)
    archive.add("b", descriptor=(1.5,), fitness=1.0)
    assert len(archive) == 2
    buckets = {elite.bucket for elite in archive.elites}
    assert buckets == {(0,), (1,)}


def test_insertion_order_does_not_affect_final_archive():
    resolution = (1.0,)
    forward = EliteArchive(resolution=resolution)
    forward.add("low", descriptor=(0.0,), fitness=1.0)
    forward.add("high", descriptor=(0.0,), fitness=9.0)

    backward = EliteArchive(resolution=resolution)
    backward.add("high", descriptor=(0.0,), fitness=9.0)
    backward.add("low", descriptor=(0.0,), fitness=1.0)

    assert forward.elites[0].candidate == backward.elites[0].candidate == "high"


def test_archive_candidates_convenience_wrapper():
    candidates = [(1, 0.9), (2, 0.9), (3, 5.0)]
    archive = archive_candidates(
        candidates,
        descriptor_fn=lambda c: (float(c[0] % 2),),
        fitness_fn=lambda c: c[1],
        resolution=(1.0,),
    )
    # candidates 1 and 3 share bucket (1.0,) [odd]; candidate 2 is bucket (0.0,) [even].
    assert len(archive) == 2
    fitnesses = sorted(elite.fitness for elite in archive.elites)
    assert fitnesses == [0.9, 5.0]
