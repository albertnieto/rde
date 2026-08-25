"""Tests for the generic enumerate/verify/holdout-rank engine (`rde.search`)."""

from __future__ import annotations

from dataclasses import dataclass

from rde.search import VerifyResult, search_with_holdout


@dataclass(frozen=True)
class _Candidate:
    name: str
    value: int


def _make_verify(pass_names: set[str], objective_by_name: dict[str, float]):
    def _verify(candidates, _domain, _batch):
        return {
            c.name: VerifyResult(ok=c.name in pass_names, objective=objective_by_name.get(c.name, 0.0))
            for c in candidates
        }

    return _verify


def test_drops_candidates_failing_train():
    candidates = [_Candidate("a", 1), _Candidate("b", 2)]
    verify = _make_verify(pass_names=set(), objective_by_name={})
    results = search_with_holdout(
        candidates, "train", "holdout", verify=verify, candidate_id=lambda c: c.name
    )
    assert results == []


def test_drops_candidates_passing_train_but_failing_holdout():
    calls = []

    def verify(candidates, _domain, batch):
        calls.append((batch, tuple(c.name for c in candidates)))
        if batch == "train":
            return {c.name: VerifyResult(ok=True, objective=1.0) for c in candidates}
        return {c.name: VerifyResult(ok=False, objective=1.0) for c in candidates}

    candidates = [_Candidate("a", 1)]
    results = search_with_holdout(
        candidates, "train", "holdout", verify=verify, candidate_id=lambda c: c.name
    )
    assert results == []
    # holdout verify is only called with train-survivors, never all candidates blindly.
    assert calls[1] == ("holdout", ("a",))


def test_survivors_ranked_by_holdout_objective_ascending_by_default():
    candidates = [_Candidate("a", 1), _Candidate("b", 2), _Candidate("c", 3)]

    def verify(candidates, _domain, batch):
        objective = {"train": {"a": 1.0, "b": 1.0, "c": 1.0}, "holdout": {"a": 3.0, "b": 1.0, "c": 2.0}}[
            batch
        ]
        return {c.name: VerifyResult(ok=True, objective=objective[c.name]) for c in candidates}

    results = search_with_holdout(
        candidates, "train", "holdout", verify=verify, candidate_id=lambda c: c.name
    )
    assert [r.candidate_id for r in results] == ["b", "c", "a"]


def test_higher_is_better_reverses_rank_order():
    candidates = [_Candidate("a", 1), _Candidate("b", 2)]

    def verify(candidates, _domain, batch):
        objective = {"a": 0.5, "b": 0.9}
        return {c.name: VerifyResult(ok=True, objective=objective[c.name]) for c in candidates}

    results = search_with_holdout(
        candidates,
        "train",
        "holdout",
        verify=verify,
        candidate_id=lambda c: c.name,
        higher_is_better=True,
    )
    assert [r.candidate_id for r in results] == ["b", "a"]


def test_train_and_holdout_verify_results_are_both_reported():
    candidates = [_Candidate("a", 1)]

    def verify(candidates, _domain, batch):
        objective = 10.0 if batch == "train" else 4.0
        return {c.name: VerifyResult(ok=True, objective=objective, detail=batch) for c in candidates}

    results = search_with_holdout(
        candidates, "train", "holdout", verify=verify, candidate_id=lambda c: c.name
    )
    assert len(results) == 1
    result = results[0]
    assert result.train.objective == 10.0
    assert result.train.detail == "train"
    assert result.holdout.objective == 4.0
    assert result.holdout.detail == "holdout"


def test_empty_candidates_returns_empty_list():
    def verify(candidates, _domain, _batch):
        return {}

    results = search_with_holdout([], "train", "holdout", verify=verify, candidate_id=lambda c: c.name)
    assert results == []
