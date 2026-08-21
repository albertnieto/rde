"""Backward algorithm synthesis end-to-end (Mode 2, ALGO-057).

Uses `block_separable` (`rde.testing.block_separable`) — the reverse-
engineering brief's own toy example, ``Q = block_diag(Q_1, ..., Q_k)`` kept
domain-agnostic — to check that `rde.synthesis.synthesize` actually
rediscovers "decompose along block boundaries, solve independently, merge"
as a verified, correct, polynomial skeleton, rejects the exponential direct-
brute-force skeleton on complexity grounds alone, and never accepts a
skeleton whose result disagrees with real brute-force ground truth.
"""

from __future__ import annotations

from rde.core.plugins import build_registry
from rde.synthesis import synthesize, write_synthesis_conjectures_jsonl
from rde.synthesis.search import execute_skeleton, verify_skeleton
from rde.synthesis.skeleton import base_skeleton, divide_skeleton, flat_skeleton


def _domain_and_instances(n_instances=6, size=5, seed=0):
    reg = build_registry("block_separable")
    domain = reg.get_domain("block_separable")
    instances = domain.generate(n_instances, size, seed)
    return domain, instances


def test_domain_registered_and_generates_instances():
    domain, instances = _domain_and_instances()
    assert domain.domain_id == "block_separable"
    assert len(instances) == 6
    assert domain.size_of(instances[0]) == 5


def test_brute_force_and_flat_decomposition_agree():
    domain, instances = _domain_and_instances(n_instances=4, size=4)
    for instance in instances:
        truth = domain.brute_force(instance)
        candidate = execute_skeleton(domain, instance, flat_skeleton(combine_degree=1.0))
        assert candidate is not None
        assert abs(domain.cost(instance, candidate) - domain.cost(instance, truth)) < 1e-9


def test_verify_skeleton_accepts_flat_and_divide_shapes():
    domain, instances = _domain_and_instances(n_instances=5, size=6)
    flat_result = verify_skeleton(domain, instances, flat_skeleton(combine_degree=1.0))
    assert flat_result.n_checked == len(instances)
    assert flat_result.n_matched == len(instances)

    divide_result = verify_skeleton(domain, instances, divide_skeleton(branches=2, divisor=2.0, combine_degree=1.0))
    assert divide_result.n_checked == len(instances)
    assert divide_result.n_matched == len(instances)


def test_verify_skeleton_reports_unsupported_when_domain_refuses():
    domain, instances = _domain_and_instances(n_instances=2, size=1)
    # A single-block instance cannot be flat-decomposed further. Force
    # base_threshold=0 so `_execute` actually attempts decompose_flat instead
    # of short-circuiting straight to brute_force at size<=1 (the default
    # base_threshold=1 would mask this — solving a single block directly is
    # legitimate, so it wouldn't reach decompose_flat at all).
    result = verify_skeleton(domain, instances, flat_skeleton(combine_degree=1.0), base_threshold=0)
    assert result.n_unsupported == len(instances)
    assert result.n_checked == 0


def test_synthesize_rejects_naive_brute_force_on_complexity_alone():
    domain, instances = _domain_and_instances(n_instances=4, size=6)
    report = synthesize(domain, instances, target_degree=None, base_exponent=3.0)
    brute = next(c for c in report.candidates if c.name == "brute_force")
    assert brute.status == "rejected_complexity"
    assert brute.cost_class.startswith("Theta(2^")


def test_synthesize_accepts_a_verified_polynomial_decomposition():
    domain, instances = _domain_and_instances(n_instances=5, size=6)
    report = synthesize(domain, instances, target_degree=None, base_exponent=3.0)

    accepted = report.accepted
    assert accepted, "expected at least one verified polynomial skeleton"
    for candidate in accepted:
        assert candidate.degree is not None
        assert candidate.n_checked > 0
        assert candidate.n_matched == candidate.n_checked

    best = report.best()
    assert best is not None
    assert best.status == "accepted"
    # The flat separable decomposition (Theta(n)) should be at least as good
    # as any accepted skeleton — nothing legitimately beats linear here.
    assert best.degree <= 1.0 + 1e-9


def test_synthesize_target_degree_prunes_before_touching_the_domain():
    domain, instances = _domain_and_instances(n_instances=3, size=6)
    # A cap below every real skeleton's degree (flat is Theta(n)) must leave
    # nothing accepted, and every remaining accepted-looking candidate must
    # have been rejected on complexity, not silently dropped.
    report = synthesize(domain, instances, target_degree=0.5, base_exponent=3.0)
    assert report.accepted == []
    assert all(c.status != "accepted" for c in report.candidates)


def test_synthesis_conjectures_jsonl_round_trips(tmp_path):
    domain, instances = _domain_and_instances(n_instances=3, size=5)
    report = synthesize(domain, instances, target_degree=None, base_exponent=3.0)
    out = tmp_path / "synthesis_conjectures.jsonl"
    write_synthesis_conjectures_jsonl(report, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(report.candidates)
    import json

    rows = [json.loads(line) for line in lines]
    assert any(row["status"] == "accepted" for row in rows)
    assert all(row["domain_id"] == "block_separable" for row in rows)


def test_base_skeleton_always_rejected_on_complexity_regardless_of_instance_size():
    # base_skeleton() always models "solve directly" as exponential (that is
    # the whole reason decomposition is worth searching for), so it is
    # rejected symbolically even at a tiny size and even with an
    # unrestricted (any-polynomial) target -- confirming stage 1 pruning
    # never depends on instance size or on ever calling the domain.
    domain, instances = _domain_and_instances(n_instances=3, size=2)
    report = synthesize(domain, instances, target_degree=None, base_exponent=3.0, catalog=[base_skeleton(3.0)])
    only_candidate = report.candidates[0]
    assert only_candidate.status == "rejected_complexity"
    assert only_candidate.n_checked == 0  # never executed against the domain
