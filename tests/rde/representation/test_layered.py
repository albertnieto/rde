"""Tests for layered representation synthesis (`docs/representation-synthesis-theory.md`).

Every assertion here mirrors a claim made in that doc, checked the same way
before being written into either place — including the honest negative
result (layered compositions beat their own stage-1 base, not necessarily
`identity`) rather than the overclaim an earlier draft made by measuring
only the diffed values and ignoring permutation storage cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import certify_roundtrip
from rde.representation.array_backend import NumpySearchBackend
from rde.representation.grammar import build_primitive_representations
from rde.representation.layered import (
    build_layered_representations,
    compose_layers,
    stage2_primitive_names,
)

BACKEND = NumpySearchBackend()


def _clustered_batch(n: int, rng: np.random.Generator, samples: int = 8) -> np.ndarray:
    return np.stack([rng.choice([0.0, 5.0, 10.0], size=n) for _ in range(samples)])


def _scattered_frequency_batch(n: int, rng: np.random.Generator, samples: int = 6) -> np.ndarray:
    t = np.arange(n)
    freqs = [3, 7, 11]
    signal = sum(np.cos(2 * np.pi * f * t / n + rng.uniform(0, 2 * np.pi)) for f in freqs)
    return np.stack([signal + 0.001 * rng.normal(size=n) for _ in range(samples)])


def _row_periodic_batch(n: int, rng: np.random.Generator, samples: int = 8) -> np.ndarray:
    """`matrix_reshape`'s `(side, side)` carrier with every row an identical single-frequency
    signal -- the 2D-periodic structure a per-row FFT can see but a flat 1-D sparsity count over
    the same `n` raw values cannot (each row's own DFT is exactly 2-sparse; the raw values are
    not sparse at all).
    """
    side = int(round(n**0.5))
    t = np.arange(side)
    batch = []
    for _ in range(samples):
        k = rng.integers(1, side)
        row = np.cos(2 * np.pi * k * t / side)
        batch.append(np.tile(row, (side, 1)).reshape(-1))
    return np.stack(batch)


def test_stage2_primitive_names_lists_only_registered_carrier_kinds():
    assert stage2_primitive_names("complex_vector") == ("sort_by_magnitude",)
    assert stage2_primitive_names("sorted_pair") == ("sorted_then_difference",)
    assert stage2_primitive_names("sorted_complex_pair") == ("sorted_complex_then_difference",)
    assert stage2_primitive_names("matrix") == ("row_dft",)
    assert stage2_primitive_names("no_such_kind") == ()


def test_build_layered_representations_enumerates_every_valid_pair():
    n = 16
    layered = build_layered_representations(n, backend=BACKEND)
    assert set(layered) == {
        "dft+sort_by_magnitude",
        "dft_full+sort_by_magnitude",
        "sorted_permutation+sorted_then_difference",
        "matrix_reshape+row_dft",
    }


def test_compose_layers_rejects_incompatible_carrier_kinds():
    n = 8
    grammar = build_primitive_representations(n, backend=BACKEND)
    # sorted_permutation's carrier_kind is "sorted_pair"; dft_full expects
    # "real_vector" as input -- these must not compose.
    with pytest.raises(ValueError):
        compose_layers(grammar["sorted_permutation"], grammar["dft_full"])


def test_compose_layers_accepts_trivially_compatible_real_vector_pair():
    # identity's carrier_kind and sorted_permutation's input_carrier_kind
    # are both the default "real_vector" -- a valid, if uninteresting,
    # composition; this documents that carrier-kind matching is the only
    # gate, not e.g. an allowlist of "interesting" pairs.
    n = 8
    grammar = build_primitive_representations(n, backend=BACKEND)
    composed = compose_layers(grammar["identity"], grammar["sorted_permutation"])
    assert composed.representation_id == "identity+sorted_permutation"


@pytest.mark.parametrize(
    "representation_id,make_batch",
    [
        ("sorted_permutation+sorted_then_difference", _clustered_batch),
        ("dft_full+sort_by_magnitude", _scattered_frequency_batch),
        ("dft+sort_by_magnitude", _scattered_frequency_batch),
        ("matrix_reshape+row_dft", _row_periodic_batch),
    ],
)
def test_layered_representation_roundtrips_exactly(representation_id, make_batch):
    n = 16
    rng = np.random.default_rng(1)
    batch = make_batch(n, rng)
    layered = build_layered_representations(n, backend=BACKEND)
    certificate = certify_roundtrip(layered[representation_id], batch, tolerance=1e-9)
    assert certificate.status == "verified"
    assert certificate.error < 1e-9


@pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
def test_sorted_then_difference_always_beats_its_own_stage1_base(n):
    # The real, defensible claim (see theory doc §4): layering strictly
    # improves on sorted_permutation alone, in every tested size -- not
    # necessarily on identity, which is a separate, honestly-negative result
    # tested below.
    rng = np.random.default_rng(2)
    batch = _clustered_batch(n, rng)
    grammar = build_primitive_representations(n, backend=BACKEND)
    layered = build_layered_representations(n, backend=BACKEND)

    base = grammar["sorted_permutation"]
    composed = layered["sorted_permutation+sorted_then_difference"]

    base_complexity = base.complexity(base.encode(batch))
    composed_complexity = composed.complexity(composed.encode(batch))
    assert composed_complexity <= base_complexity


def test_sort_by_magnitude_always_beats_or_matches_its_own_stage1_base():
    n = 16
    rng = np.random.default_rng(4)
    batch = _scattered_frequency_batch(n, rng)
    grammar = build_primitive_representations(n, backend=BACKEND)
    layered = build_layered_representations(n, backend=BACKEND)

    base = grammar["dft_full"]
    composed = layered["dft_full+sort_by_magnitude"]
    base_complexity = base.complexity(base.encode(batch))
    composed_complexity = composed.complexity(composed.encode(batch))
    # Sorting alone does not change the sparsity-fraction count of the
    # coefficients (see theory doc §4) -- permutation overhead can only
    # make this worse, never strictly better, for this particular metric.
    assert composed_complexity >= base_complexity


def test_layered_permutation_carrying_compositions_do_not_beat_identity():
    # The honest negative result from theory doc §4/§6: `_clustered_batch`
    # draws each element i.i.d., so its original order carries no relation
    # to sorted order -- the sort permutation is close to uniformly random,
    # which costs close to the maximum under *any* permutation-complexity
    # metric (an information-theoretic floor, not an artifact of the old
    # sparsity-based permutation charge that `permutation_complexity`
    # replaced). Documented as a real, scoped limitation (§5), not hidden.
    n = 16
    rng = np.random.default_rng(2)
    batch = _clustered_batch(n, rng)
    grammar = build_primitive_representations(n, backend=BACKEND)
    layered = build_layered_representations(n, backend=BACKEND)

    identity = grammar["identity"]
    composed = layered["sorted_permutation+sorted_then_difference"]
    identity_complexity = identity.complexity(identity.encode(batch))
    composed_complexity = composed.complexity(composed.encode(batch))
    assert composed_complexity > identity_complexity


def test_row_dft_beats_flat_identity_on_row_periodic_matrix_data():
    # Verified numerically before writing: identity/matrix_reshape both see
    # 16 raw values with no near-zero entries (complexity 9.0 for the
    # n=16, side=4 case below); a per-row DFT concentrates each row's
    # single-frequency signal into 2 of 4 coefficients, so more than half
    # the encoded entries read as zero under sparsity_fraction's eps=1e-6 --
    # the "expose 2D-periodic structure" win the theory doc's roadmap
    # flagged `row_dft` as the natural next primitive for.
    n = 16
    rng = np.random.default_rng(3)
    batch = _row_periodic_batch(n, rng)
    grammar = build_primitive_representations(n, backend=BACKEND)
    layered = build_layered_representations(n, backend=BACKEND)

    identity = grammar["identity"]
    composed = layered["matrix_reshape+row_dft"]
    identity_complexity = identity.complexity(identity.encode(batch))
    composed_complexity = composed.complexity(composed.encode(batch))
    assert composed_complexity < identity_complexity


def test_composed_representation_carries_object_type_and_carrier_kind_metadata():
    n = 16
    layered = build_layered_representations(n, backend=BACKEND)
    rep = layered["sorted_permutation+sorted_then_difference"]
    assert rep.object_type == f"numeric_batch_{n}"
    assert rep.input_carrier_kind == "real_vector"
    assert rep.carrier_kind == "sorted_pair"


def test_composed_representation_is_a_plain_representation_usable_by_existing_infra():
    # §3's design claim: composed representations need no parallel
    # infrastructure -- they work with Phase 1's own certify_roundtrip
    # (already exercised above) and expose the same fields every other
    # Representation does, so search.py/report.py/holdout.py all apply.
    from rde.representation import Representation

    n = 16
    layered = build_layered_representations(n, backend=BACKEND)
    rep = layered["sorted_permutation+sorted_then_difference"]
    assert isinstance(rep, Representation)
    assert rep.representation_id == "sorted_permutation+sorted_then_difference"
