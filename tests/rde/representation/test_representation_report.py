"""Tests for durable JSON reports of representation-search results."""

from __future__ import annotations

import json

import numpy as np

from rde.io.store import Store
from rde.representation import (
    build_primitive_representations,
    diagonalization_report_payload,
    rank_by_diagonalization,
    rank_representations,
    search_report_payload,
    write_diagonalization_report,
    write_diagonalization_report_to_store,
    write_search_report,
    write_search_report_to_store,
)
from rde.representation.array_backend import NumpySearchBackend

BACKEND = NumpySearchBackend()


def test_search_report_payload_shape_and_content():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 6))
    ranked = rank_representations(batch, n=6, backend=BACKEND)
    payload = search_report_payload(ranked)
    assert payload["kind"] == "representation_search_report"
    assert len(payload["candidates"]) == len(ranked)
    first = payload["candidates"][0]
    assert first["representation_id"] == ranked[0].representation_id
    assert first["complexity"] == ranked[0].complexity
    assert first["conversion_cost"] == ranked[0].conversion_cost
    assert first["certificate"]["status"] == ranked[0].certificate.status


def test_search_report_payload_is_json_serializable():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 6))
    ranked = rank_representations(batch, n=6, backend=BACKEND)
    payload = search_report_payload(ranked)
    # Must not raise — every value must already be a plain JSON type.
    text = json.dumps(payload)
    assert '"representation_search_report"' in text


def test_write_search_report_creates_file_and_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    batch = rng.normal(size=(4, 6))
    ranked = rank_representations(batch, n=6, backend=BACKEND)
    out_path = tmp_path / "nested" / "search_report.json"
    written = write_search_report(ranked, out_path)
    assert written == out_path
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded == search_report_payload(ranked)


def test_diagonalization_report_payload_excludes_transported_operator_array():
    from scipy.linalg import circulant

    n = 6
    rng = np.random.default_rng(2)
    U = circulant(rng.normal(size=n)).T
    ranked = rank_by_diagonalization(U, n=n, backend=BACKEND)
    payload = diagonalization_report_payload(ranked)
    assert payload["kind"] == "operator_diagonalization_report"
    for candidate in payload["candidates"]:
        assert set(candidate.keys()) == {"representation_id", "off_diagonal_energy"}
    # Must be JSON serializable now that the ndarray field is excluded.
    json.dumps(payload)


def test_write_diagonalization_report_round_trips(tmp_path):
    from scipy.linalg import circulant

    n = 6
    rng = np.random.default_rng(3)
    U = circulant(rng.normal(size=n)).T
    ranked = rank_by_diagonalization(U, n=n, backend=BACKEND)
    out_path = tmp_path / "diag_report.json"
    written = write_diagonalization_report(ranked, out_path)
    loaded = json.loads(written.read_text())
    assert loaded == diagonalization_report_payload(ranked)


def test_write_search_report_on_real_grammar_all_primitives_present(tmp_path):
    n = 9  # perfect square -> matrix_reshape included too
    rng = np.random.default_rng(4)
    batch = rng.normal(size=(3, n))
    ranked = rank_representations(batch, n=n, backend=BACKEND)
    out_path = tmp_path / "full_grammar_report.json"
    write_search_report(ranked, out_path)
    loaded = json.loads(out_path.read_text())
    ids = {c["representation_id"] for c in loaded["candidates"]}
    assert ids == set(build_primitive_representations(n, backend=BACKEND).keys())


def test_search_report_payload_serializes_composed_chain_candidates(tmp_path):
    # `rank_representations`'s chain_max_depth wiring (see search.py §10 of
    # docs/representation-synthesis-theory.md) produces SearchCandidates
    # whose representation_id is a "+"-joined composed chain -- report.py's
    # generic payload builder must handle those with no changes of its own
    # (composed ids need cost.py's additive-over-stages computational_cost,
    # not report.py awareness of composition).
    n = 12
    seg1 = np.linspace(0.0, 5.0, 7)
    seg2 = np.linspace(5.2, 20.0, 5)
    batch_row = np.concatenate([seg1, seg2])
    batch = np.stack([batch_row, batch_row.copy()])
    ranked = rank_representations(batch, n=n, backend=BACKEND, chain_max_depth=3)
    composed_ids = {c.representation_id for c in ranked if "+" in c.representation_id}
    assert composed_ids  # sanity: this scenario actually exercises composed chains

    payload = search_report_payload(ranked)
    json.dumps(payload)  # must not raise
    out_path = tmp_path / "chain_report.json"
    write_search_report(ranked, out_path)
    loaded = json.loads(out_path.read_text())
    assert {c["representation_id"] for c in loaded["candidates"]} == {
        c.representation_id for c in ranked
    }


def test_write_search_report_to_store_appends_the_same_payload(tmp_path):
    rng = np.random.default_rng(5)
    batch = rng.normal(size=(4, 6))
    ranked = rank_representations(batch, n=6, backend=BACKEND)

    store = Store(tmp_path)
    write_search_report_to_store(ranked, store, "run_repr_001")
    store.flush("run_repr_001")

    rows = store.read_representation_reports("run_repr_001")
    assert rows == [search_report_payload(ranked)]


def test_write_diagonalization_report_to_store_appends_the_same_payload(tmp_path):
    from scipy.linalg import circulant

    n = 6
    rng = np.random.default_rng(6)
    U = circulant(rng.normal(size=n)).T
    ranked = rank_by_diagonalization(U, n=n, backend=BACKEND)

    store = Store(tmp_path)
    write_diagonalization_report_to_store(ranked, store, "run_repr_002")
    store.flush("run_repr_002")

    rows = store.read_representation_reports("run_repr_002")
    assert rows == [diagonalization_report_payload(ranked)]


def test_write_search_report_to_store_is_appendable_across_multiple_calls(tmp_path):
    store = Store(tmp_path)
    for seed_offset in range(3):
        batch = np.random.default_rng(7 + seed_offset).normal(size=(3, 5))
        ranked = rank_representations(batch, n=5, backend=BACKEND)
        write_search_report_to_store(ranked, store, "run_repr_003")
    store.flush("run_repr_003")

    rows = store.read_representation_reports("run_repr_003")
    assert len(rows) == 3
