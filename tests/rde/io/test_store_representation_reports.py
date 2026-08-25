"""Tests for `Store.append_representation_report` / `iter_representation_reports`.

A separate, plain-append JSONL file kind — not part of the resumable
campaign completion index `append_features`/`append_instance_features`
maintain, and not validated against `rde.core.schema` (there is no schema
shape for representation-search results; see `report.py`'s docstring for
why forcing one would be a bad fit).
"""

from __future__ import annotations

from pathlib import Path

from rde.io.store import Store


def test_append_and_read_representation_reports_round_trip(tmp_path: Path):
    store = Store(tmp_path)
    row = {"kind": "representation_search_report", "candidates": [{"representation_id": "identity"}]}
    store.append_representation_report("run001", row)
    store.flush("run001")

    read_back = store.read_representation_reports("run001")
    assert read_back == [row]


def test_append_representation_report_accumulates_multiple_rows(tmp_path: Path):
    store = Store(tmp_path)
    for i in range(3):
        store.append_representation_report("run002", {"kind": "row", "i": i})
    store.flush("run002")

    rows = list(store.iter_representation_reports("run002"))
    assert [r["i"] for r in rows] == [0, 1, 2]


def test_representation_reports_are_isolated_per_run(tmp_path: Path):
    store = Store(tmp_path)
    store.append_representation_report("run_a", {"kind": "a"})
    store.append_representation_report("run_b", {"kind": "b"})
    store.flush("run_a")
    store.flush("run_b")

    assert store.read_representation_reports("run_a") == [{"kind": "a"}]
    assert store.read_representation_reports("run_b") == [{"kind": "b"}]


def test_representation_reports_file_lives_alongside_other_run_artifacts(tmp_path: Path):
    store = Store(tmp_path)
    store.append_representation_report("run003", {"kind": "row"})
    store.flush("run003")

    assert (store.run_dir("run003") / "representation_reports.jsonl").exists()
