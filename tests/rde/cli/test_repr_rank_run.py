"""Tests for `rde repr-rank-run` — representation search over a real stored run.

Purely additive: builds a real run via `run_pipeline` (the actual
`rde run` machinery, not a hand-built fixture), then confirms
`repr-rank-run` reads it correctly and writes a report through the same
`Store` — never modifying `run`/`campaign`'s own output.
"""

from __future__ import annotations

import json

import pytest

from rde.cli.commands import build_parser
from rde.io.store import Store
from rde.runtime.pipeline import RunConfig, run_pipeline


@pytest.fixture
def hsp_run(tmp_path):
    store_root = tmp_path / "store"
    cfg = RunConfig(
        domain_id="hsp_functions",
        n_instances=6,
        size=8,
        seed=0,
        indices=[0],
        store_root=store_root,
        run_id="testrun",
        save_arrays=True,
    )
    run_pipeline(cfg)
    return store_root


def test_repr_rank_run_reads_real_stored_arrays_and_ranks_them(hsp_run, capsys):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--plain",
            "repr-rank-run",
            "--store-root",
            str(hsp_run),
            "--run-id",
            "testrun",
            "--array-key",
            "diff_profile",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "size=8 (n=8, samples=6)" in out
    assert "identity" in out


def test_repr_rank_run_writes_representation_reports_via_store(hsp_run):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--plain",
            "repr-rank-run",
            "--store-root",
            str(hsp_run),
            "--run-id",
            "testrun",
            "--array-key",
            "diff_profile",
        ]
    )
    args.func(args)

    store = Store(hsp_run)
    reports = store.read_representation_reports("testrun")
    assert len(reports) == 1
    assert reports[0]["kind"] == "representation_search_report"
    ids = {c["representation_id"] for c in reports[0]["candidates"]}
    assert "identity" in ids
    assert "dft_full" in ids


def test_repr_rank_run_does_not_modify_existing_run_artifacts(hsp_run):
    store = Store(hsp_run)
    before = store.read_instance_features("testrun")

    parser = build_parser()
    args = parser.parse_args(
        [
            "--plain",
            "repr-rank-run",
            "--store-root",
            str(hsp_run),
            "--run-id",
            "testrun",
            "--array-key",
            "diff_profile",
        ]
    )
    args.func(args)

    after = store.read_instance_features("testrun")
    assert before == after


def test_repr_rank_run_returns_nonzero_for_unknown_array_key(hsp_run, capsys):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--plain",
            "repr-rank-run",
            "--store-root",
            str(hsp_run),
            "--run-id",
            "testrun",
            "--array-key",
            "not_a_real_key",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 1
    assert "not_a_real_key" in capsys.readouterr().out


def test_repr_rank_run_returns_nonzero_for_unknown_run_id(tmp_path, capsys):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--plain",
            "repr-rank-run",
            "--store-root",
            str(tmp_path / "store"),
            "--run-id",
            "does_not_exist",
            "--array-key",
            "diff_profile",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 1


def test_repr_rank_run_is_registered_in_top_level_help():
    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    assert "repr-rank-run" in subparsers_action.choices
