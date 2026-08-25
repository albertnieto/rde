"""Tests for the `rde repr-rank` CLI command."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rde.cli.commands import _repr_demo_batch, build_parser


def test_repr_demo_batch_shapes():
    assert _repr_demo_batch("random", n=6, samples=4, seed=0).shape == (4, 6)
    assert _repr_demo_batch("periodic", n=6, samples=4, seed=0).shape == (4, 6)
    assert _repr_demo_batch("polynomial", n=6, samples=4, seed=0).shape == (4, 6)


def test_repr_demo_batch_rejects_unknown_pattern():
    with pytest.raises(ValueError):
        _repr_demo_batch("not_a_pattern", n=6, samples=4, seed=0)


def test_repr_demo_batch_is_reproducible_for_fixed_seed():
    a = _repr_demo_batch("random", n=6, samples=4, seed=1)
    b = _repr_demo_batch("random", n=6, samples=4, seed=1)
    assert np.array_equal(a, b)


def test_repr_rank_command_parses_and_runs(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["--plain", "repr-rank", "--n", "6", "--samples", "4", "--pattern", "random"]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "representation ranking" in out
    assert "identity" in out


def test_repr_rank_command_writes_report(tmp_path, capsys):
    parser = build_parser()
    output_path = tmp_path / "report.json"
    args = parser.parse_args(
        [
            "--plain",
            "repr-rank",
            "--n",
            "6",
            "--samples",
            "4",
            "--pattern",
            "polynomial",
            "--output",
            str(output_path),
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["kind"] == "representation_search_report"
    # polynomial pattern should make polynomial_vandermonde the top candidate
    assert payload["candidates"][0]["representation_id"] == "polynomial_vandermonde"


def test_repr_rank_command_loads_input_file(tmp_path, capsys):
    input_path = tmp_path / "batch.npy"
    rng = np.random.default_rng(0)
    np.save(input_path, rng.normal(size=(5, 7)))

    parser = build_parser()
    args = parser.parse_args(["--plain", "repr-rank", "--input", str(input_path)])
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "n                     7" in out
    assert "samples               5" in out


def test_repr_rank_is_registered_in_top_level_help():
    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    assert "repr-rank" in subparsers_action.choices
