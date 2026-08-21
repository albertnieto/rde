"""Tests for shared CLI helpers."""

from __future__ import annotations

import argparse

from rde.cli.common import (
    add_perf_flags,
    add_store_root,
    console,
    is_plain,
    parse_int_list,
)


def test_parse_int_list():
    assert parse_int_list("1, 2, 4") == [1, 2, 4]


def test_perf_and_store_flags():
    parser = argparse.ArgumentParser()
    add_perf_flags(parser)
    add_store_root(parser)
    args = parser.parse_args(
        ["--backend", "numpy", "--workers", "2", "--store-root", "/tmp/rde"]
    )
    assert args.backend == "numpy"
    assert args.workers == 2
    assert args.store_root == "/tmp/rde"
    assert console(args) is not None
    assert is_plain(argparse.Namespace(plain=True)) is True
