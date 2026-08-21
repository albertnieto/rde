"""Shared CLI helpers."""

from __future__ import annotations

import argparse

from rde.backends import available_backends, default_backend
from rde.io.console import RdeConsole, get_console, resolve_plain


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def add_perf_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default=default_backend(),
        choices=available_backends(),
        help="Compute backend: auto, mlx, numpy; torch_* opt-in.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument(
        "--compute-batch-size",
        type=int,
        default=None,
        help="Fused QUBO batch size for workers=1",
    )
    parser.add_argument("--generator", default=None, help="Instance generator id (optional)")


def console(args: argparse.Namespace) -> RdeConsole:
    return get_console()


def is_plain(args: argparse.Namespace) -> bool:
    return resolve_plain(flag=getattr(args, "plain", False))


def add_store_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store-root", default="rde_runs", help="Store root directory")
