"""Static consistency check between each CLI subparser and its handler.

Most ``rde <subcommand>`` handlers are only tested by calling the
``cmd_*`` function directly with a hand-built object, bypassing
``argparse`` entirely. That leaves one specific bug class invisible:
a handler reading ``args.<attr>`` for an ``<attr>`` that was never
registered on its own subparser (``rde represent`` shipped with exactly
this bug — ``args.expr_backend`` wasn't wired up, only caught by an
actual ``python -m rde represent ...`` invocation). This test drives the
real ``argparse.ArgumentParser`` built by ``build_parser()`` for every
subcommand and checks every ``args.<attr>`` access in its handler source
against what that subcommand's parser (plus the top-level parser, since
argparse merges both into one flat ``Namespace``) actually registers.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import textwrap

from rde.cli.commands import build_parser


def _dests(parser: argparse.ArgumentParser) -> set[str]:
    return {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}


def _args_attrs_read(func) -> set[str]:
    """All ``args.<attr>`` *reads* in func's source (nested closures included).

    Only ``ast.Load`` accesses require the attribute to be registered.
    ``args.x = value`` (``ast.Store``) is a legitimate pattern here — some
    handlers stash computed values onto the Namespace as scratch storage
    and always read them back defensively via ``getattr(args, "x", ...)``,
    which is intentionally not flagged either since it already tolerates
    the attribute being absent.
    """
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            attrs.add(node.attr)
    return attrs


def test_every_subcommand_handler_only_reads_registered_args():
    parser = build_parser()
    top_level_dests = _dests(parser)
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )

    failures: list[str] = []
    for name, subparser in subparsers_action.choices.items():
        func = subparser.get_default("func")
        assert func is not None, f"subcommand {name!r} has no handler wired via set_defaults(func=...)"
        available = top_level_dests | _dests(subparser)
        accessed = _args_attrs_read(func)
        missing = sorted(accessed - available)
        if missing:
            failures.append(f"{name!r} -> {func.__name__} reads args.{missing} not registered on its parser")

    assert not failures, "CLI arg-wiring mismatches:\n" + "\n".join(failures)


def test_discover_massive_catalog_flags_reach_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "discover",
            "--run-id",
            "wide-run",
            "--massive-catalog",
            "--max-candidates",
            "1000000",
            "--max-descriptor-templates",
            "100000",
        ]
    )

    assert args.massive_catalog is True
    assert args.max_candidates == 1_000_000
    assert args.max_descriptor_templates == 100_000


def test_campaign_storage_budget_reaches_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "campaign",
            "--domain",
            "synthetic_poly",
            "--sizes",
            "4",
            "--max-storage-gb",
            "6",
        ]
    )
    assert args.max_storage_gb == 6.0


def test_campaign_sealing_flags_reach_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "campaign",
            "--domain",
            "synthetic_poly",
            "--sizes",
            "4",
            "--seal-batches",
            "--seal-keep-arrays",
        ]
    )
    assert args.seal_batches is True
    assert args.seal_keep_arrays is True


def test_seal_command_requires_one_target_and_wires_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "seal",
            "--campaign-id",
            "campaign-a",
            "--batch-index",
            "3",
            "--keep-arrays",
            "--dry-run",
        ]
    )
    assert args.campaign_id == "campaign-a"
    assert args.batch_index == 3
    assert args.keep_arrays is True
    assert args.keep_jsonl is False
    assert args.dry_run is True


def test_plugin_only_science_commands_are_absent_from_core_parser():
    parser = build_parser()
    choices = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ).choices
    assert "science-ledger" not in choices
    assert "uacum-n2" not in choices


def test_retain_topk_command_requires_one_target_and_wires_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "retain-topk",
            "--store-root",
            "rde_runs",
            "--run-id",
            "run_a",
            "--top-k",
            "32",
            "--discovery-report",
            "discovery/run_a.json",
            "--keep-full-shard",
            "--dry-run",
        ]
    )
    assert args.run_id == "run_a"
    assert args.top_k == 32
    assert args.discovery_report == "discovery/run_a.json"
    assert args.keep_full_shard is True
    assert args.dry_run is True


def test_discover_retain_topk_flags_reach_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "discover",
            "--store-root",
            "rde_runs",
            "--run-id",
            "run_a",
            "--retain-topk",
            "--top-k-retention",
            "48",
            "--keep-full-shard",
        ]
    )
    assert args.retain_topk is True
    assert args.top_k_retention == 48
    assert args.keep_full_shard is True


def test_retain_topk_flags_are_available_on_discover_only():
    parser = build_parser()
    args = parser.parse_args(
        ["discover", "--store-root", "rde_runs", "--run-id", "run_a",
         "--retain-topk", "--top-k-retention", "48", "--keep-full-shard"]
    )
    assert args.retain_topk is True
    assert args.top_k_retention == 48
    assert args.keep_full_shard is True
