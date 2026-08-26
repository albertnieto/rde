"""Gated Mode-2 campaign for approximate-label abelian HSP recovery.

Run from the repository root with the project's virtual environment.  The
campaign is restartable: each instance's full catalog score is appended to a
durable JSONL checkpoint before the next instance starts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rde.experiment import ExperimentGate
from rde.experiment.runner import run_experiment_main
from rde.io.events import append_jsonl, log_progress, log_stage_event
from rde.recovery.campaign import (
    CONFIRMATORY_SIZES,
    DISCOVERY_SIZES,
    PIPELINE_MIN_RECALL,
    append_instance_record,
    iter_jsonl,
    load_done_keys,
    rows_from_instance_records,
)
from rde.recovery.programs import PIPELINE_PROTOCOL_BY_FAMILY, TEXTBOOK_PROTOCOL_IDS
from rde.recovery.search import evaluate_protocols
from rde.recovery.search_space import enumerate_recovery_chains
from rde_domains.hsp_functions.functions import make_instance
from rde_domains.hsp_functions.recovery import HspFunctionRecovery


EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_DIR = EXPERIMENT_DIR / "runs"
RECORDS_PATH = RUN_DIR / "instance_records.jsonl"
MANIFEST_PATH = RUN_DIR / "manifest.json"
EVENTS_PATH = RUN_DIR / "events.jsonl"

FAMILIES = (
    "simon",
    "shor_cyclic",
    "dihedral_kuperberg",
    "approximate_cyclic_period",
    "approximate_cyclic_period_alt",
    "approximate_xor_shift",
    "generic_random_control",
)
DISCOVERY_FAMILY = "approximate_cyclic_period"
HELD_OUT_FAMILY = "approximate_cyclic_period_alt"
CONTROL_FAMILY = "generic_random_control"
CONTROL_MIN_SPECIFICITY = 0.95


def _rate(rows: Sequence[Any], protocol_id: str, family: str, size: int) -> float:
    selected = [
        row
        for row in rows
        if row.protocol_id == protocol_id and row.family == family and int(row.size) == int(size)
    ]
    if not selected:
        return float("nan")
    return float(sum(bool(row.matched) for row in selected)) / float(len(selected))


def _holds(
    rows: Sequence[Any], protocol_id: str, family: str, sizes: Sequence[int], minimum: float
) -> bool:
    return all(
        (rate := _rate(rows, protocol_id, family, size)) == rate and rate >= minimum
        for size in sizes
    )


def _control_specificity(rows: Sequence[Any], protocol_id: str, sizes: Sequence[int]) -> float:
    rates = [_rate(rows, protocol_id, CONTROL_FAMILY, size) for size in sizes]
    finite = [rate for rate in rates if rate == rate]
    return min(finite) if finite else float("nan")


def _records_for_gate(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"size": record["size"], "family": record["family"], "seed": record["seed"]}
        for record in records
    ]


def _write_manifest(*, n_per_family: int, protocol_ids: Sequence[str]) -> None:
    manifest = {
        "experiment": EXPERIMENT_DIR.name,
        "n_per_family": n_per_family,
        "discovery_sizes": list(DISCOVERY_SIZES),
        "confirmatory_sizes": list(CONFIRMATORY_SIZES),
        "families": list(FAMILIES),
        "protocol_ids": list(protocol_ids),
    }
    if MANIFEST_PATH.exists():
        prior = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if prior != manifest:
            raise RuntimeError(
                "existing checkpoint uses a different frozen manifest; start a new experiment directory"
            )
        return
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _record_instance(
    domain: HspFunctionRecovery, instance: Any, catalog: Sequence[Any], tape_seed: int
) -> dict[str, Any]:
    report = evaluate_protocols(domain, [instance], catalog, rng=np.random.default_rng(tape_seed))
    results = {
        row.protocol_id: {"matched": bool(row.matched), "recovered": row.recovered}
        for row in report.rows
    }
    first = report.rows[0]
    return {
        "size": int(instance.n_bits),
        "family": instance.family,
        "seed": int(instance.seed),
        "queries_used": int(first.queries_used),
        "planted": first.planted,
        "results": results,
    }


def _write_results(
    receipt: dict[str, Any], selected: Sequence[str], heldout: Sequence[str]
) -> None:
    text = "\n".join(
        [
            "# EXP-001 results",
            "",
            f"Outcome: **{receipt['verdict']}** (G{receipt['grade']}).",
            "",
            f"Discovery-selected protocols: {', '.join(selected) if selected else 'none'}.",
            f"Held-out survivors: {', '.join(heldout) if heldout else 'none'}.",
            "",
            "See `receipt.json`, `runs/manifest.json`, and `runs/instance_records.jsonl` for the frozen design and complete per-instance record.",
        ]
    )
    (EXPERIMENT_DIR / "results.md").write_text(text + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-per-family",
        type=int,
        default=100,
        help="registered population size per family and n_bits",
    )
    args = parser.parse_args(argv)
    if args.n_per_family < 50:
        parser.error("--n-per-family must be at least 50 to satisfy the recovery gate")

    all_sizes = tuple(dict.fromkeys((*DISCOVERY_SIZES, *CONFIRMATORY_SIZES)))
    catalog = enumerate_recovery_chains(max_depth=1)
    protocol_ids = [protocol.protocol_id for protocol in catalog]
    _write_manifest(n_per_family=args.n_per_family, protocol_ids=protocol_ids)

    gate = ExperimentGate(
        experiment_dir=EXPERIMENT_DIR,
        domain_id="hsp_functions",
        target="recovery.approximate_cyclic_period",
        gate_kind="recovery",
    )
    gate.check_plan(sizes=all_sizes, n_per_size=args.n_per_family)
    gate.check_extractor_isolation(extract_sees_planted=False, extract_sees_family=False)

    done = load_done_keys(RECORDS_PATH)
    total = len(all_sizes) * len(FAMILIES) * args.n_per_family
    started = time.monotonic()
    completed = len(done)
    domain = HspFunctionRecovery()
    log_stage_event(
        {"stage": "recovery_start", "total_instances": total, "catalog_size": len(catalog)},
        jsonl_path=EVENTS_PATH,
    )
    for size in all_sizes:
        for family_index, family in enumerate(FAMILIES):
            for seed in range(args.n_per_family):
                key = (int(size), family, seed)
                if key in done:
                    continue
                instance = make_instance(family, n_bits=int(size), seed=seed)
                # Fixed independently of loop order so a resumed campaign has
                # the exact same tape for each instance.
                tape_seed = 1_000_003 * int(size) + 10_007 * family_index + seed
                record = _record_instance(domain, instance, catalog, tape_seed)
                append_instance_record(RECORDS_PATH, record)
                completed += 1
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                eta = (total - completed) / rate if rate else float("inf")
                log_progress(
                    f"recovery {completed}/{total} n={size} family={family} seed={seed} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                )

    records = iter_jsonl(RECORDS_PATH)
    gate.check_recovery_population(_records_for_gate(records))
    rows = rows_from_instance_records(records)
    discovery_rows = [row for row in rows if int(row.seed) % 2 == 0]
    confirm_rows = [row for row in rows if int(row.seed) % 2 == 1]

    pipeline_ok = all(
        _holds(confirm_rows, protocol_id, family, CONFIRMATORY_SIZES, PIPELINE_MIN_RECALL)
        for family, protocol_id in PIPELINE_PROTOCOL_BY_FAMILY.items()
    )
    gate.record_phase(
        "pipeline_check",
        work_units=len(PIPELINE_PROTOCOL_BY_FAMILY) * len(CONFIRMATORY_SIZES),
        work_kind="textbook_protocol_rates",
        detail={"ok": pipeline_ok},
    )

    selected = [
        protocol_id
        for protocol_id in protocol_ids
        if protocol_id not in TEXTBOOK_PROTOCOL_IDS
        and _holds(
            discovery_rows, protocol_id, DISCOVERY_FAMILY, DISCOVERY_SIZES, PIPELINE_MIN_RECALL
        )
        and _control_specificity(discovery_rows, protocol_id, DISCOVERY_SIZES)
        >= CONTROL_MIN_SPECIFICITY
    ]
    gate.record_phase(
        "protocol_search",
        work_units=len(protocol_ids) * len(discovery_rows),
        work_kind="frozen_catalog_scores",
        detail={"selected": len(selected), "catalog_size": len(protocol_ids)},
    )

    heldout = [
        protocol_id
        for protocol_id in selected
        if _holds(
            confirm_rows, protocol_id, HELD_OUT_FAMILY, CONFIRMATORY_SIZES, PIPELINE_MIN_RECALL
        )
        and _control_specificity(confirm_rows, protocol_id, CONFIRMATORY_SIZES)
        >= CONTROL_MIN_SPECIFICITY
    ]
    gate.record_phase(
        "heldout_confirm",
        work_units=len(selected) * len(confirm_rows),
        work_kind="preselected_protocol_scores",
        detail={"survivors": len(heldout)},
    )

    criteria = {
        "pipeline_ok": pipeline_ok,
        "n_discovery_selected": len(selected),
        "n_heldout_survivors": len(heldout),
        "discovery_control_specificity": max(
            (
                _control_specificity(discovery_rows, protocol_id, DISCOVERY_SIZES)
                for protocol_id in selected
            ),
            default=0.0,
        ),
        "confirmatory_control_specificity": max(
            (
                _control_specificity(confirm_rows, protocol_id, CONFIRMATORY_SIZES)
                for protocol_id in heldout
            ),
            default=0.0,
        ),
    }
    signal = pipeline_ok and bool(selected) and bool(heldout)
    receipt = gate.finalize(
        verdict="SIGNAL" if signal else "NULL",
        grade=1 if signal else 0,
        criteria=criteria,
        decisive_criteria=("pipeline_ok", "n_discovery_selected", "n_heldout_survivors"),
        extra={"selected_protocols": selected, "heldout_protocols": heldout},
    )
    _write_results(receipt, selected, heldout)
    append_jsonl(
        EVENTS_PATH, {"stage": "complete", "verdict": receipt["verdict"], "grade": receipt["grade"]}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_experiment_main(EXPERIMENT_DIR, main, run_dir=RUN_DIR))
