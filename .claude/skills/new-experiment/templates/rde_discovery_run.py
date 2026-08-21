#!/usr/bin/env python3
"""EXP-NNN: <one-line description>.

RDE discovery experiment template. The gate and the full discovery loop are
already wired: do NOT remove them. Deleting `ExperimentGate` calls or narrowing
`run_discovery` turns this into a screen, and
`tests/rde/test_experiment_receipts.py` will fail the suite if `results.md`
claims a verdict without a valid `receipt.json`.

See src/rde/docs/experiment-playbook.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rde.analyze.outcome import assess_outcome, decisive_criteria_for
from rde.analyze.query import correlate_with_target, flatten_features
from rde.backends import format_dev_machine_profile, resolve_compute_backend
from rde.core.domain_contract import domain_contract
from rde.discovery.loop import run_discovery, write_discovery_report
from rde.experiment import (
    ExperimentGate,
    prepare_leak_clean_discovery,
    run_experiment_main,
)
from rde.experiment.stages import (
    log_machine_profile,
    record_discovery_stage,
    record_gated_outcome,
    record_population_run,
)
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.runtime.progress import (
    default_discovery_progress,
    default_experiment_progress,
)

DOMAIN_ID = "<your_domain_id>"          # must have a DomainContract
TARGET = "metric.<your_target>"         # contract primary_target
DEFAULT_SIZES = (6, 8, 10)              # >= 3 sizes for cross-N


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    p.add_argument("--n-per-size", type=int, default=200)   # >= 50 enforced
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="auto")
    p.add_argument("--store-root", type=Path, default=ROOT / "rde_runs")
    p.add_argument("--run-prefix", default="expNNN")
    p.add_argument("--max-expr-candidates", type=int, default=20_000)
    p.add_argument("--no-pysr", action="store_true")
    p.add_argument("--no-operon", action="store_true")
    args = p.parse_args()

    exp_dir = Path(__file__).resolve().parent
    run_dir = exp_dir / "runs"
    run_dir.mkdir(exist_ok=True)
    log_machine_profile(format_dev_machine_profile())

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    backend = resolve_compute_backend(args.backend)

    # Gate: raises before any compute if the plan cannot support a discovery.
    gate = ExperimentGate(experiment_dir=exp_dir, domain_id=DOMAIN_ID, target=TARGET)
    gate.check_plan(sizes=sizes, n_per_size=args.n_per_size)

    # 1. Population — one independent random instance per row, per size.
    all_rows: list[dict] = []
    run_ids: list[str] = []
    campaign_progress = default_experiment_progress(args.run_prefix, len(sizes))
    for size_index, size in enumerate(sizes, start=1):
        run_id = f"{args.run_prefix}_N{size}_s{args.seed}"
        t0 = perf_counter()
        size_progress = campaign_progress.begin_size(size, size_index, run_id)
        try:
            run_pipeline(
                RunConfig(
                    domain_id=DOMAIN_ID,
                    n_instances=args.n_per_size,
                    size=size,
                    seed=args.seed + size,
                    indices=[0],
                    store_root=args.store_root,
                    run_id=run_id,
                    compute_backend=backend,
                ),
                progress=size_progress,
            )
            rows = flatten_features(run_id, args.store_root)
        except BaseException:
            campaign_progress.abort()
            raise
        all_rows.extend(rows)
        run_ids.append(run_id)
        summary = {
            "stage": "population_run",
            "run_id": run_id,
            "size": size,
            "rows": len(rows),
            "elapsed_s": round(perf_counter() - t0, 3),
        }
        record_population_run(run_dir, summary)
        campaign_progress.end_size(
            feature_rows=len(rows),
            elapsed_s=perf_counter() - t0,
        )

    # 2. Population variety (a reused instance fails here).
    gate.check_population(all_rows)

    # 3. Cross-N merge + leak-clean discovery dataset.
    clean_run, raw_rows, clean_rows = prepare_leak_clean_discovery(
        args.store_root,
        run_ids,
        f"{args.run_prefix}_merged_s{args.seed}",
        f"{args.run_prefix}_clean_s{args.seed}",
        target_metric=TARGET,
    )
    best = lambda rs: max(  # noqa: E731
        (abs(h["pearson_r"]) for h in correlate_with_target(rs, TARGET, min_abs_r=0.0)),
        default=0.0,
    )
    gate.check_leak_audit(raw_best_abs_r=best(raw_rows), clean_best_abs_r=best(clean_rows))

    # 4. FULL discovery loop. Do not narrow this without recording why.
    discovery_progress = default_discovery_progress()

    def _on_stage(name: str) -> None:
        discovery_progress.on_stage(name)
        record_discovery_stage(run_dir, name)

    report = run_discovery(
        clean_run,
        args.store_root,
        target=TARGET,
        max_expr_candidates=args.max_expr_candidates,
        use_pysr=not args.no_pysr,
        use_operon=not args.no_operon,
        use_native_gp=True,
        on_stage=_on_stage,
        on_progress=discovery_progress.on_progress,
    )
    gate.record_discovery_report(report)
    gate.check_discovery_report(report)
    write_discovery_report(report, run_dir / "discovery_report.json")

    # 5. Gated outcome against the pre-registered rule.
    assessment = assess_outcome(
        clean_rows,
        TARGET,
        metric_candidates=report.metric_candidates,
        latent=report.latent,
        phase6=report.phase6,
        leak_audit_summary=report.leak_audit_summary,
        domain_contract=domain_contract(DOMAIN_ID),
    )
    verdict = (
        "HIDDEN_CLASS" if assessment.grade >= 2
        else "SIGNAL" if assessment.grade >= 1
        else "NULL" if assessment.g0_met
        else "WEAK_SUBTHRESHOLD"
    )
    record_gated_outcome(verdict=verdict, grade=assessment.grade)

    # 6. Receipt — written only if every gate passed. `decisive_criteria` names
    # the criteria the decision rule reads, so the gate can refuse a verdict
    # resting on a criterion that never computed (NaN).
    gate.finalize(
        verdict=verdict,
        grade=assessment.grade,
        criteria=dict(assessment.criteria),
        decisive_criteria=decisive_criteria_for(assessment),
    )
    return 0


if __name__ == "__main__":
    _exp_dir = Path(__file__).resolve().parent
    raise SystemExit(run_experiment_main(_exp_dir, main))
