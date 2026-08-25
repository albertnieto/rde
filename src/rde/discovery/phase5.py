"""Phase 5 orchestration — symbolic equation search + latent interpretation + promotion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rde.discovery.datasets import load_rows_from_source
from rde.discovery.promote import promote_top_conjectures
from rde.discovery.symbolic import (
    LatentSource,
    SymbolicDiscoveryResult,
    run_symbolic_discovery,
    symbolic_backends,
)
from rde.io.json_util import json_default
from rde.runtime.targets import resolve_target_for_run
from rde.runtime.heartbeat import ProgressCallback


@dataclass
class Phase5Report:
    run_id: str
    target: str
    n_rows: int
    symbolic: dict[str, Any] = field(default_factory=dict)
    latent_interpretations: list[dict[str, Any]] = field(default_factory=list)
    promoted_conjectures: list[dict[str, Any]] = field(default_factory=list)
    backends: dict[str, Any] = field(default_factory=dict)
    g1_met: bool = False

    def to_symbolic_dict(self) -> dict[str, Any]:
        return dict(self.symbolic)

    def to_phase5_dict(self) -> dict[str, Any]:
        return {
            "g1_met": self.g1_met,
            "symbolic": self.symbolic,
            "latent_interpretations": self.latent_interpretations,
            "promoted_conjectures": self.promoted_conjectures,
            "backends": self.backends,
        }


def _symbolic_payload(sym_phase: SymbolicDiscoveryResult) -> dict[str, Any]:
    sym = sym_phase.target_fit
    latent_interps = sym_phase.latent_interpretations
    return {
        "method": sym.method,
        "equation": sym.equation,
        "r_squared": sym.r_squared,
        **sym.metadata,
        "backends": sym_phase.backends,
        "skipped_engines": sym.metadata.get("skipped_engines", []),
        "latent_interpretation": latent_interps[0] if latent_interps else None,
        "latent_interpretations": latent_interps,
    }


def run_phase5(
    run_id: str,
    store_root: Path | str,
    *,
    target: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    feature_columns: list[str] | None = None,
    latent_sources: list[LatentSource] | None = None,
    metric_candidates: list[dict[str, Any]] | None = None,
    programs: list[dict[str, Any]] | None = None,
    top_features: int = 8,
    top_latents: int = 3,
    max_promoted: int = 10,
    use_pysr: bool = True,
    use_operon: bool = True,
    use_physics_templates: bool = True,
    use_native_gp: bool = True,
    use_ai_feynman: bool = False,
    symbolic_min_r_squared: float = 0.5,
    promote_min_r_squared: float = 0.5,
    gp_generations: int = 20,
    gp_population: int = 48,
    pysr_iterations: int = 40,
    pysr_maxsize: int = 20,
    pysr_populations: int = 15,
    operon_generations: int = 50,
    operon_population: int = 300,
    expr_eval_backend: str | None = None,
    require_gpu: bool = False,
    promote: bool = True,
    on_stage: Callable[[str], None] | None = None,
    on_symbolic_stage: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
) -> Phase5Report:
    """Run Phase 5 symbolic discovery on one run ledger."""
    def _stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    store_root = Path(store_root)
    resolved_target = resolve_target_for_run(run_id, store_root, override=target)
    if rows is None:
        rows = load_rows_from_source(run_id=run_id, store_root=store_root)
    try:
        from rde.io.store import Store

        domain_id = Store(store_root).read_manifest(run_id).domain_id
    except (FileNotFoundError, OSError, ValueError, KeyError):
        domain_id = None
    report = Phase5Report(run_id=run_id, target=resolved_target, n_rows=len(rows))
    report.backends = symbolic_backends()
    if not rows:
        return report

    _stage("symbolic regression")
    sym_phase = run_symbolic_discovery(
        rows,
        resolved_target,
        feature_columns=feature_columns,
        latent_sources=latent_sources,
        top_features=top_features,
        top_latents=top_latents,
        use_pysr=use_pysr,
        use_operon=use_operon,
        use_physics_templates=use_physics_templates,
        use_native_gp=use_native_gp,
        use_ai_feynman=use_ai_feynman,
        symbolic_min_r_squared=symbolic_min_r_squared,
        gp_generations=gp_generations,
        gp_population=gp_population,
        pysr_iterations=pysr_iterations,
        pysr_maxsize=pysr_maxsize,
        pysr_populations=pysr_populations,
        operon_generations=operon_generations,
        operon_population=operon_population,
        eval_backend=expr_eval_backend,
        require_gpu=require_gpu,
        on_stage=on_symbolic_stage,
        on_progress=on_progress,
        domain_id=domain_id,
    )
    report.symbolic = _symbolic_payload(sym_phase)
    report.latent_interpretations = sym_phase.latent_interpretations
    report.g1_met = float(report.symbolic.get("r_squared", 0.0)) >= 0.75

    _stage("promote conjectures")
    if promote:
        report.promoted_conjectures = promote_top_conjectures(
            target=resolved_target,
            metric_candidates=metric_candidates or [],
            symbolic=report.symbolic,
            latent_interpretations=report.latent_interpretations,
            programs=programs or [],
            max_results=max_promoted,
            min_r_squared=promote_min_r_squared,
            leak_audit=False,
        )
    return report


def write_phase5_report(report: Phase5Report, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": report.run_id,
        "target": report.target,
        "n_rows": report.n_rows,
        "phase5": report.to_phase5_dict(),
        "symbolic": report.symbolic,
        "promoted_conjectures": report.promoted_conjectures,
    }
    out.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return out

