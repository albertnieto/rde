"""Human-readable discovery summaries and latent ↔ descriptor reports."""

from __future__ import annotations

from rde.discovery.latent import correlate_latent_descriptors
from rde.discovery.loop import DiscoveryReport
from rde.discovery.phase4 import Phase4Report
from rde.discovery.phase5 import Phase5Report

# Backward-compatible alias
latent_descriptor_correlations = correlate_latent_descriptors


def format_phase4_summary(report: Phase4Report) -> str:
    lines = [
        f"run_id={report.run_id} target={report.target} n_rows={report.n_rows}",
        f"checkpoints={report.checkpoint_dir}",
        "",
    ]
    if report.pca is not None:
        lines.append(
            f"PCA: dim={report.pca.latent_dim} recon_err={report.pca.reconstruction_error:.4g} "
            f"top_latent_corr={_max_abs(report.pca.target_correlations)}"
        )
    if report.feature_autoencoder is not None:
        ae = report.feature_autoencoder
        lines.append(
            f"Feature AE [{ae.method}]: hidden={ae.hidden_dim} recon={ae.reconstruction_error:.4g} "
            f"target_r={ae.target_correlation:+.3f} R²={ae.train_r_squared:.3f}"
        )
    if report.instance_q_autoencoder is not None and report.instance_q_autoencoder.n_instances > 0:
        iq = report.instance_q_autoencoder
        lines.append(
            f"Instance Q AE [{iq.method}]: n={iq.n_instances} q_dim={iq.q_dim} "
            f"target_r={iq.target_correlation:+.3f} top_latent={_max_abs(iq.target_correlations)}"
        )
    if report.trajectory is not None:
        tr = report.trajectory
        lines.append(
            f"Trajectory: train_R²={tr.train_r_squared:.3f} extrap_R²={tr.extrapolation_r_squared:.3f} "
            f"n_train={tr.n_train} n_test={tr.n_test}"
        )
    if report.dynamics_descriptor is not None:
        dd = report.dynamics_descriptor
        lines.append(f"Descriptor dynamics: R²={dd.train_r_squared:.3f} pairs={dd.n_pairs}")
    if report.dynamics_state is not None and report.dynamics_state.n_pairs > 0:
        sd = report.dynamics_state
        lines.append(
            f"State dynamics: R²={sd.train_r_squared:.3f} pairs={sd.n_pairs} "
            f"rank={sd.transition_rank}/{sd.state_dim} low_rank={sd.low_rank}"
        )
    if report.manifold is not None:
        mf = report.manifold
        lines.append(
            f"Manifold: k={mf.n_components} target_corr={mf.target_correlation:+.3f} "
            f"var={mf.explained_variance}"
        )
    if report.latent_descriptor_correlations:
        lines.extend(["", "Top latent ↔ descriptor:"])
        for hit in report.latent_descriptor_correlations[:5]:
            lines.append(
                f"  {hit['latent_column']} ~ {hit['descriptor_column']} "
                f"r={hit['pearson_r']:+.3f}"
            )
    return "\n".join(lines)


def format_phase5_summary(report: Phase5Report) -> str:
    lines = [
        f"run_id={report.run_id} target={report.target} n_rows={report.n_rows}",
        f"g1_met={report.g1_met}",
        "",
        f"Target fit [{report.symbolic.get('method')}]: "
        f"R²={report.symbolic.get('r_squared')} "
        f"equation={str(report.symbolic.get('equation', ''))[:96]}",
    ]
    skipped = report.symbolic.get("skipped_engines") or []
    if skipped:
        lines.append(f"Skipped engines: {skipped}")
        optional_engines = {"pysr", "operon", "native_gp", "ai_feynman"}
        tried = set(report.symbolic.get("candidates_tried") or [])
        skipped_names = {item.get("engine") for item in skipped}
        if skipped_names & optional_engines and not tried & optional_engines:
            lines.append(
                "No optional symbolic engine returned a usable fit; "
                "this is not evidence that no relation exists."
            )
    backends = report.symbolic.get("backends") or report.backends
    if backends:
        usable = [k for k, v in backends.items() if isinstance(v, bool) and v]
        lines.append(f"Backends usable: {usable}")
    for interp in report.latent_interpretations[:3]:
        lines.append(
            f"  Latent {interp.get('latent_column')}@{interp.get('latent_source')}: "
            f"R²={interp.get('r_squared')} corr={interp.get('main_target_correlation'):+.3f}"
        )
    if report.promoted_conjectures:
        lines.extend(["", "Promoted:"])
        for cand in report.promoted_conjectures[:5]:
            lines.append(
                f"  [{cand.get('source')}] {str(cand.get('expression', ''))[:72]} "
                f"R²={cand.get('r_squared')}"
            )
    return "\n".join(lines)


def _max_abs(corrs: dict[str, float]) -> str:
    if not corrs:
        return "nan"
    key = max(corrs, key=lambda k: abs(corrs[k]))
    return f"{key}={corrs[key]:+.3f}"


def format_discovery_summary(report: DiscoveryReport) -> str:
    """Plain-text summary for logs or EXP notes."""
    lines = [
        f"run_id={report.run_id} target={report.target} n_rows={report.n_rows}",
        f"outcome_grade_hint={report.outcome_grade_hint}",
        "",
        "Top correlations:",
    ]
    for hit in report.top_correlations[:5]:
        lines.append(
            f"  {hit.get('column')} r={hit.get('pearson_r'):+.3f} "
            f"R²={hit.get('r_squared', float('nan')):.3f} "
            f"cross_N={hit.get('cross_n_sign_stability', float('nan'))}"
        )
    lines.extend(
        [
            "",
            f"Descriptor candidates: {len(report.descriptor_candidates)}",
            (
                "Massive descriptor catalog: skipped because persisted slice arrays "
                "are missing (save_arrays=True required)."
                if report.massive_catalog_skipped_no_arrays
                else (
                    "Massive descriptor catalog: evaluated."
                    if report.parameters.get("use_massive_catalog")
                    else "Massive descriptor catalog: not requested."
                )
            ),
        ]
    )
    lines.append("")
    lines.append("Metric candidates:")
    for cand in report.metric_candidates[:5]:
        lines.append(
            f"  {cand.get('expression')} R²={cand.get('r_squared')} "
            f"level={cand.get('level_hint')}"
        )
    lat = report.latent or {}
    if lat:
        lines.extend(["", "Phase 4 latent:"])
        if lat.get("autoencoder"):
            ae = lat["autoencoder"]
            lines.append(
                f"  Feature AE [{ae.get('method')}]: target_r={ae.get('target_correlation'):+.3f}"
            )
        if lat.get("instance_q_autoencoder", {}).get("n_instances"):
            iq = lat["instance_q_autoencoder"]
            lines.append(
                f"  Instance Q AE: n={iq.get('n_instances')} target_r={iq.get('target_correlation'):+.3f}"
            )
        if lat.get("state_dynamics", {}).get("n_pairs"):
            sd = lat["state_dynamics"]
            lines.append(
                f"  State dynamics: R²={sd.get('train_r_squared'):.3f} "
                f"rank={sd.get('transition_rank')}/{sd.get('state_dim')}"
            )
    if report.symbolic:
        lines.extend(
            [
                "",
                f"Symbolic [{report.symbolic.get('method')}]: "
                f"{report.symbolic.get('equation')} R²={report.symbolic.get('r_squared')}",
            ]
        )
        latent_interp = report.symbolic.get("latent_interpretation")
        if latent_interp:
            lines.append(
                f"Latent [{latent_interp.get('latent_column')}]: "
                f"{latent_interp.get('equation')} R²={latent_interp.get('r_squared')} "
                f"(corr w/ target={latent_interp.get('main_target_correlation'):+.3f})"
            )
        for interp in report.symbolic.get("latent_interpretations") or []:
            if interp is latent_interp:
                continue
            lines.append(
                f"Latent [{interp.get('latent_column')}@{interp.get('latent_source')}]: "
                f"R²={interp.get('r_squared')} corr={interp.get('main_target_correlation'):+.3f}"
            )
    if report.promoted_conjectures:
        lines.extend(["", "Promoted conjectures (top-ranked):"])
        for cand in report.promoted_conjectures[:5]:
            lines.append(
                f"  [{cand.get('source')}] {str(cand.get('expression', ''))[:72]} "
                f"R²={cand.get('r_squared')}"
            )
    if report.cross_n_rule:
        lines.extend(["", "Cross-N rule:"])
        for cn in report.cross_n_rule[:3]:
            if cn.get("status") != "ok":
                continue
            tag = "stable" if cn.get("stable") else "unstable"
            slope_rule = cn.get("slope_rule") or {}
            lines.append(
                f"  [{cn.get('candidate_source')}] slope({cn.get('candidate_expression')}) ~ "
                f"{slope_rule.get('equation')} R²={slope_rule.get('r_squared')} [{tag}]"
            )
    if report.coordinates:
        lines.extend(
            [
                "",
                f"Coordinates: {report.coordinates.get('method')} "
                f"k={report.coordinates.get('n_components')} "
                f"target_corr={report.coordinates.get('target_correlation')} "
                f"update_R²={report.coordinates.get('update_simplicity')}",
            ]
        )
    phase6 = report.phase6 or {}
    rep = phase6.get("representation") or {}
    if rep.get("n_samples"):
        lines.extend(
            [
                "",
                f"Representation (Q,r)→z→Ψ: latent_dim={rep.get('latent_dim')} "
                f"state_R²={rep.get('state_reconstruction_r2')} "
                f"update_R²={rep.get('latent_update_r2')} "
                f"g4={rep.get('g4_met')}",
            ]
        )
    walsh = phase6.get("walsh_synthesis") or {}
    if walsh.get("top_programs"):
        top_w = walsh["top_programs"][0]
        lines.append(
            f"Walsh synthesis: {top_w.get('expression', '')[:60]} "
            f"r={top_w.get('pearson_r'):+.3f}"
        )
    redisc = phase6.get("rediscovery") or {}
    if redisc:
        lines.extend(
            [
                "",
                f"Rediscovery: freq={redisc.get('rediscovery_frequency')} "
                f"splits={redisc.get('splits_hit')}/{phase6.get('rediscovery', {}).get('n_splits', 5)} "
                f"G5={redisc.get('g5_met')} hint={redisc.get('g5_hint')}",
            ]
        )
    if report.operator:
        lines.extend(
            [
                "",
                f"Operator: {report.operator.get('method')} "
                f"R²={report.operator.get('train_r_squared')} "
                f"n={report.operator.get('n_rows')}",
            ]
        )
    if report.outcome:
        lines.extend(
            [
                "",
                f"Outcome: grade={report.outcome.get('grade')} "
                f"G0={report.outcome.get('g0_met')} "
                f"G1={report.outcome.get('g1_met')} "
                f"G4={report.outcome.get('g4_met')} "
                f"G5={report.outcome.get('g5_met')}",
            ]
        )
    return "\n".join(lines)
