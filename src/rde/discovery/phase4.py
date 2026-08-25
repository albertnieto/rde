"""Phase 4 orchestration — latent discovery (ML as hypothesis generator)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rde.discovery.autoencoder import (
    AutoencoderResult,
    InstanceQResult,
    discover_autoencoder_with_checkpoint,
    discover_instance_q_autoencoder,
)
from rde.discovery.checkpoint import checkpoint_dir, save_latent_checkpoint
from rde.discovery.context import DiscoveryContext
from rde.io.json_util import json_default
from rde.discovery.datasets import build_feature_dataset, load_rows_from_source
from rde.discovery.state_dynamics import DynamicsStepResult, StateDynamicsResult, predict_dynamics_step, predict_state_dynamics
from rde.discovery.latent import (
    LatentResult,
    correlate_latent_descriptors,
    discover_latent_from_run,
    pca_latent,
    ridge_predict_columns,
)
from rde.discovery.manifold import ManifoldResult, manifold_from_rows
from rde.discovery.trajectory import TrajectoryPredictorResult, predict_trajectory_target
from rde.runtime.targets import resolve_target_for_run


@dataclass
class Phase4Report:
    run_id: str
    target: str
    n_rows: int
    pca: LatentResult | None = None
    feature_autoencoder: AutoencoderResult | None = None
    instance_q_autoencoder: InstanceQResult | None = None
    trajectory: TrajectoryPredictorResult | None = None
    dynamics_descriptor: DynamicsStepResult | None = None
    dynamics_state: StateDynamicsResult | None = None
    manifold: ManifoldResult | None = None
    latent_descriptor_correlations: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_dir: str = ""
    ridge_r_squared: float = float("nan")

    def to_latent_dict(self) -> dict[str, Any]:
        """Backward-compatible ``report.latent`` payload for discovery loop / outcome."""
        latent: dict[str, Any] = {}
        if self.pca is not None:
            latent.update(
                {
                    "latent_dim": self.pca.latent_dim,
                    "explained_variance_ratio": self.pca.explained_variance_ratio,
                    "target_correlations": self.pca.target_correlations,
                    "reconstruction_error": self.pca.reconstruction_error,
                }
            )
        latent["ridge_r_squared"] = self.ridge_r_squared

        if self.feature_autoencoder is not None:
            latent["autoencoder"] = {
                "method": self.feature_autoencoder.method,
                "hidden_dim": self.feature_autoencoder.hidden_dim,
                "reconstruction_error": self.feature_autoencoder.reconstruction_error,
                "train_r_squared": self.feature_autoencoder.train_r_squared,
                "target_correlation": self.feature_autoencoder.target_correlation,
                "latent_columns": self.feature_autoencoder.latent_columns,
            }
        else:
            latent["autoencoder"] = {}

        if self.instance_q_autoencoder is not None:
            latent["instance_q_autoencoder"] = {
                "method": self.instance_q_autoencoder.method,
                "n_instances": self.instance_q_autoencoder.n_instances,
                "q_dim": self.instance_q_autoencoder.q_dim,
                "hidden_dim": self.instance_q_autoencoder.hidden_dim,
                "reconstruction_error": self.instance_q_autoencoder.reconstruction_error,
                "train_r_squared": self.instance_q_autoencoder.train_r_squared,
                "target_correlation": self.instance_q_autoencoder.target_correlation,
                "target_correlations": self.instance_q_autoencoder.target_correlations,
            }
        else:
            latent["instance_q_autoencoder"] = {}

        if self.trajectory is not None:
            latent["trajectory_predictor"] = {
                "train_r_squared": self.trajectory.train_r_squared,
                "extrapolation_r_squared": self.trajectory.extrapolation_r_squared,
                "n_train": self.trajectory.n_train,
                "n_test": self.trajectory.n_test,
                "feature_columns": self.trajectory.feature_columns,
            }
        else:
            latent["trajectory_predictor"] = {}

        if self.manifold is not None:
            latent["manifold"] = {
                "n_components": self.manifold.n_components,
                "explained_variance": self.manifold.explained_variance,
                "target_correlation": self.manifold.target_correlation,
            }
        else:
            latent["manifold"] = {}

        if self.dynamics_descriptor is not None:
            latent["dynamics_step"] = {
                "train_r_squared": self.dynamics_descriptor.train_r_squared,
                "n_pairs": self.dynamics_descriptor.n_pairs,
                "feature_columns": self.dynamics_descriptor.feature_columns,
            }
        else:
            latent["dynamics_step"] = {}

        if self.dynamics_state is not None:
            latent["state_dynamics"] = {
                "train_r_squared": self.dynamics_state.train_r_squared,
                "n_pairs": self.dynamics_state.n_pairs,
                "state_dim": self.dynamics_state.state_dim,
                "transition_rank": self.dynamics_state.transition_rank,
                "low_rank": self.dynamics_state.low_rank,
                "method": self.dynamics_state.method,
            }
        else:
            latent["state_dynamics"] = {}

        latent["latent_descriptor_correlations"] = self.latent_descriptor_correlations
        latent["checkpoint_dir"] = self.checkpoint_dir
        return latent


def run_phase4(
    run_id: str,
    store_root: Path | str,
    *,
    target: str | None = None,
    n_pca_components: int = 8,
    hidden_dim: int = 16,
    prefer_torch: bool = False,
    prefer_mlx: bool | None = None,
    require_gpu: bool = False,
    compute_backend: str | None = None,
    on_ae_epoch: Callable[[int, int], None] | None = None,
    on_instance_ae_epoch: Callable[[int, int], None] | None = None,
    checkpoint: bool = True,
    on_stage: Callable[[str], None] | None = None,
    context: DiscoveryContext | None = None,
) -> Phase4Report:
    """Run all Phase 4 latent-discovery experiments on one run ledger."""
    def _stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    store_root = Path(store_root)
    if context is not None:
        resolved_target = context.target
        rows = context.rows
        domain_id = context.domain_id()
    else:
        resolved_target = resolve_target_for_run(run_id, store_root, override=target)
        rows = load_rows_from_source(run_id=run_id, store_root=store_root)
        from rde.io.store import Store

        try:
            domain_id = Store(store_root).read_manifest(run_id).domain_id
        except (FileNotFoundError, OSError, ValueError, KeyError):
            domain_id = None
    report = Phase4Report(
        run_id=run_id,
        target=resolved_target,
        n_rows=len(rows),
        checkpoint_dir=str(checkpoint_dir(store_root, run_id)),
    )
    if not rows:
        return report

    from rde.analyze.tables import contract_excluded_columns

    excluded = contract_excluded_columns(rows, domain_id)
    feat_cols = [
        c
        for c in sorted({k for row in rows for k in row if isinstance(row.get(k), (int, float))})
        if c not in {resolved_target, "size", "seed", "run_id", "instance_id", "family_index"}
        and c not in excluded
        and not str(c).startswith("metric.")
    ][:32]

    _stage("pca latent")
    if context is not None:
        context_X, context_cols = context.feature_matrix()
        pca_cols = [
            c
            for c in context_cols
            if c != resolved_target
            and c not in excluded
            and not str(c).startswith(("metric.", "run_id"))
        ]
        pca_idx = [context_cols.index(c) for c in pca_cols]
        pca_target = context.target_array()
        report.pca = pca_latent(
            context_X[:, pca_idx],
            n_components=n_pca_components,
            target=pca_target,
        )
        if (
            checkpoint
            and report.pca.latent_codes.size
            and report.pca.components is not None
            and report.pca.singular_values is not None
        ):
            save_latent_checkpoint(
                store_root,
                run_id,
                "pca",
                arrays={
                    "components": report.pca.components,
                    "singular_values": report.pca.singular_values,
                },
                metadata={
                    "method": "pca",
                    "target_column": resolved_target,
                    "feature_columns": pca_cols,
                    "latent_dim": report.pca.latent_dim,
                    "reconstruction_error": report.pca.reconstruction_error,
                },
            )
    else:
        report.pca = discover_latent_from_run(
            run_id,
            store_root,
            target_column=resolved_target,
            n_components=n_pca_components,
            domain_id=domain_id,
            checkpoint=checkpoint,
        )
    if report.pca is not None and report.pca.latent_codes.size:
        _, report.ridge_r_squared = ridge_predict_columns(
            report.pca.latent_codes,
            context.target_array()
            if context is not None
            else np.array([row.get(resolved_target, float("nan")) for row in rows], dtype=float),
        )

    dataset = (
        context.feature_dataset()
        if context is not None
        else build_feature_dataset(run_id, store_root, target=resolved_target)
    )
    _stage("feature autoencoder")
    report.feature_autoencoder = discover_autoencoder_with_checkpoint(
        run_id,
        store_root,
        target=dataset.target,
        X=dataset.features,
        hidden_dim=hidden_dim,
        prefer_torch=prefer_torch,
        prefer_mlx=prefer_mlx,
        require_gpu=require_gpu,
        on_epoch=on_ae_epoch,
        checkpoint=checkpoint,
    )

    _stage("instance Q autoencoder")
    report.instance_q_autoencoder = discover_instance_q_autoencoder(
        run_id,
        store_root,
        target=resolved_target,
        hidden_dim=min(hidden_dim, 12),
        prefer_torch=prefer_torch,
        prefer_mlx=prefer_mlx,
        require_gpu=require_gpu,
        checkpoint=checkpoint,
        on_epoch=on_instance_ae_epoch,
    )

    _stage("trajectory predictor")
    report.trajectory = predict_trajectory_target(
        rows,
        resolved_target,
        feature_columns=feat_cols[:12],
    )

    _stage("descriptor dynamics")
    report.dynamics_descriptor = predict_dynamics_step(
        rows,
        resolved_target,
        feature_columns=feat_cols[:12],
    )

    _stage("state dynamics")
    state_backend = compute_backend or ("mlx" if prefer_mlx or require_gpu else "numpy")
    report.dynamics_state = predict_state_dynamics(
        run_id,
        store_root,
        checkpoint=checkpoint,
        compute_backend=state_backend,
    )

    _stage("manifold embedding")
    report.manifold = manifold_from_rows(
        rows,
        resolved_target,
        feature_columns=feat_cols[:20],
    )

    _stage("latent-descriptor correlations")
    if report.feature_autoencoder is not None and report.feature_autoencoder.latent_codes.size:
        report.latent_descriptor_correlations = correlate_latent_descriptors(
            report.feature_autoencoder.latent_codes,
            report.feature_autoencoder.latent_columns,
            rows,
            target=resolved_target,
        )

    return report


def write_phase4_report(report: Phase4Report, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": report.run_id,
        "target": report.target,
        "n_rows": report.n_rows,
        "checkpoint_dir": report.checkpoint_dir,
        "latent": report.to_latent_dict(),
    }
    out.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return out

