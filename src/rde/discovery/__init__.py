"""Representation discovery — Phases 4–6."""

from rde.discovery.autoencoder import (
    InstanceQResult,
    discover_autoencoder,
    discover_instance_q_autoencoder,
    random_feature_autoencoder,
)
from rde.discovery.checkpoint import load_latent_checkpoint, save_latent_checkpoint
from rde.discovery.coordinates import CoordinateResult, search_linear_coordinates
from rde.discovery.context import DiscoveryContext, StageTimer
from rde.discovery.datasets import (
    FeatureDataset,
    InstanceQBatch,
    StatePairBatch,
    broadcast_instance_latents,
    build_feature_dataset,
    build_feature_matrix_from_rows,
    load_campaign_matrix,
    load_feature_matrix,
    load_instance_q_batch,
    load_parquet_features,
    load_sealed_campaign,
    load_sealed_run_rows,
    load_state_pair_batch,
)
from rde.discovery.state_dynamics import DynamicsStepResult, StateDynamicsResult, predict_dynamics_step, predict_state_dynamics
from rde.discovery.latent import (
    LatentResult,
    correlate_latent_descriptors,
    discover_latent_from_run,
    load_pca_from_checkpoint,
    pca_latent,
)
from rde.discovery.loop import (
    DiscoveryReport,
    run_discovery,
    write_conjectures_jsonl,
    write_descriptor_conjectures_jsonl,
    write_discovery_report,
)
from rde.discovery.manifold import ManifoldResult, manifold_from_rows
from rde.discovery.phase4 import Phase4Report, run_phase4, write_phase4_report
from rde.discovery.phase5 import Phase5Report, run_phase5, write_phase5_report
from rde.discovery.trajectory import TrajectoryPredictorResult, predict_trajectory_target
from rde.discovery.operator import OperatorResult, learn_update_predictor
from rde.discovery.phase6 import Phase6Report, run_phase6, write_phase6_report
from rde.discovery.programs import ProgramResult, evolve_programs
from rde.discovery.promote import promote_top_conjectures
from rde.discovery.rediscovery import RediscoveryAssessment, assess_rediscovery, canonicalize_expression
from rde.discovery.representation import RepresentationResult, discover_representation_encoder_decoder
from rde.discovery.walsh_synthesis import WalshSynthesisResult, run_walsh_synthesis, search_walsh_generating_functions
from rde.discovery.report import format_discovery_summary, format_phase4_summary, format_phase5_summary, latent_descriptor_correlations
from rde.discovery.symbolic import (
    LatentSource,
    SymbolicDiscoveryResult,
    SymbolicResult,
    discover_equation,
    discover_latent_interpretation,
    discover_latent_interpretations,
    run_symbolic_discovery,
    symbolic_backends,
)

__all__ = [
    "CoordinateResult",
    "DiscoveryContext",
    "DiscoveryReport",
    "DynamicsStepResult",
    "FeatureDataset",
    "InstanceQBatch",
    "InstanceQResult",
    "LatentResult",
    "LatentSource",
    "ManifoldResult",
    "OperatorResult",
    "Phase4Report",
    "Phase5Report",
    "Phase6Report",
    "ProgramResult",
    "RediscoveryAssessment",
    "RepresentationResult",
    "StateDynamicsResult",
    "StatePairBatch",
    "SymbolicDiscoveryResult",
    "SymbolicResult",
    "WalshSynthesisResult",
    "TrajectoryPredictorResult",
    "StageTimer",
    "assess_rediscovery",
    "broadcast_instance_latents",
    "build_feature_dataset",
    "build_feature_matrix_from_rows",
    "canonicalize_expression",
    "correlate_latent_descriptors",
    "discover_autoencoder",
    "discover_equation",
    "discover_instance_q_autoencoder",
    "discover_representation_encoder_decoder",
    "discover_latent_from_run",
    "discover_latent_interpretation",
    "discover_latent_interpretations",
    "evolve_programs",
    "format_discovery_summary",
    "format_phase4_summary",
    "format_phase5_summary",
    "latent_descriptor_correlations",
    "learn_update_predictor",
    "load_campaign_matrix",
    "load_feature_matrix",
    "load_instance_q_batch",
    "load_latent_checkpoint",
    "load_parquet_features",
    "load_pca_from_checkpoint",
    "load_state_pair_batch",
    "manifold_from_rows",
    "pca_latent",
    "predict_dynamics_step",
    "predict_state_dynamics",
    "predict_trajectory_target",
    "promote_top_conjectures",
    "random_feature_autoencoder",
    "run_discovery",
    "run_phase5",
    "run_phase6",
    "run_phase4",
    "run_walsh_synthesis",
    "run_symbolic_discovery",
    "save_latent_checkpoint",
    "search_walsh_generating_functions",
    "search_linear_coordinates",
    "symbolic_backends",
    "write_conjectures_jsonl",
    "write_descriptor_conjectures_jsonl",
    "write_discovery_report",
    "write_phase5_report",
    "write_phase6_report",
    "write_phase4_report",
]
