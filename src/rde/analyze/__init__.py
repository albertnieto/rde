"""Analysis helpers — query tables, ranker, calibration, outcome gates."""

from rde.analyze.calibration import complexity_by_group, separation_score
from rde.analyze.outcome import (
    OutcomeAssessment,
    assess_outcome,
    decisive_criteria_for,
)
from rde.analyze.query import (
    correlate_with_target,
    cross_n_report,
    distribution_summary,
    flatten_features,
    iter_flatten_features,
    kmeans_clusters,
    load_rows_from_shard,
    numeric_columns,
    outlier_rows,
    summarize_run,
    top_correlation_matrix,
)
from rde.analyze.ranker import Conjecture, ConjectureRanker, extrapolation_r_squared

__all__ = [
    "Conjecture",
    "ConjectureRanker",
    "OutcomeAssessment",
    "assess_outcome",
    "decisive_criteria_for",
    "complexity_by_group",
    "correlate_with_target",
    "cross_n_report",
    "distribution_summary",
    "extrapolation_r_squared",
    "flatten_features",
    "iter_flatten_features",
    "kmeans_clusters",
    "load_rows_from_shard",
    "numeric_columns",
    "outlier_rows",
    "separation_score",
    "summarize_run",
    "top_correlation_matrix",
]
