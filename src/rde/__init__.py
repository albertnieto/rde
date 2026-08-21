"""Representation Discovery Engine — domain-agnostic experimental-mathematics platform."""

from rde.analyze.query import flatten_features, summarize_run
from rde.runtime.campaign import CampaignConfig, run_campaign
from rde.core.instance import InstanceRecord
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.core.protocols import (
    Descriptor,
    Domain,
    FamilySlice,
    Metric,
    QueryIntent,
)
from rde.core.registry import Registry, get_registry

__all__ = [
    "CampaignConfig",
    "Descriptor",
    "Domain",
    "FamilySlice",
    "InstanceRecord",
    "Metric",
    "QueryIntent",
    "Registry",
    "RunConfig",
    "flatten_features",
    "get_registry",
    "run_campaign",
    "run_pipeline",
    "summarize_run",
]

__version__ = "0.1.0"
