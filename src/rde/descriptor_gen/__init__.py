"""Parameterized descriptor generators for RDE Phase 2b."""

from rde.descriptor_gen.compute import evaluate_template, evaluate_templates
from rde.descriptor_gen.enumerate import (
    catalog_size,
    default_pipeline_templates,
    enumerate_descriptor_templates,
)
from rde.descriptor_gen.materialize import load_row_arrays, materialize_template_column
from rde.descriptor_gen.pipeline import generated_descriptor, list_pipeline_template_keys
from rde.descriptor_gen.rank import (
    DerivedDescriptorRanker,
    DescriptorCandidate,
    DescriptorTemplateRanker,
    candidates_to_records,
    descriptor_variable_columns,
    rank_descriptor_generators,
)
from rde.descriptor_gen.spec import DescriptorTemplate

__all__ = [
    "DescriptorCandidate",
    "DescriptorTemplate",
    "DescriptorTemplateRanker",
    "DerivedDescriptorRanker",
    "catalog_size",
    "candidates_to_records",
    "default_pipeline_templates",
    "descriptor_variable_columns",
    "enumerate_descriptor_templates",
    "evaluate_template",
    "evaluate_templates",
    "generated_descriptor",
    "list_pipeline_template_keys",
    "load_row_arrays",
    "materialize_template_column",
    "rank_descriptor_generators",
]
