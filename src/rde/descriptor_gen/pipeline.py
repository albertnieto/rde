"""Pipeline-time generated descriptor module."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice
from rde.descriptor_gen.compute import evaluate_templates
from rde.descriptor_gen.enumerate import default_pipeline_templates

_PIPELINE_TEMPLATES = default_pipeline_templates()


def generated_descriptor(
    _instance: InstanceRecord,
    _slice_: SimpleFamilySlice | None,
    array: np.ndarray | None,
) -> dict[str, float]:
    """Compute the default parameterized descriptor catalog (gen.* keys)."""
    if array is None:
        return {}
    return evaluate_templates(_PIPELINE_TEMPLATES, np.asarray(array, dtype=float))


def list_pipeline_template_keys() -> list[str]:
    return [t.key for t in _PIPELINE_TEMPLATES]
