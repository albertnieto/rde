from __future__ import annotations

import numpy as np

from rde.descriptor_gen.compute import evaluate_template, template_family_executable
from rde.descriptor_gen.enumerate_massive import enumerate_massive_templates


def test_massive_catalog_has_100k_executable_unique_templates():
    templates = enumerate_massive_templates(max_templates=100_000)
    assert len(templates) == 100_000
    assert len({template.canonical_id for template in templates}) == len(templates)
    assert all(template.provenance["executable"] for template in templates)
    assert all(template_family_executable(template.family) for template in templates)


def test_massive_quantile_template_is_real_and_evaluable():
    template = enumerate_massive_templates(max_templates=100_000)[-1]
    value = evaluate_template(template, np.arange(16, dtype=float))
    assert np.isfinite(value)
