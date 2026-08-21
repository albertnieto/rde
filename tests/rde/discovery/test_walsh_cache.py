"""Walsh-transform caching equivalence in descriptor template evaluation."""

from __future__ import annotations

import numpy as np

from rde.descriptor_gen.compute import evaluate_template, evaluate_templates
from rde.descriptor_gen.spec import DescriptorTemplate


def _walsh_templates() -> list[DescriptorTemplate]:
    return [
        DescriptorTemplate(
            template_id=f"walsh.{lo}_{hi}",
            family="fourier",
            key=f"gen.walsh.{lo}_{hi}",
            params={"lo": lo, "hi": hi},
            description="walsh band",
        )
        for lo, hi in ((0, 4), (4, 8), (8, 16))
    ]


def test_walsh_cache_matches_uncached_evaluation():
    rng = np.random.default_rng(0)
    vec = rng.normal(size=16)
    templates = _walsh_templates()
    uncached = {t.key: evaluate_template(t, vec, walsh_cache=None) for t in templates}
    cached_batch = evaluate_templates(templates, vec)
    assert uncached == cached_batch
