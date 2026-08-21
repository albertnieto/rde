"""End-to-end pipeline tests with SyntheticPoly domain."""

from __future__ import annotations

from pathlib import Path

import pytest

from rde.features import register_builtin_descriptors, register_builtin_metrics
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.core.registry import Registry
from rde.testing import SyntheticPolyDomain

# Domain-specific descriptors are registered by their optional plugin.

@pytest.fixture
def fresh_registry() -> Registry:
    reg = Registry()
    reg.register_domain(SyntheticPolyDomain())
    register_builtin_descriptors(reg)
    register_builtin_metrics(reg)
    return reg


def test_pipeline_end_to_end(tmp_path: Path, fresh_registry: Registry):
    config = RunConfig(
        domain_id="synthetic_poly",
        n_instances=3,
        size=16,
        seed=7,
        indices=[1, 2, 4],
        store_root=tmp_path,
        run_id="testrun",
        max_bruteforce_n=16,
    )
    result = run_pipeline(config, registry=fresh_registry)

    assert result.run_id == "testrun"
    assert result.n_feature_rows == 3 * 3
    assert result.n_instance_feature_rows == 3
    assert len(result.instance_ids) == 3

    from rde.io.store import Store

    store = Store(tmp_path)
    features = store.read_features("testrun")
    assert len(features) == 9
    instance_features = store.read_instance_features("testrun")
    assert len(instance_features) == 3

    row = features[0]
    assert "descriptors" in row
    assert "metrics" in row
    assert "spectral.effective_rank" in row["descriptors"]
    assert "representation_complexity" in row["metrics"]
    assert "compressibility_ratio" in row["metrics"]
    assert "array_ref" in row

    manifest = store.read_manifest("testrun")
    assert manifest.descriptor_names
    assert manifest.metric_names


def test_pipeline_deadline_is_checked_before_generation(
    tmp_path: Path, fresh_registry: Registry
):
    with pytest.raises(TimeoutError, match="max_wall_seconds"):
        run_pipeline(
            RunConfig(
                domain_id="synthetic_poly",
                n_instances=1,
                size=8,
                store_root=tmp_path,
                run_id="deadline",
                max_wall_seconds=0,
            ),
            registry=fresh_registry,
        )
