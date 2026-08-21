"""Tests for parameterized descriptor generators."""

from __future__ import annotations

import numpy as np

from rde.descriptor_gen.compute import evaluate_template
from rde.descriptor_gen.enumerate import enumerate_descriptor_templates
from rde.descriptor_gen.materialize import materialize_template_column
from rde.descriptor_gen.rank import rank_descriptor_generators
from rde.descriptor_gen.spec import DescriptorTemplate
from rde.features import register_builtin_descriptors, register_builtin_metrics
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.core.registry import Registry
from rde.testing import SyntheticPolyDomain


def test_template_spectral_sv_ratio():
    template = DescriptorTemplate(
        template_id="test.ratio",
        family="spectral",
        key="gen.spectral.sv_ratio_0_1",
        params={"kind": "sv_ratio", "i": 0, "j": 1},
    )
    arr = np.array([3.0, 2.0, 1.0, 0.5])
    val = evaluate_template(template, arr)
    assert np.isclose(val, 3.0 / 2.0)


def test_template_fourier_band_energy_power_of_two():
    templates = [t for t in enumerate_descriptor_templates() if t.family == "fourier"]
    assert templates
    n = 8
    arr = np.linspace(-1.0, 1.0, n)
    vals = [evaluate_template(t, arr) for t in templates[:3]]
    assert all(np.isfinite(v) for v in vals)
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_pipeline_includes_gen_descriptors(tmp_path):
    reg = Registry()
    reg.register_domain(SyntheticPolyDomain())
    register_builtin_descriptors(reg)
    register_builtin_metrics(reg)
    config = RunConfig(
        domain_id="synthetic_poly",
        n_instances=2,
        size=16,
        seed=3,
        indices=[1, 2],
        store_root=tmp_path,
        run_id="gen_test",
        max_bruteforce_n=16,
    )
    run_pipeline(config, registry=reg)
    from rde.io.store import Store

    store = Store(tmp_path)
    row = store.read_features("gen_test")[0]
    keys = row["descriptors"]
    gen_keys = [k for k in keys if k.startswith("gen.")]
    assert len(gen_keys) >= 20
    assert "gen.spectral.effective_rank" in keys


def test_rank_descriptor_generators_on_pipeline_run(tmp_path):
    reg = Registry()
    reg.register_domain(SyntheticPolyDomain())
    register_builtin_descriptors(reg)
    register_builtin_metrics(reg)
    run_id = "rank_gen"
    config = RunConfig(
        domain_id="synthetic_poly",
        n_instances=4,
        size=16,
        seed=11,
        indices=[1, 2, 4],
        store_root=tmp_path,
        run_id=run_id,
        max_bruteforce_n=16,
    )
    run_pipeline(config, registry=reg)
    from rde.analyze.query import flatten_features

    rows = flatten_features(run_id, tmp_path)
    target = "metric.representation_complexity"
    templates = enumerate_descriptor_templates(max_templates=40)
    ranked = rank_descriptor_generators(
        rows,
        run_id=run_id,
        store_root=tmp_path,
        target=target,
        templates=templates,
        max_derived=200,
        max_derived_depth=2,
        min_abs_r=0.0,
        max_results=5,
        include_derived=True,
    )
    assert ranked
    assert ranked[0].key
    assert ranked[0].kind in {"template", "derived"}


def test_materialize_reduction_column_vectorized_matches_per_row():
    """_materialize_reduction_column batches rows by raveled array shape;
    verify that against a naive per-row reference, including ragged shapes
    (mixed array lengths) and a row with no matching array (NaN)."""
    rows = [
        {"instance_id": "a", "family_index": 0},
        {"instance_id": "b", "family_index": 0},
        {"instance_id": "c", "family_index": 1},
        {"instance_id": "d", "family_index": 1},
        {"instance_id": "missing", "family_index": 0},
    ]
    arrays = {
        ("a", 0): np.array([1.0, -2.0, 3.0]),
        ("b", 0): np.array([-4.0, 5.0, -6.0]),
        ("c", 1): np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ("d", 1): np.array([-1.0, -2.0, -3.0, -4.0, -5.0]),
    }

    def expected(template):
        out = []
        for row in rows:
            key = (str(row["instance_id"]), int(row["family_index"]))
            arr = arrays.get(key)
            if arr is None:
                out.append(np.nan)
                continue
            arr = np.asarray(arr, dtype=float)
            if template.family == "norm":
                p = template.params["p"]
                if p == float("inf"):
                    out.append(np.max(np.abs(arr)))
                else:
                    out.append(np.sum(np.abs(arr) ** p) ** (1.0 / p))
            elif template.params["kind"] == "quantile":
                out.append(np.quantile(arr, template.params["q"]))
            elif template.params["kind"] == "mad":
                med = np.median(arr)
                out.append(np.median(np.abs(arr - med)))
            elif template.params["kind"] == "iqr":
                q1, q3 = np.quantile(arr, [0.25, 0.75])
                out.append(q3 - q1)
        return np.array(out)

    templates = [
        DescriptorTemplate(template_id="t1", family="norm", key="gen.norm.l2", params={"p": 2.0}),
        DescriptorTemplate(template_id="t2", family="norm", key="gen.norm.linf", params={"p": float("inf")}),
        DescriptorTemplate(
            template_id="t3",
            family="stats",
            key="gen.stats.quantile_50",
            params={"kind": "quantile", "q": 0.5},
        ),
        DescriptorTemplate(template_id="t4", family="stats", key="gen.stats.mad", params={"kind": "mad"}),
        DescriptorTemplate(template_id="t5", family="stats", key="gen.stats.iqr", params={"kind": "iqr"}),
    ]
    for template in templates:
        values = materialize_template_column(rows, template, arrays)
        np.testing.assert_allclose(values, expected(template), equal_nan=True)


def test_materialize_sv_ratio_out_of_range_is_nan_not_crash():
    """Massive catalog asks for sv_ratio i,j up to 6; N=2 arrays have dim 4."""
    rows = [
        {"instance_id": "a", "family_index": 0},
        {"instance_id": "b", "family_index": 0},
    ]
    arrays = {
        ("a", 0): np.linspace(0.1, 1.0, 4),
        ("b", 0): np.linspace(0.2, 0.9, 4),
    }
    template = DescriptorTemplate(
        template_id="spectral_oob",
        family="massive_spectral",
        key="gen.massive.svd.ratio_4_5",
        params={"kind": "sv_ratio", "i": 4, "j": 5},
    )
    values = materialize_template_column(rows, template, arrays)
    assert values.shape == (2,)
    assert np.all(np.isnan(values))
    expected = np.array(
        [evaluate_template(template, arrays[(row["instance_id"], 0)]) for row in rows]
    )
    np.testing.assert_allclose(values, expected, equal_nan=True)


def test_materialize_transform_columns_match_per_row():
    rows = [
        {"instance_id": "a", "family_index": 0},
        {"instance_id": "b", "family_index": 0},
        {"instance_id": "c", "family_index": 0},
        {"instance_id": "missing", "family_index": 0},
    ]
    arrays = {
        ("a", 0): np.array([0.0, 1.0, 0.5, 0.25, 0.75, 0.0, 1.0, 0.5]),
        ("b", 0): np.array([-1.0, 0.5, 2.0, -0.25, 0.75, 1.0, -0.5, 0.25]),
        ("c", 0): np.array([[2.0, 1.0], [0.5, -1.0]]),
    }
    templates = [
        DescriptorTemplate(
            template_id="fourier",
            family="fourier",
            key="gen.fourier.band",
            params={"lo": 1, "hi": 4},
        ),
        DescriptorTemplate(
            template_id="spectral_vector",
            family="spectral",
            key="gen.spectral.vector_top",
            params={"kind": "top_k_energy", "k": 2},
        ),
        DescriptorTemplate(
            template_id="spectral_matrix",
            family="spectral",
            key="gen.spectral.matrix_ratio",
            params={"kind": "sv_ratio", "i": 0, "j": 1},
        ),
        DescriptorTemplate(
            template_id="autocorr",
            family="autocorr",
            key="gen.autocorr.lag1",
            params={"lag": 1},
        ),
        DescriptorTemplate(
            template_id="trimmed",
            family="stats",
            key="gen.stats.trimmed",
            params={"kind": "trimmed_mean", "frac": 0.25},
        ),
    ]
    for template in templates:
        values = materialize_template_column(rows, template, arrays)
        expected = np.array(
            [
                evaluate_template(template, arrays[(row["instance_id"], 0)])
                if (row["instance_id"], 0) in arrays
                else np.nan
                for row in rows
            ]
        )
        np.testing.assert_allclose(values, expected, equal_nan=True)
