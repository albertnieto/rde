"""Performance tier tests: vectorization, workers, backends, parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rde.backends import (
    available_backends,
    compute_backend_for_size,
    get_compute_backend,
    load_crossover_table,
    recommended_batch_size,
    usable_backends,
)
from rde.backends.numpy_backend import NumpyBackend
from rde.features.landscape import _hamming_weights
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.io.store import Store
from tests.rde.helpers import toy_registry
from rde.testing import SyntheticPolyDomain


def _random_upper_triangular(rng: np.random.Generator, n: int) -> np.ndarray:
    matrix = rng.normal(size=(n, n))
    return np.triu(matrix)


def test_numpy_backend_trajectory_states():
    rng = np.random.default_rng(42)
    q = _random_upper_triangular(rng, 3)
    backend = NumpyBackend()
    w_amp, *_ = backend.bound_normalized_weights(q)
    states = backend.trajectory_states(w_amp, [1, 2, 4])
    for r, vec in states.items():
        assert vec.shape == w_amp.shape
        assert np.isclose(np.linalg.norm(vec), 1.0)


@pytest.mark.parametrize("workers", [1, 2])
def test_pipeline_workers(tmp_path: Path, workers: int):
    reg = toy_registry()
    result = run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=4,
            size=8,
            seed=7,
            indices=[1, 2],
            store_root=tmp_path,
            run_id=f"workers_{workers}",
            workers=workers,
        ),
        registry=reg,
    )
    assert result.n_feature_rows == 8


@pytest.mark.parametrize("backend_name", ["numpy"])
def test_compute_backend_costs(backend_name: str):
    if backend_name not in available_backends():
        pytest.skip(f"{backend_name} not available")
    rng = np.random.default_rng(1)
    q = _random_upper_triangular(rng, 4)
    ref = NumpyBackend().all_costs(q)
    backend = get_compute_backend(backend_name)
    got = backend.all_costs(q)
    assert np.allclose(ref, got, rtol=1e-10, atol=1e-10)


def test_torch_cpu_backend_if_available():
    if "torch_cpu" not in usable_backends():
        pytest.skip("torch_cpu not usable on this machine")
    rng = np.random.default_rng(1)
    q = _random_upper_triangular(rng, 4)
    ref = NumpyBackend().all_costs(q)
    got = get_compute_backend("torch_cpu").all_costs(q)
    assert np.allclose(ref, got, rtol=1e-10, atol=1e-10)


def test_torch_mps_backend_if_available():
    if "torch_mps" not in usable_backends():
        pytest.skip("torch_mps not usable on this machine")
    rng = np.random.default_rng(2)
    q = _random_upper_triangular(rng, 4)
    ref = NumpyBackend().all_costs(q)
    got = get_compute_backend("torch_mps").all_costs(q)
    assert np.allclose(ref, got, rtol=1e-5, atol=1e-5)


def test_mlx_backend_if_available():
    if "mlx" not in usable_backends():
        pytest.skip("mlx not available on this machine")
    rng = np.random.default_rng(3)
    q = _random_upper_triangular(rng, 4)
    ref = NumpyBackend().all_costs(q)
    got = get_compute_backend("mlx").all_costs(q)
    assert np.allclose(ref, got, rtol=1e-5, atol=1e-5)


def test_mlx_assignment_matrix_is_reused():
    if "mlx" not in usable_backends():
        pytest.skip("mlx not available on this machine")
    backend = get_compute_backend("mlx")
    backend._assignment_cache.clear()
    rng = np.random.default_rng(31)
    q = _random_upper_triangular(rng, 4)
    backend.all_costs(q)
    first = backend._assignment_cache[4]
    backend.all_costs(q)
    assert backend._assignment_cache[4] is first


def test_numpy_assignment_matrix_is_reused():
    backend = NumpyBackend()
    backend._assignment_cache.clear()
    rng = np.random.default_rng(32)
    q = _random_upper_triangular(rng, 4)
    backend.all_costs(q)
    first = backend._assignment_cache[4]
    backend.all_costs(q)
    assert backend._assignment_cache[4] is first


def test_numpy_assignment_matrix_cache_is_read_only():
    backend = NumpyBackend()
    rng = np.random.default_rng(33)
    q = _random_upper_triangular(rng, 4)
    expected = backend.all_costs(q)
    assignments = backend._assignment_cache[4]

    assert not assignments.flags.writeable
    with pytest.raises(ValueError):
        assignments[0, 0] = 1.0
    np.testing.assert_array_equal(backend.all_costs(q), expected)


def test_hamming_weights_are_cached_and_use_exact_bit_width():
    _hamming_weights.cache_clear()
    first = _hamming_weights(8)
    second = _hamming_weights(8)
    assert first is second
    np.testing.assert_array_equal(first, np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=float))


def test_default_backend_is_auto():
    from rde.backends import default_backend

    assert default_backend() == "auto"


def test_compute_backend_for_size():
    from rde.backends import MLX_GPU_CROSSOVER_N
    from rde.backends.resolve import mlx_usable

    table = load_crossover_table()
    if table is None:
        assert compute_backend_for_size(MLX_GPU_CROSSOVER_N - 1) == "numpy"
        if mlx_usable():
            assert compute_backend_for_size(MLX_GPU_CROSSOVER_N) == "mlx"
    else:
        assert compute_backend_for_size(3, batch_size=1) == "numpy"
        if mlx_usable():
            assert compute_backend_for_size(14, batch_size=1) == "mlx"
            assert compute_backend_for_size(6, batch_size=2048) == "mlx"


def test_full_instance_batch_matches_scalar():
    rng = np.random.default_rng(4)
    indices = [1, 2, 4, 8, 16]
    backend = NumpyBackend()
    for n in (3, 4, 6):
        qs = np.stack([_random_upper_triangular(rng, n) for _ in range(5)], axis=0)
        batch = backend.full_instance_batch(qs, indices)
        assert batch.costs.shape == (5, 2**n)
        for i in range(5):
            q = qs[i]
            costs = backend.all_costs(q)
            w_amp, w_prob, qm, a, b = backend.bound_normalized_weights(q, costs=costs)
            states = backend.trajectory_states(w_amp, indices)
            assert np.allclose(batch.costs[i], costs)
            assert np.allclose(batch.w_amp[i], w_amp)
            assert np.allclose(batch.w_prob[i], w_prob)
            assert batch.qm[i] == qm
            assert batch.a[i] == a
            assert batch.b[i] == b
            for r in indices:
                assert np.allclose(batch.states[r][i], states[r])


def test_mlx_full_instance_batch_if_available():
    if "mlx" not in usable_backends():
        pytest.skip("mlx not available")
    rng = np.random.default_rng(5)
    indices = [1, 2, 4]
    ref = NumpyBackend()
    mlx = get_compute_backend("mlx")
    q = _random_upper_triangular(rng, 4)
    qs = np.stack([q, _random_upper_triangular(rng, 4)], axis=0)
    got = mlx.full_instance_batch(qs, indices)
    expect = ref.full_instance_batch(qs, indices)
    assert np.allclose(got.costs, expect.costs, rtol=1e-5, atol=1e-5)
    assert np.allclose(got.w_amp, expect.w_amp, rtol=1e-5, atol=1e-5)
    for r in indices:
        assert np.allclose(got.states[r], expect.states[r], rtol=1e-5, atol=1e-5)


def test_crossover_table_and_batch_dispatch():
    table = load_crossover_table()
    if table is None:
        pytest.skip("crossover_table.json not shipped yet")
    assert "cells" in table and "best" in table
    assert compute_backend_for_size(6, batch_size=1) in {"numpy", "mlx"}
    assert recommended_batch_size(6) >= 1


def test_export_features_parquet(tmp_path: Path):
    pytest.importorskip("pyarrow")
    reg = toy_registry()
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=1,
            size=4,
            seed=0,
            indices=[1],
            store_root=tmp_path,
            run_id="parquet_test",
        ),
        registry=reg,
    )
    store = Store(tmp_path)
    out = store.export_features_parquet("parquet_test", tmp_path / "out.parquet")
    store.close()
    assert out.exists()
    assert out.stat().st_size > 0
