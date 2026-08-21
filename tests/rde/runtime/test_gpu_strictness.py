"""Tests for strict GPU backend resolution and expression eval policy."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from rde.backends.resolve import GpuBackendError, resolve_compute_backend, resolve_expr_backend
from rde.expression.batch import prepare_device_envs
from rde.expression.ast import Expr


class TestBackendResolveStrict(unittest.TestCase):
    def test_explicit_numpy_allowed(self) -> None:
        self.assertEqual(resolve_compute_backend("numpy"), "numpy")
        self.assertEqual(resolve_expr_backend("numpy"), "numpy")

    @patch("rde.backends.resolve.mlx_usable", return_value=False)
    def test_gpu_alias_raises_without_mlx(self, _mock: object) -> None:
        with self.assertRaises(GpuBackendError):
            resolve_compute_backend("gpu")
        with self.assertRaises(GpuBackendError):
            resolve_expr_backend("metal")

    @patch("rde.backends.resolve.mlx_usable", return_value=True)
    def test_gpu_alias_resolves_to_mlx(self, _mock: object) -> None:
        self.assertEqual(resolve_compute_backend("gpu"), "mlx")
        self.assertEqual(resolve_expr_backend("mps"), "mlx")

    @patch("rde.backends.resolve.mlx_usable", return_value=True)
    def test_auto_respects_crossover_table(self, _mock: object) -> None:
        self.assertEqual(resolve_compute_backend("auto", size=3, batch_size=1), "numpy")
        self.assertEqual(resolve_compute_backend("auto", size=14, batch_size=1), "mlx")
        self.assertEqual(resolve_compute_backend("auto", size=6, batch_size=2048), "mlx")

    @patch("rde.backends.resolve.mlx_usable", return_value=True)
    def test_require_gpu_keeps_mlx_on_small_rows(self, _mock: object) -> None:
        env = {"x": np.array([1.0, 2.0, 3.0])}
        backend, *_rest, reason = prepare_device_envs(
            env,
            "mlx",
            3,
            np.array([0.1, 0.2, 0.3]),
            require_gpu=True,
            requested_backend="mlx",
        )
        self.assertEqual(backend, "mlx")
        self.assertIsNone(reason)


class TestComputeBackendFallbackLogging(unittest.TestCase):
    @patch("rde.backends.resolve.mlx_usable", return_value=False)
    def test_auto_fallback_to_numpy_logs_warning(self, _mock: object) -> None:
        from rde.backends.resolve import compute_backend_for_size

        with self.assertLogs("rde.backends.resolve", level="WARNING") as cm:
            result = compute_backend_for_size(14, batch_size=1)
        self.assertEqual(result, "numpy")
        self.assertTrue(any("compute backend fallback" in msg for msg in cm.output))


class TestTorchMpsStrictDefault(unittest.TestCase):
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_torch_mps_compute_backend_raises_without_mps(self, _mock: object) -> None:
        from rde.backends import get_compute_backend
        from rde.backends.resolve import GpuBackendError

        backend = get_compute_backend("torch_mps")
        with self.assertRaises(GpuBackendError):
            _ = backend.device

    @patch("torch.backends.mps.is_available", return_value=False)
    def test_torch_cpu_compute_backend_unaffected(self, _mock: object) -> None:
        from rde.backends import get_compute_backend

        backend = get_compute_backend("torch_cpu")
        self.assertEqual(backend.device.type, "cpu")


class TestExprDispatch(unittest.TestCase):
    def test_numpy_dispatch_matches_direct_eval(self) -> None:
        from rde.expression.evaluate import evaluate_expr

        env = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        expr = Expr("add", left=Expr.var("a"), right=Expr.var("b"))
        out = evaluate_expr(expr, env)
        np.testing.assert_allclose(out, [4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
