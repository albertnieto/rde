"""Tests for discovery imputation helpers."""

from __future__ import annotations

import numpy as np

from rde.discovery.impute import impute_column


def test_impute_column_fills_nan():
    col = np.array([1.0, np.nan, 3.0])
    np.testing.assert_allclose(impute_column(col), [1.0, 2.0, 3.0])

    all_nan = np.array([np.nan, np.nan])
    np.testing.assert_allclose(impute_column(all_nan), [0.0, 0.0])
