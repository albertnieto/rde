"""Unit tests for compression-proxy descriptors."""

from __future__ import annotations

import numpy as np

from rde.features.compress import descriptor
from rde.core.instance import InstanceRecord


def _inst() -> InstanceRecord:
    return InstanceRecord(domain_id="test", size=4, seed=0, params={})


def test_compress_monotonic_with_structure():
    random = np.random.default_rng(0).normal(size=256)
    structured = np.zeros(256)
    structured[::4] = 1.0
    random_out = descriptor(_inst(), None, random)
    structured_out = descriptor(_inst(), None, structured)
    assert structured_out["compress.gzip_bits"] > 0
    assert random_out["compress.gzip_bits"] > 0
    assert structured_out["compress.gzip_bits"] <= random_out["compress.gzip_bits"]


def test_compress_no_bwt_lz77_keys():
    random = np.random.default_rng(0).normal(size=256)
    out = descriptor(_inst(), None, random)
    assert "compress.gzip_bits" in out
    assert "compress.bwt_gzip_bits" not in out
    assert "compress.lz77_tokens" not in out


def test_compress_finite_only_when_array_has_nan():
    import warnings

    mixed = np.array([1.0, float("nan"), 2.0, float("inf"), -3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = descriptor(_inst(), None, mixed)
    assert out["compress.raw_bytes"] > 0
    assert np.isfinite(out["compress.gzip_bits"])


def test_compress_all_nan_returns_empty_payload():
    import warnings

    all_nan = np.array([float("nan"), float("nan")])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = descriptor(_inst(), None, all_nan)
    assert out["compress.raw_bytes"] == 0.0
    assert np.isnan(out["compress.kolmogorov_proxy"])
