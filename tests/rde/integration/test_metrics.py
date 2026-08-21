"""Tests for composite metrics."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.runtime.metrics import compressibility_ratio, representation_complexity
from rde.core.protocols import SimpleFamilySlice


def test_compressibility_ratio():
    d = {"compress.raw_bytes": 1000.0, "compress.gzip_bits": 400.0}
    assert compressibility_ratio(d) == 0.4


def test_representation_complexity_increases_with_spread():
    inst = InstanceRecord(domain_id="t", size=4, seed=0, params={})
    concentrated = SimpleFamilySlice(values=np.array([1.0, 0, 0, 0]), index=1)
    spread = SimpleFamilySlice(values=np.ones(4) / 2.0, index=1)
    d_conc = {
        "spectral.effective_rank": 1.0,
        "compress.raw_bytes": 100.0,
        "compress.gzip_bits": 80.0,
        "fourier.top_mass_fraction": 0.9,
        "stats.participation_ratio": 1.0,
        "array_size": 4.0,
    }
    d_spread = {
        **d_conc,
        "spectral.effective_rank": 3.5,
        "compress.gzip_bits": 95.0,
        "fourier.top_mass_fraction": 0.2,
        "stats.participation_ratio": 0.25,
    }
    assert representation_complexity(inst, spread, d_spread) > representation_complexity(
        inst, concentrated, d_conc
    )
