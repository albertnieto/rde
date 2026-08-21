"""Observe-layer cache contract for tsp_cost_landscape."""

from __future__ import annotations

import numpy as np

from rde.core.plugins import build_registry
from rde_domains.tsp import landscape as landscape_mod


def test_cost_landscape_prepare_instance_runs_full_landscape_once(monkeypatch):
    calls = {"n": 0}
    real = landscape_mod.full_landscape

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(landscape_mod, "full_landscape", wrapped)
    domain = build_registry("tsp_cost_landscape").get_domain("tsp_cost_landscape")
    [inst] = domain.generate(n=1, size=4, seed=11)
    cache = domain.prepare_instance(inst)
    assert calls["n"] == 1
    sl = domain.materialize(inst, 0, cache=cache)
    feats = domain.primitive_features(inst, cache=cache)
    assert calls["n"] == 1
    np.testing.assert_allclose(sl.values, cache["costs"])
    assert feats["n_cities"] == 4.0
    assert feats["D"].shape == (4, 4)
    assert np.isfinite(feats["degree2_entropy"])


def test_worker_process_instance_runs_full_landscape_once(monkeypatch):
    from rde.runtime.worker import process_instance

    calls = {"n": 0}
    real = landscape_mod.full_landscape

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(landscape_mod, "full_landscape", wrapped)
    reg = build_registry("tsp_cost_landscape")
    domain = reg.get_domain("tsp_cost_landscape")
    [inst] = domain.generate(n=1, size=4, seed=11)
    process_instance(
        domain,
        reg,
        inst,
        indices=[0],
        pending_indices=[0],
        descriptor_names=[],
        metric_names=[],
        run_id="tsp-landscape-cache-contract",
        write_instance=False,
        write_instance_features=True,
        save_arrays=False,
    )
    assert calls["n"] == 1
