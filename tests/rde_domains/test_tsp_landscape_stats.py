"""`tsp_landscape_stats`'s `repr.best_complexity` wiring.

Domain-integration gap closure: `rde_domains.tsp.representations` already
ran `rde.representation` against real TSP distance profiles as exploratory
tooling (no contract, no predictor column). This wires the same mechanism
into `TspLandscapeStatsDomain.primitive_features` as a real, contract-tracked
predictor -- same pattern as `hsp_functions`'s `repr.best_complexity`
(`src/rde_domains/hsp_functions/domain.py`).
"""

from __future__ import annotations

import numpy as np

from rde.core.plugins import build_registry
from rde.experiment.gate import structural_columns
from rde_domains.contracts import domain_contract, feature_catalog_for_domain
from rde_domains.tsp.landscape_stats import TspLandscapeStatsDomain


def test_repr_best_complexity_present_and_finite_across_sizes():
    dom = TspLandscapeStatsDomain()
    for size in (4, 6, 9):
        [inst] = dom.generate(n=1, size=size, seed=5)
        cache = dom.prepare_instance(inst)
        feats = dom.primitive_features(inst, cache=cache)
        assert "repr.best_complexity" in feats
        assert np.isfinite(feats["repr.best_complexity"])


def test_repr_best_complexity_is_predictor_eligible_in_contract():
    c = domain_contract("tsp_landscape_stats")
    assert "repr.best_complexity" in [s.key_pattern for s in c.feature_specs]
    cat = feature_catalog_for_domain("tsp_landscape_stats")
    assert cat.is_predictor_eligible("repr.best_complexity")


def test_repr_best_complexity_never_reads_tour_costs():
    """Computed purely from D -- must be identical whether or not `costs` was ever touched."""
    dom = TspLandscapeStatsDomain()
    [inst] = dom.generate(n=1, size=5, seed=9)
    cache = dom.prepare_instance(inst)
    feats_with_costs = dom.primitive_features(inst, cache=cache)

    d = cache["D"]
    fake_cache = {"D": d, "costs": np.full_like(cache["costs"], np.nan)}
    feats_without_real_costs = dom.primitive_features(inst, cache=fake_cache)
    assert feats_with_costs["repr.best_complexity"] == feats_without_real_costs["repr.best_complexity"]


def test_repr_best_complexity_recognized_as_structural_column():
    dom = TspLandscapeStatsDomain()
    instances = dom.generate(n=8, size=5, seed=13)
    rows: list[dict] = []
    for inst in instances:
        cache = dom.prepare_instance(inst)
        feats = dom.primitive_features(inst, cache=cache)
        row = {"size": inst.size, "generator": inst.params["generator"]}
        for key, val in feats.items():
            if isinstance(val, (int, float)):
                row[key] = float(val)
        rows.append(row)
    assert "repr.best_complexity" in structural_columns(rows)


def test_domain_registered_via_plugin_loader():
    reg = build_registry("tsp_landscape_stats")
    dom = reg.get_domain("tsp_landscape_stats")
    assert dom.domain_id == "tsp_landscape_stats"
