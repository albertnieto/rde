"""Tests for named descriptor panels and catalog estimates."""

from __future__ import annotations

import pytest

from rde.features.catalog import (
    PANEL_FULL_CATALOG,
    PANEL_LABELS_V03,
    PANEL_ORACLE_DIAGNOSTICS_V03,
    PANEL_POLY_INPUT_V03,
    PANEL_SMOKE_MINIMAL,
    builtin_descriptor_modules,
    estimate_keys_per_slice,
    panel_descriptor_modules,
    panel_enable_cross_slice,
    panel_instance_descriptor_modules,
    resolve_descriptor_panel,
)
from tests.rde.helpers import toy_registry


def test_resolve_core_panels():
    reg = toy_registry()
    full_desc, full_metrics = resolve_descriptor_panel(reg, PANEL_FULL_CATALOG)
    assert full_desc == reg.list_descriptors()
    assert full_metrics == reg.list_metrics()

    labels_desc, labels_metrics = resolve_descriptor_panel(reg, PANEL_LABELS_V03)
    assert labels_desc == []
    assert labels_metrics

    smoke_desc, _ = resolve_descriptor_panel(reg, PANEL_SMOKE_MINIMAL)
    assert smoke_desc
    assert panel_descriptor_modules(PANEL_POLY_INPUT_V03)
    assert panel_instance_descriptor_modules(PANEL_SMOKE_MINIMAL) == ["matrix", "graph"]
    assert panel_enable_cross_slice(PANEL_POLY_INPUT_V03) is False
    assert panel_enable_cross_slice(PANEL_ORACLE_DIAGNOSTICS_V03) is True

    counts = estimate_keys_per_slice(panel=PANEL_SMOKE_MINIMAL)
    assert counts["panel"] == PANEL_SMOKE_MINIMAL
    assert counts["total_per_slice_estimate"] >= counts["hand_descriptors_estimate"]

    with pytest.raises(ValueError, match="Unknown descriptor panel"):
        resolve_descriptor_panel(reg, "not_a_panel")


def test_oracle_panel_variants():
    reg = toy_registry()
    oracle_desc, oracle_metrics = resolve_descriptor_panel(reg, PANEL_ORACLE_DIAGNOSTICS_V03)
    assert oracle_desc
    assert oracle_metrics

    assert panel_instance_descriptor_modules(PANEL_LABELS_V03) == []
    assert builtin_descriptor_modules(reg) == reg.list_descriptors()
    assert estimate_keys_per_slice(panel=PANEL_ORACLE_DIAGNOSTICS_V03)["hand_descriptors_estimate"] == 80

    full_catalog_counts = estimate_keys_per_slice(panel=PANEL_FULL_CATALOG)
    assert full_catalog_counts["generated_templates_pipeline"] >= 0
