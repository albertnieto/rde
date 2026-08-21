"""Named descriptor profiles for RDE v0.3."""

from __future__ import annotations

from rde.core.registry import Registry

PANEL_POLY_INPUT_V03 = "poly_input_v03"
PANEL_ORACLE_DIAGNOSTICS_V03 = "oracle_diagnostics_v03"
PANEL_LABELS_V03 = "labels_v03"
PANEL_SMOKE_MINIMAL = "smoke_minimal"
PANEL_FULL_CATALOG = "full_catalog"

_POLY_INPUT_MODULES = (
    "matrix",
    "graph",
    "stats",
)

_ORACLE_MODULES = (
    "landscape",
    "spectral",
    "fourier",
    "walsh_log",
    "compress",
    "dynamics",
    "overlap",
    "update",
    "recurrence",
    "slice_rank",
)

_SMOKE_MODULES = ("matrix", "graph")


def resolve_descriptor_panel(
    registry: Registry,
    panel: str,
) -> tuple[list[str], list[str]]:
    """Return (descriptor_names, metric_names) for a named panel."""
    all_desc = registry.list_descriptors()
    all_metrics = registry.list_metrics()

    if panel == PANEL_FULL_CATALOG:
        return list(all_desc), list(all_metrics)

    if panel == PANEL_LABELS_V03:
        return [], list(all_metrics)

    module_map = {
        PANEL_POLY_INPUT_V03: _POLY_INPUT_MODULES,
        PANEL_ORACLE_DIAGNOSTICS_V03: _ORACLE_MODULES,
        PANEL_SMOKE_MINIMAL: _SMOKE_MODULES,
    }
    modules = module_map.get(panel)
    if modules is None:
        raise ValueError(f"Unknown descriptor panel {panel!r}")

    desc = [d for d in all_desc if d.split(".", 1)[0] in modules]
    if panel == PANEL_POLY_INPUT_V03:
        # Labels/targets only — registered metric ids are bare names (not "metric.*").
        metrics = list(all_metrics)
    elif panel == PANEL_ORACLE_DIAGNOSTICS_V03:
        metrics = list(all_metrics)
    else:
        metrics = [m for m in all_metrics if "log_slice_rank" in m or "representation" in m]
    return desc, metrics


def panel_descriptor_modules(panel: str) -> tuple[str, ...]:
    return {
        PANEL_POLY_INPUT_V03: _POLY_INPUT_MODULES,
        PANEL_ORACLE_DIAGNOSTICS_V03: _ORACLE_MODULES,
        PANEL_SMOKE_MINIMAL: _SMOKE_MODULES,
    }.get(panel, ())


def panel_instance_descriptor_modules(panel: str) -> list[str] | None:
    """Instance-array extractors allowed for a panel; None means unrestricted."""
    if panel == PANEL_POLY_INPUT_V03:
        return ["matrix", "graph"]
    if panel == PANEL_SMOKE_MINIMAL:
        return ["matrix", "graph"]
    if panel == PANEL_LABELS_V03:
        return []
    if panel == PANEL_ORACLE_DIAGNOSTICS_V03:
        return ["landscape", "matrix", "graph", "walsh_log"]
    return None


def panel_enable_cross_slice(panel: str) -> bool:
    """Cross-slice dynamics/overlap/update/recurrence are oracle-side diagnostics."""
    if panel in {PANEL_POLY_INPUT_V03, PANEL_SMOKE_MINIMAL, PANEL_LABELS_V03}:
        return False
    return True


def builtin_descriptor_modules(registry: Registry) -> list[str]:
    return registry.list_descriptors()


def generated_template_count(*, pipeline_only: bool = False) -> int:
    from rde.descriptor_gen.enumerate import catalog_size, default_pipeline_templates

    if pipeline_only:
        return len(default_pipeline_templates())
    return catalog_size()


def estimate_keys_per_slice(*, include_gen: bool = True, panel: str = PANEL_POLY_INPUT_V03) -> dict[str, int]:
    """Rough counts of scalar keys emitted per slice."""
    if panel == PANEL_POLY_INPUT_V03:
        hand_estimate = 40
        gen_count = 0
    elif panel == PANEL_ORACLE_DIAGNOSTICS_V03:
        hand_estimate = 80
        gen_count = 0
    elif panel == PANEL_SMOKE_MINIMAL:
        hand_estimate = 20
        gen_count = 0
    else:
        hand_estimate = 120
        from rde.descriptor_gen.enumerate import default_pipeline_templates

        gen_count = len(default_pipeline_templates()) if include_gen else 0
    return {
        "panel": panel,
        "hand_descriptors_estimate": hand_estimate,
        "generated_templates_pipeline": gen_count,
        "total_per_slice_estimate": hand_estimate + gen_count,
        "full_generated_catalog": generated_template_count(),
    }
