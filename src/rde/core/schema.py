"""JSONL row schema validation (Phase 1)."""

from __future__ import annotations

from typing import Any

FEATURE_ROW_LAYOUT_SLICE_ONLY = "slice-only-v1"

SLICE_DESCRIPTOR_RESERVED_KEYS = frozenset({"family_index", "array_size"})

FEATURE_ROW_REQUIRED = frozenset(
    {
        "run_id",
        "instance_id",
        "domain_id",
        "size",
        "seed",
        "family_index",
        "slice_kind",
        "descriptors",
        "metrics",
    }
)

INSTANCE_ROW_REQUIRED = frozenset({"domain_id", "size", "seed", "params"})

INSTANCE_FEATURES_ROW_REQUIRED = frozenset(
    {
        "run_id",
        "instance_id",
        "domain_id",
        "size",
        "seed",
        "scalars",
    }
)


def validate_feature_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = FEATURE_ROW_REQUIRED - set(row)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if "descriptors" in row and not isinstance(row["descriptors"], dict):
        errors.append("descriptors must be a dict")
    if "metrics" in row and not isinstance(row["metrics"], dict):
        errors.append("metrics must be a dict")
    if "family_index" in row and not isinstance(row["family_index"], int):
        errors.append("family_index must be int")
    return errors


def validate_instance_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = INSTANCE_ROW_REQUIRED - set(row)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if "params" in row and not isinstance(row["params"], dict):
        errors.append("params must be a dict")
    return errors


def validate_instance_features_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = INSTANCE_FEATURES_ROW_REQUIRED - set(row)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if "scalars" in row and not isinstance(row["scalars"], dict):
        errors.append("scalars must be a dict")
    return errors


def validate_instance_features_file(rows: list[dict[str, Any]]) -> list[tuple[int, list[str]]]:
    out: list[tuple[int, list[str]]] = []
    for i, row in enumerate(rows):
        errs = validate_instance_features_row(row)
        if errs:
            out.append((i, errs))
    return out


def validate_features_file(rows: list[dict[str, Any]]) -> list[tuple[int, list[str]]]:
    out: list[tuple[int, list[str]]] = []
    for i, row in enumerate(rows):
        errs = validate_feature_row(row)
        if errs:
            out.append((i, errs))
    return out


def duplicated_instance_scalar_keys(
    descriptors: dict[str, Any],
    instance_scalars: dict[str, Any],
) -> list[str]:
    """Return instance-only scalar keys incorrectly duplicated on a slice row."""
    overlap = sorted(
        set(descriptors.keys()) & set(instance_scalars.keys()) - SLICE_DESCRIPTOR_RESERVED_KEYS
    )
    return overlap
