"""Deterministic grouped split assignment for RDE v0.3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rde.core.instance import InstanceRecord


class SplitFold(str, Enum):
    CALIBRATION = "calibration"
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    TEST = "test"
    HELDOUT_N = "heldout_n"
    HELDOUT_GENERATOR = "heldout_generator"
    REPLICATION = "replication"


@dataclass(frozen=True)
class SplitPolicy:
    split_seed: int = 0
    discovery_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    held_out_sizes: tuple[int, ...] = ()
    held_out_generator_groups: tuple[str, ...] = ()
    calibration_sizes: tuple[int, ...] = (2,)
    confirmatory_sizes: tuple[int, ...] = (6,)


def _instance_hash(instance_id: str, split_seed: int) -> int:
    digest = hashlib.sha256(f"{split_seed}:{instance_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def assign_instance_fold(
    instance: InstanceRecord,
    policy: SplitPolicy,
    *,
    campaign_seed: int = 0,
) -> SplitFold:
    size = int(instance.size)
    if size in policy.calibration_sizes:
        return SplitFold.CALIBRATION
    if size in policy.confirmatory_sizes:
        return SplitFold.TEST
    if size in policy.held_out_sizes:
        return SplitFold.HELDOUT_N

    gen = str(instance.params.get("generator", ""))
    for group in policy.held_out_generator_groups:
        if gen.startswith(group) or group in gen:
            return SplitFold.HELDOUT_GENERATOR

    h = _instance_hash(instance.instance_id, policy.split_seed + campaign_seed)
    bucket = (h % 1000) / 1000.0
    if bucket < policy.discovery_fraction:
        return SplitFold.DISCOVERY
    if bucket < policy.discovery_fraction + policy.validation_fraction:
        return SplitFold.VALIDATION
    return SplitFold.TEST


def assign_splits(
    instances: list[InstanceRecord],
    policy: SplitPolicy,
    *,
    campaign_seed: int = 0,
) -> dict[str, str]:
    return {
        inst.instance_id: assign_instance_fold(inst, policy, campaign_seed=campaign_seed).value
        for inst in instances
    }


def filter_rows_by_folds(
    rows: list[dict[str, Any]],
    folds: set[str],
) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("split_fold") in folds]


def attach_splits_to_rows(
    rows: list[dict[str, Any]],
    assignments: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        iid = str(row.get("instance_id", ""))
        if iid not in assignments:
            raise ValueError(
                f"row {row_index} has no split assignment for instance_id={iid!r}"
            )
        fold = assignments[iid]
        out.append({**row, "split_fold": fold})
    return out
