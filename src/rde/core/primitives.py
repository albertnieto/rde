"""Split and persist domain primitive features."""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.io.store import Store


def split_primitives(
    primitives: dict[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Separate scalar primitives from array payloads."""
    scalars: dict[str, float] = {}
    arrays: dict[str, np.ndarray] = {}
    for key, value in primitives.items():
        if isinstance(value, (bool, np.bool_)):
            scalars[key] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            scalars[key] = float(value)
        elif isinstance(value, np.ndarray):
            arrays[key] = np.asarray(value)
    return scalars, arrays


def persist_instance_features(
    store: Store,
    run_id: str,
    instance_id: str,
    domain_id: str,
    size: int,
    seed: int,
    primitives: dict[str, Any],
    *,
    instance_descriptors: dict[str, float] | None = None,
    save_arrays: bool = True,
) -> dict[str, Any]:
    """Write one instance_features row; array primitives go to NPZ sidecars."""
    scalars, arrays = split_primitives(primitives)
    if instance_descriptors:
        scalars.update(instance_descriptors)
    array_refs: dict[str, str] = {}

    if save_arrays:
        for name, array in arrays.items():
            path = store.save_array(run_id, instance_id, f"primitive_{name}", array)
            array_refs[name] = str(path.relative_to(store.run_dir(run_id)))

    row: dict[str, Any] = {
        "run_id": run_id,
        "instance_id": instance_id,
        "domain_id": domain_id,
        "size": size,
        "seed": seed,
        "scalars": scalars,
    }
    if array_refs:
        row["array_refs"] = array_refs

    store.append_instance_features(run_id, row)
    return row
