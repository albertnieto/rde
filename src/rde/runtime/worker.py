"""Single-instance pipeline worker (picklable for multiprocessing)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import inspect

from rde.features.dynamics import compute_r_dynamics
from rde.features.overlap import compute_overlap_descriptors
from rde.features.recurrence import estimate_recurrence_order
from rde.features.update import compute_update_descriptors
from rde.core.instance import InstanceRecord
from rde.io.completion_cache import extract_cross_slice_scalars
from rde.runtime.instance_descriptors import compute_instance_descriptors
from rde.core.primitives import split_primitives
from rde.core.registry import Registry


@dataclass
class InstanceWorkResult:
    instance: InstanceRecord
    write_instance: bool
    instance_features_row: dict[str, Any] | None
    feature_rows: list[dict[str, Any]]
    array_payloads: list[tuple[str, str, np.ndarray]]
    pending_count: int
    skipped_slice_count: int
    all_slices_complete: bool


def _materialize_indices(domain: Any, instance: InstanceRecord, indices: list[int], cache: dict[str, Any] | None):
    if hasattr(domain, "materialize_indices"):
        method = domain.materialize_indices
        if "cache" in inspect.signature(method).parameters:
            return method(instance, indices, cache=cache)
        return method(instance, indices)
    method = domain.materialize
    if "cache" in inspect.signature(method).parameters:
        return {idx: method(instance, idx, cache=cache) for idx in indices}
    return {idx: method(instance, idx) for idx in indices}


def _extract_cross_slice(scalars: dict[str, float]) -> dict[str, float]:
    return extract_cross_slice_scalars(scalars)


def _cross_slice_complete(cross_slice: dict[str, float], indices: list[int]) -> bool:
    return cross_slice.get("dynamics.n_points") == float(len(indices))


def process_instance(
    domain: Any,
    registry: Registry,
    instance: InstanceRecord,
    *,
    indices: list[int],
    pending_indices: list[int],
    descriptor_names: list[str],
    metric_names: list[str],
    run_id: str,
    write_instance: bool,
    write_instance_features: bool,
    save_arrays: bool,
    cached_instance_scalars: dict[str, float] | None = None,
    prebuilt_cache: dict[str, Any] | None = None,
    instance_descriptor_modules: list[str] | None = None,
    enable_cross_slice: bool = True,
) -> InstanceWorkResult:
    if prebuilt_cache is not None:
        cache = prebuilt_cache
    elif hasattr(domain, "prepare_instance"):
        cache = domain.prepare_instance(instance, indices=indices)
    else:
        cache = None
    if hasattr(domain, "primitive_features"):
        primitive = domain.primitive_features
        prim = (
            primitive(instance, cache=cache)
            if "cache" in inspect.signature(primitive).parameters
            else primitive(instance)
        )
    else:
        prim = {}
    prim_scalars, prim_arrays = split_primitives(prim)
    allowed = (
        frozenset(instance_descriptor_modules)
        if instance_descriptor_modules is not None
        else None
    )
    inst_desc = compute_instance_descriptors(
        instance, prim_arrays, allowed_modules=allowed
    )

    cross_slice: dict[str, float] = {}
    slice_map: dict[int, Any] = {}

    if enable_cross_slice and cached_instance_scalars:
        cached = _extract_cross_slice(cached_instance_scalars)
        if _cross_slice_complete(cached, indices):
            cross_slice = cached

    if pending_indices:
        if enable_cross_slice and not cross_slice:
            materialize = list(indices)
            slice_map = _materialize_indices(domain, instance, materialize, cache)
            slice_values = [slice_map[i].values for i in indices]
            w_amp = None
            if cache and "w_amp" in cache:
                w_amp = cache["w_amp"]
            elif isinstance(prim_arrays.get("w_amp"), np.ndarray):
                w_amp = prim_arrays["w_amp"]
            cross_slice = {
                **compute_r_dynamics(list(indices), slice_values),
                **compute_overlap_descriptors(list(indices), slice_values),
                **compute_update_descriptors(list(indices), slice_values, w_amp=w_amp),
                **estimate_recurrence_order(list(indices), slice_values),
            }
        else:
            slice_map = _materialize_indices(domain, instance, list(pending_indices), cache)

    gen_id = instance.params.get("generator")

    instance_features_row = None
    if write_instance_features:
        scalars = dict(prim_scalars)
        scalars.update(inst_desc)
        scalars.update(cross_slice)
        if gen_id is not None:
            scalars["generator"] = str(gen_id)
        array_refs: dict[str, str] = {}
        if save_arrays:
            for name, array in prim_arrays.items():
                array_refs[name] = f"arrays/{instance.instance_id}/primitive_{name}.npz"
        instance_features_row = {
            "run_id": run_id,
            "instance_id": instance.instance_id,
            "domain_id": instance.domain_id,
            "size": instance.size,
            "seed": instance.seed,
            "scalars": scalars,
        }
        if array_refs:
            instance_features_row["array_refs"] = array_refs

    feature_rows: list[dict[str, Any]] = []
    array_payloads: list[tuple[str, str, np.ndarray]] = []

    if write_instance_features and save_arrays:
        for name, array in prim_arrays.items():
            array_payloads.append((instance.instance_id, f"primitive_{name}", array))

    # Metrics may consume cached primitive arrays (for example a domain's
    # already-enumerated QUBO costs) in addition to scalar descriptors.
    metric_context = {**prim_scalars, **prim_arrays, **inst_desc, **cross_slice}

    for index in pending_indices:
        slice_ = slice_map[index]
        descriptors: dict[str, float] = {
            "family_index": float(index),
            "array_size": float(slice_.values.size),
        }

        for desc_name in descriptor_names:
            desc = registry.get_descriptor(desc_name)
            if "context" in inspect.signature(desc.compute).parameters:
                descriptors.update(
                    desc.compute(
                        instance,
                        slice_,
                        slice_.values,
                        context=metric_context,
                    )
                )
            else:
                descriptors.update(desc.compute(instance, slice_, slice_.values))

        metric_scores: dict[str, float] = {}
        scoring_context = {**metric_context, **descriptors}
        for metric_name in metric_names:
            metric = registry.get_metric(metric_name)
            metric_scores[metric_name] = float(
                metric.score(instance, slice_, scoring_context)
            )

        row: dict[str, Any] = {
            "run_id": run_id,
            "instance_id": instance.instance_id,
            "domain_id": instance.domain_id,
            "size": instance.size,
            "seed": instance.seed,
            "family_index": index,
            "slice_kind": slice_.kind,
            "descriptors": descriptors,
            "metrics": metric_scores,
        }
        if gen_id is not None:
            row["generator"] = str(gen_id)
        array_name = f"{slice_.kind}_r{index}"
        if save_arrays:
            row["array_ref"] = f"arrays/{instance.instance_id}/{array_name}.npz"
            array_payloads.append((instance.instance_id, array_name, slice_.values))
        feature_rows.append(row)

    return InstanceWorkResult(
        instance=instance,
        write_instance=write_instance,
        instance_features_row=instance_features_row,
        feature_rows=feature_rows,
        array_payloads=array_payloads,
        pending_count=len(pending_indices),
        skipped_slice_count=len(indices) - len(pending_indices),
        all_slices_complete=bool(pending_indices),
    )


def _worker_init(
    domain_id: str,
    compute_backend: str,
    max_bruteforce_n: int | None,
    *,
    coin_shift_grammar: bool = False,
) -> tuple[Any, Registry]:
    from rde.core.plugins import build_registry, registry_loader_kwargs

    reg = build_registry(
        domain_id,
        **registry_loader_kwargs(
            domain_id,
            compute_backend=compute_backend,
            max_bruteforce_n=max_bruteforce_n,
            loader_kwargs={"coin_shift_grammar": coin_shift_grammar},
        ),
    )
    return reg.get_domain(domain_id), reg


def process_instance_worker(payload: dict[str, Any]) -> InstanceWorkResult:
    domain, registry = _worker_init(
        payload["domain_id"],
        payload.get("compute_backend", "numpy"),
        payload.get("max_bruteforce_n"),
        coin_shift_grammar=bool(payload.get("coin_shift_grammar", False)),
    )
    instance = InstanceRecord.from_dict(payload["instance"])
    return process_instance(
        domain,
        registry,
        instance,
        indices=payload["indices"],
        pending_indices=payload["pending_indices"],
        descriptor_names=payload["descriptor_names"],
        metric_names=payload["metric_names"],
        run_id=payload["run_id"],
        write_instance=payload["write_instance"],
        write_instance_features=payload["write_instance_features"],
        save_arrays=payload["save_arrays"],
        cached_instance_scalars=payload.get("cached_instance_scalars"),
        prebuilt_cache=payload.get("prebuilt_cache"),
        instance_descriptor_modules=payload.get("instance_descriptor_modules"),
        enable_cross_slice=payload.get("enable_cross_slice", True),
    )
