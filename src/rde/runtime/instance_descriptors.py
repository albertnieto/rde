"""Compute descriptors on instance-level primitive arrays."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.features import graph, landscape, matrix
from rde.features.walsh_log import descriptor_for_log_amplitudes

_ALL_INSTANCE_MODULES = frozenset({"landscape", "matrix", "graph", "walsh_log"})


def _classify_primitive(name: str, array: np.ndarray) -> str:
    arr = np.asarray(array)
    lname = name.lower()
    if lname in {"w_amp", "wamp"}:
        return "walsh_log"
    if "cost" in lname or lname.endswith("_c"):
        return "landscape"
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return "matrix_graph"
    if arr.ndim == 1:
        return "landscape"
    return "unknown"


def compute_instance_descriptors(
    instance: InstanceRecord,
    primitive_arrays: dict[str, np.ndarray],
    *,
    allowed_modules: frozenset[str] | None = None,
) -> dict[str, float]:
    """Run matrix/graph/landscape/walsh descriptors on named primitive arrays.

    ``allowed_modules`` restricts which extractors run (e.g. poly-input panels
    should pass ``{"matrix", "graph"}`` and omit oracle ``landscape``).
    ``None`` keeps the historical full instance-descriptor set.
    """
    del instance  # reserved for future instance-conditioned hooks
    allow = _ALL_INSTANCE_MODULES if allowed_modules is None else allowed_modules
    out: dict[str, float] = {}
    for name, array in primitive_arrays.items():
        kind = _classify_primitive(name, array)
        if kind == "landscape":
            if "landscape" in allow:
                out.update(landscape.descriptor_for_vector(array, name=name))
        elif kind == "matrix_graph":
            if "matrix" in allow:
                out.update(matrix.descriptor_for_matrix(array, name=name))
            if "graph" in allow:
                out.update(graph.descriptor_for_matrix(array, name=name))
        elif kind == "walsh_log":
            if "walsh_log" in allow:
                out.update(descriptor_for_log_amplitudes(array, name=name))
    return out
