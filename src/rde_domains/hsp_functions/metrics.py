"""Domain metric for hsp_functions: exposes the generation-time
planted-structure ground truth as `metric.structure_strength`, matching
the `metric.*`-target convention every other RDE v0.3 domain contract
uses (see `rde.core.domain_contract`).
"""

from __future__ import annotations

from rde.core.protocols import QueryIntent
from rde.core.registry import Registry


def register_hsp_functions_metrics(registry: Registry) -> None:
    def structure_strength(instance, slice_, descriptors: dict) -> float:
        del instance, slice_
        if "structure_strength" not in descriptors:
            raise KeyError("hsp_functions metrics require domain-prepared 'structure_strength'")
        return float(descriptors["structure_strength"])

    # Registered under the bare name -- the pipeline prefixes registered
    # metric names with "metric." itself when flattening rows (confirmed
    # empirically: registering "metric.structure_strength" here produced
    # "metric.metric.structure_strength" in the row). Every other domain's
    # metrics.py in this repo follows the same bare-name convention (e.g.
    # coined_walk registers "E_peak", not "metric.E_peak").
    registry.register_metric_fn("structure_strength", QueryIntent.EVALUATE, structure_strength)

    def algorithm_class(instance, slice_, descriptors: dict) -> float:
        del slice_
        if "algorithm_class" in descriptors:
            return float(descriptors["algorithm_class"])
        from rde_domains.hsp_functions.kind_screen import algorithm_class_for_generator

        gen = str(instance.params.get("generator") or instance.params.get("family") or "")
        return float(algorithm_class_for_generator(gen))

    registry.register_metric_fn("algorithm_class", QueryIntent.EVALUATE, algorithm_class)
