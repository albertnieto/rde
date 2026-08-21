"""Entry-point registrations for HSP and TSP domain adapters."""

from __future__ import annotations

from typing import Any

from rde.core.registry import Registry
from rde_domains import contracts as _contracts


def _register_contract(domain_id: str) -> None:
    _contracts.domain_contract(domain_id)


def register_tsp_clustered(
    reg: Registry, *, max_bruteforce_n: int | None = None, **_kwargs: Any
) -> None:
    from rde_domains.tsp import DEFAULT_MAX_BRUTEFORCE_N, clustered_domain

    cap = DEFAULT_MAX_BRUTEFORCE_N if max_bruteforce_n is None else max_bruteforce_n
    reg.register_domain(clustered_domain(max_bruteforce_n=cap))


def register_tsp_uniform_control(
    reg: Registry, *, max_bruteforce_n: int | None = None, **_kwargs: Any
) -> None:
    from rde_domains.tsp import DEFAULT_MAX_BRUTEFORCE_N, uniform_control_domain

    cap = DEFAULT_MAX_BRUTEFORCE_N if max_bruteforce_n is None else max_bruteforce_n
    reg.register_domain(uniform_control_domain(max_bruteforce_n=cap))


def register_tsp_circulant_symmetry(reg: Registry, **_kwargs: Any) -> None:
    from rde_domains.tsp import circulant_symmetry_domain

    reg.register_domain(circulant_symmetry_domain())


def register_tsp_cost_landscape(reg: Registry, **_kwargs: Any) -> None:
    from rde_domains.tsp import cost_landscape_domain

    reg.register_domain(cost_landscape_domain())


def register_tsp_landscape_stats(
    reg: Registry, *, max_bruteforce_n: int | None = None, **_kwargs: Any
) -> None:
    _register_contract("tsp_landscape_stats")
    from rde_domains.tsp import DEFAULT_MAX_BRUTEFORCE_N, landscape_stats_domain
    from rde_domains.tsp.landscape_stats_metrics import register_landscape_stats_metrics

    cap = DEFAULT_MAX_BRUTEFORCE_N if max_bruteforce_n is None else max_bruteforce_n
    reg.register_domain(landscape_stats_domain(max_bruteforce_n=cap))
    register_landscape_stats_metrics(reg)


def register_hsp_functions(reg: Registry, **_kwargs: Any) -> None:
    _register_contract("hsp_functions")
    from rde_domains.hsp_functions import hsp_functions_domain
    from rde_domains.hsp_functions.metrics import register_hsp_functions_metrics

    reg.register_domain(hsp_functions_domain())
    register_hsp_functions_metrics(reg)
