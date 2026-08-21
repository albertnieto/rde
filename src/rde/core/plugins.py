"""Domain and generator plugin loading.

The standalone RDE package ships only reference toy domains. All other
domains, including repository-specific adapters, are loaded through the
``rde.domains`` entry-point group.
"""

from __future__ import annotations

from typing import Callable, Mapping

from rde.features import register_builtin_descriptors, register_builtin_metrics
from rde.generators import register_builtin_generators
from rde.backends.resolve import resolve_compute_backend
from rde.core.registry import Registry

_BUILTIN_LOADERS: dict[str, Callable[..., None]] = {}


def build_registry(
    domain_id: str,
    *,
    compute_backend: str | None = None,
    max_bruteforce_n: int | None = 14,
    loader_kwargs: Mapping[str, object] | None = None,
) -> Registry:
    """Load domain plugin and register builtins."""
    backend = resolve_compute_backend(compute_backend)
    register_builtin_generators()
    reg = Registry()
    register_builtin_descriptors(reg)
    register_builtin_metrics(reg)
    kwargs: dict[str, object] = {
        "compute_backend": backend,
        "max_bruteforce_n": max_bruteforce_n,
    }
    if loader_kwargs:
        kwargs.update(loader_kwargs)
    loader = _BUILTIN_LOADERS.get(domain_id)
    if loader is not None:
        loader(reg, **kwargs)
    elif not try_load_entrypoint_domain(domain_id, reg, kwargs):
        raise KeyError(
            f"Unknown domain: {domain_id!r}. Known: {list_domain_ids()} "
            "(plus setuptools entry points group 'rde.domains')"
        )
    return reg


def registry_loader_kwargs(
    domain_id: str,
    *,
    compute_backend: str | None = None,
    max_bruteforce_n: int | None = 14,
    loader_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Keyword args for ``build_registry`` from pipeline/campaign config."""
    backend = resolve_compute_backend(compute_backend)
    return {
        "compute_backend": backend,
        "max_bruteforce_n": max_bruteforce_n,
        "loader_kwargs": dict(loader_kwargs or {}),
    }


def list_domain_ids() -> list[str]:
    ids = set(_BUILTIN_LOADERS)
    try:
        from importlib.metadata import entry_points

        ids.update(ep.name for ep in entry_points(group="rde.domains"))
    except Exception:
        pass
    return sorted(ids)


def _load_synthetic_poly(reg: Registry, **_kw) -> None:
    from rde.testing import SyntheticPolyDomain

    reg.register_domain(SyntheticPolyDomain())


def _load_block_separable(reg: Registry, **_kw) -> None:
    from rde.testing import BlockSeparableDomain

    reg.register_domain(BlockSeparableDomain())


_BUILTIN_LOADERS["synthetic_poly"] = _load_synthetic_poly
_BUILTIN_LOADERS["block_separable"] = _load_block_separable


def try_load_entrypoint_domain(
    domain_id: str,
    reg: Registry,
    loader_kwargs: dict[str, object] | None = None,
) -> bool:
    """Load domain from setuptools entry point group 'rde.domains' if registered."""
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="rde.domains")
    except Exception:
        return False
    for ep in eps:
        if ep.name == domain_id:
            fn = ep.load()
            fn(reg, **(loader_kwargs or {}))
            return True
    return False
