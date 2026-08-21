"""Generic domain contracts and registration.

RDE core defines the contract shape but does not know any research domain.
Domain packages register their own contracts while their ``rde.domains``
entry-point loader is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points

from rde.core.feature_contract import FeatureCatalog, FeatureSpec


@dataclass(frozen=True)
class StageSizePolicy:
    calibration_sizes: tuple[int, ...] = (2,)
    discovery_sizes: tuple[int, ...] = (3, 4)
    validation_sizes: tuple[int, ...] = (5,)
    confirmatory_sizes: tuple[int, ...] = (6,)
    max_science_size: int = 6


@dataclass(frozen=True)
class DomainContract:
    domain_id: str
    primary_target: str
    secondary_targets: tuple[str, ...]
    feature_specs: tuple[FeatureSpec, ...]
    stage_sizes: StageSizePolicy
    indices: tuple[int, ...]
    generator_id: str | None
    held_out_sizes: tuple[int, ...] = ()
    held_out_generator_groups: tuple[str, ...] = ()
    recurrence_applicable: bool = True
    representation_applicable: bool = True
    encoding_sanity_metrics: tuple[str, ...] = ()


_CONTRACTS: dict[str, DomainContract] = {}


def register_domain_contract(contract: DomainContract) -> None:
    """Register a plugin-owned contract without importing the plugin in core."""
    existing = _CONTRACTS.get(contract.domain_id)
    if existing is not None and existing != contract:
        raise ValueError(f"Domain contract already registered: {contract.domain_id!r}")
    _CONTRACTS[contract.domain_id] = contract


def _load_entrypoint_contract(domain_id: str) -> None:
    """Ask an installed domain plugin to register its contract.

    The plugin entry point also registers the domain itself. A throwaway
    registry is sufficient here because only the shared contract registry is
    needed by direct analysis helpers.
    """
    try:
        from rde.core.registry import Registry

        for ep in entry_points(group="rde.domains"):
            if ep.name == domain_id:
                ep.load()(Registry())
                return
    except Exception:
        return


def domain_contract(domain_id: str) -> DomainContract:
    if domain_id not in _CONTRACTS:
        _load_entrypoint_contract(domain_id)
    try:
        return _CONTRACTS[domain_id]
    except KeyError as exc:
        raise KeyError(f"No domain contract registered for {domain_id!r}") from exc


def feature_catalog_for_domain(domain_id: str) -> FeatureCatalog:
    contract = domain_contract(domain_id)
    return FeatureCatalog(domain_id=domain_id, specs=contract.feature_specs)


def n_per_size_v03(domain_id: str, size: int) -> int:
    """Return a conservative staged count for a registered domain."""
    contract = domain_contract(domain_id)
    if size not in (
        *contract.stage_sizes.calibration_sizes,
        *contract.stage_sizes.discovery_sizes,
        *contract.stage_sizes.validation_sizes,
        *contract.stage_sizes.confirmatory_sizes,
    ):
        return 0
    if size <= 4:
        return 200
    if size <= 6:
        return 300
    return 200
