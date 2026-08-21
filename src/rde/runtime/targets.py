"""Resolve discovery targets from a domain contract."""

from __future__ import annotations

from pathlib import Path

from rde.io.store import Store

def default_target_for_domain(domain_id: str) -> str:
    try:
        from rde.core.domain_contract import domain_contract

        return domain_contract(domain_id).primary_target
    except KeyError:
        return "metric.representation_complexity"


def resolve_target_for_run(
    run_id: str,
    store_root: Path | str,
    *,
    override: str | None = None,
) -> str:
    if override:
        return override
    manifest = Store(store_root).read_manifest(run_id)
    return default_target_for_domain(manifest.domain_id)
