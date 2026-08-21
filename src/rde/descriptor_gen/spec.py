"""Parameterized descriptor template specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


CATALOG_SCHEMA_VERSION = "descriptor-template-v1"


@dataclass(frozen=True)
class DescriptorTemplate:
    """One enumerable measurement family instance (pipeline or discovery)."""

    template_id: str
    family: str
    key: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __str__(self) -> str:
        return self.key

    def canonical_payload(self) -> dict[str, Any]:
        """Return the identity-bearing, order-independent template payload."""
        return {
            "schema": CATALOG_SCHEMA_VERSION,
            "family": self.family,
            "key": self.key,
            "params": self.params,
        }

    @property
    def canonical_id(self) -> str:
        """Stable identity for a descriptor independent of display text."""
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
        return f"{self.family}@{hashlib.sha256(encoded).hexdigest()[:20]}"

    @property
    def provenance(self) -> dict[str, Any]:
        """Machine-readable origin and declared computational contract."""
        catalog = "massive" if self.family.startswith("massive_") else "registered"
        return {
            "catalog": catalog,
            "catalog_schema": CATALOG_SCHEMA_VERSION,
            "source": "rde.descriptor_gen.enumerate_massive"
            if catalog == "massive"
            else "rde.descriptor_gen.enumerate",
            "input": "array_payload",
            "declared_cost": "polynomial_in_input_shape",
            "executable": True,
        }

    def to_record(self) -> dict[str, Any]:
        """Serialize a template with identity and provenance metadata."""
        return {
            "template_id": self.template_id,
            "family": self.family,
            "key": self.key,
            "params": dict(self.params),
            "description": self.description,
            "canonical_id": self.canonical_id,
            "provenance": self.provenance,
        }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)
