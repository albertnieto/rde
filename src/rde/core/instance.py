"""Instance records and stable identifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from rde.io.json_util import json_default


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=json_default)


def compute_instance_id(domain_id: str, size: int, seed: int, params: dict[str, Any]) -> str:
    """Stable content hash for an instance."""
    payload = {"domain_id": domain_id, "size": size, "seed": seed, "params": params}
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass(frozen=True)
class InstanceRecord:
    """Opaque instance parameters plus metadata."""

    domain_id: str
    size: int
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    instance_id: str = field(default="")

    def __post_init__(self) -> None:
        params = dict(self.params)
        object.__setattr__(self, "params", params)
        expected = compute_instance_id(self.domain_id, self.size, self.seed, params)
        if self.instance_id and self.instance_id != expected:
            raise ValueError(
                f"instance_id {self.instance_id!r} does not match content hash {expected!r}"
            )
        object.__setattr__(self, "instance_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "size": self.size,
            "seed": self.seed,
            "params": dict(self.params),
            "instance_id": self.instance_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceRecord:
        return cls(
            domain_id=data["domain_id"],
            size=int(data["size"]),
            seed=int(data["seed"]),
            params=dict(data.get("params", {})),
            instance_id=data.get("instance_id", ""),
        )
