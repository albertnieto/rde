"""Cross-slice resume scalars kept in the completion index."""

from __future__ import annotations

import json
from typing import Any, Callable

CROSS_SLICE_PREFIXES = ("dynamics.", "overlap.", "update.", "recurrence.")


def extract_cross_slice_scalars(scalars: dict[str, Any]) -> dict[str, float]:
    """Return only cross-slice scalars needed to resume a partial instance."""
    return {
        key: float(value)
        for key, value in scalars.items()
        if any(str(key).startswith(prefix) for prefix in CROSS_SLICE_PREFIXES)
    }


def resume_scalars_json(
    scalars: dict[str, Any],
    *,
    json_default: Callable[[Any], Any] | None = None,
) -> str | None:
    """Serialize cross-slice resume payload, or ``None`` when nothing is needed."""
    slim = extract_cross_slice_scalars(scalars)
    if not slim:
        return None
    return json.dumps(slim, default=json_default)
