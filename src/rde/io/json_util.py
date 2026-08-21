"""Shared JSON serialization helpers for RDE artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def atomic_write(path: Path | str, writer: Callable[[Path], None]) -> None:
    """Write a file beside its destination and replace it atomically.

    Run manifests and array sidecars are resume inputs.  A process crash while
    writing either must leave the previous complete artifact intact rather than
    leave a truncated file at the canonical path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=out.parent,
            prefix=f".{out.name}.",
            suffix=out.suffix,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, out)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> None:
    out = Path(path)
    text = json.dumps(payload, indent=indent, default=json_default)
    atomic_write(out, lambda temporary: temporary.write_text(text, encoding="utf-8"))
