"""Soft-fail helpers so one bad item/stage cannot abort a long RDE run."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

_LOG = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")

# Broad but intentional: multi-hour RDE campaigns must survive one bad
# expression, template, GP tree, or optional sub-analysis.
SOFT_FAIL_EXCEPTIONS = (
    RecursionError,
    MemoryError,
    OverflowError,
    FloatingPointError,
    ValueError,
    KeyError,
    IndexError,
    AttributeError,
    RuntimeError,
    TypeError,
    ImportError,
    OSError,
    ArithmeticError,
    json.JSONDecodeError,
)


def failure_detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def failure_record(
    *,
    stage: str,
    exc: BaseException,
    traceback_limit: int = 12,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": failure_detail(exc),
        "detail": failure_detail(exc),
        "traceback": traceback.format_exc(limit=traceback_limit),
    }


def write_failure_artifact(
    run_dir: Path,
    payload: dict[str, Any],
    *,
    prefix: str = "run_failure",
) -> Path:
    """Persist a failure record and refresh ``{prefix}_latest.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = run_dir / f"{prefix}_{stamp}.json"
    text = json.dumps(payload, indent=2, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    (run_dir / f"{prefix}_latest.json").write_text(text, encoding="utf-8")
    return path


def emit_failure_line(
    *,
    kind: str,
    payload: dict[str, Any],
    artifact: Path | None = None,
) -> None:
    from rde.io.events import log_json_line

    line = {
        "stage": "run_failed",
        "kind": kind,
        "error_type": payload.get("error_type"),
        "error": payload.get("error"),
        "failure_stage": payload.get("stage"),
        "artifact": str(artifact) if artifact is not None else None,
    }
    log_json_line(line)
    _LOG.error("run_failed kind=%s stage=%s error=%s", kind, payload.get("stage"), payload.get("error"))


def soft_call(
    label: str,
    fn: Callable[[], T],
    *,
    fallback: T,
    errors: list[dict[str, Any]] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> T:
    """Run ``fn``; on failure record the error and return ``fallback``."""
    try:
        return fn()
    except SOFT_FAIL_EXCEPTIONS as exc:
        detail = failure_detail(exc)
        _LOG.warning("soft-failed (%s): %s", label, detail)
        if errors is not None:
            errors.append(failure_record(stage=label, exc=exc))
        if on_error is not None:
            on_error(f"{label} soft-failed: {detail}")
        return fallback


def soft_map(
    label: str,
    items: Iterable[T],
    fn: Callable[[T], U],
    *,
    skip_value: U | None = None,
    errors: list[dict[str, Any]] | None = None,
    max_logged_errors: int = 20,
) -> list[U | None]:
    """Map ``fn`` over ``items``, skipping individuals that raise."""
    out: list[U | None] = []
    n_errors = 0
    for i, item in enumerate(items):
        try:
            out.append(fn(item))
        except SOFT_FAIL_EXCEPTIONS as exc:
            n_errors += 1
            if errors is not None and len(errors) < max_logged_errors:
                errors.append(
                    {
                        **failure_record(stage=label, exc=exc, traceback_limit=8),
                        "index": i,
                    }
                )
            elif n_errors == max_logged_errors + 1:
                _LOG.warning(
                    "soft-map (%s): further per-item errors suppressed after %d",
                    label,
                    max_logged_errors,
                )
            out.append(skip_value)
    if n_errors:
        _LOG.warning("soft-map (%s): skipped %d item(s)", label, n_errors)
    return out
