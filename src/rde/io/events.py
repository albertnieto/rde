"""Durable structured event logging for RDE runs and campaigns.

Stdout contract
---------------
* ``log_progress`` — tee-safe live progress (message-only, flushed stdout).
* ``log_json_line`` / ``log_stage_event`` — one JSON object per line on stdout.
* ``configure_logging`` / ``event`` — durable JSONL under ``runs/<id>/events.jsonl``.

Rich/TTY rendering in ``rde.io.console`` intentionally keeps direct ``print`` calls.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROGRESS_LOGGER: logging.Logger | None = None
_STDOUT_JSON_LOGGER: logging.Logger | None = None


class FlushingStreamHandler(logging.StreamHandler):
    """Stream handler that flushes after every record (tee-safe progress)."""

    def __init__(self, stream: object | None = None) -> None:
        super().__init__(stream)
        self._follow_stdout = stream is None or stream is sys.stdout

    def emit(self, record: logging.LogRecord) -> None:
        stream = sys.stdout if self._follow_stdout else self.stream
        if stream is None or getattr(stream, "closed", False):
            return
        try:
            msg = self.format(record)
            stream.write(msg + self.terminator)
            stream.flush()
        except (ValueError, OSError):
            self.handleError(record)


def _message_only_logger(name: str, stream: object | None = None) -> logging.Logger:
    """Return a logger that writes only the message, flushed to ``stream``."""
    logger = logging.getLogger(name)
    marker = "_rde_message_only"
    if getattr(logger, marker, False):
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = FlushingStreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    setattr(logger, marker, True)
    return logger


def progress_logger() -> logging.Logger:
    """Logger for tee-safe live progress lines on stdout."""
    global _PROGRESS_LOGGER
    if _PROGRESS_LOGGER is None:
        _PROGRESS_LOGGER = _message_only_logger("rde.progress")
    return _PROGRESS_LOGGER


def stdout_json_logger() -> logging.Logger:
    """Logger for one JSON object per line on stdout (experiment stage events)."""
    global _STDOUT_JSON_LOGGER
    if _STDOUT_JSON_LOGGER is None:
        _STDOUT_JSON_LOGGER = _message_only_logger("rde.stdout.json")
    return _STDOUT_JSON_LOGGER


def log_progress(message: str) -> None:
    """Emit a flushed progress line visible under ``tee`` and CI redirects."""
    progress_logger().info(message)


def log_json_line(payload: dict[str, Any]) -> None:
    """Emit one flushed JSON object on stdout."""
    stdout_json_logger().info(json.dumps(payload, default=str))


def append_jsonl(path: Path | str, payload: dict[str, Any]) -> str:
    """Append one JSON line to a durable log and return the serialized line."""
    line = json.dumps(payload, default=str)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return line


def log_stage_event(
    payload: dict[str, Any],
    *,
    jsonl_path: Path | str | None = None,
) -> None:
    """Log a structured stage event to stdout and optionally append to JSONL."""
    line = append_jsonl(jsonl_path, payload) if jsonl_path is not None else json.dumps(payload, default=str)
    stdout_json_logger().info(line)


class JsonEventFormatter(logging.Formatter):
    """Serialize log records as one searchable JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "run_id", "stage", "instance_id", "backend"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(
    *,
    event_path: Path | str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the RDE logger once, optionally with a durable JSONL sink."""
    logger = logging.getLogger("rde")
    logger.setLevel(level)
    logger.propagate = False
    marker = "_rde_event_handler"
    if event_path is not None:
        path = Path(event_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for handler in list(logger.handlers):
            if getattr(handler, marker, False):
                logger.removeHandler(handler)
                handler.close()
        handler = logging.FileHandler(path, encoding="utf-8")
        setattr(handler, marker, True)
        handler.setFormatter(JsonEventFormatter())
        logger.addHandler(handler)
    elif not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def configure_run_logging(
    store_root: Path | str,
    run_id: str,
    *,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the event stream at ``runs/<run_id>/events.jsonl``."""
    root = Path(store_root)
    return configure_logging(event_path=root / "runs" / run_id / "events.jsonl", level=level)


def event(
    logger: logging.Logger,
    event_name: str,
    message: str | None = None,
    *,
    run_id: str | None = None,
    stage: str | None = None,
    level: int = logging.INFO,
    exc_info: Any = None,
    **fields: Any,
) -> None:
    """Emit a structured event while retaining normal logger semantics."""
    logger.log(
        level,
        message or event_name,
        exc_info=exc_info,
        extra={
            "event": event_name,
            "run_id": run_id,
            "stage": stage,
            "fields": fields,
        },
    )
