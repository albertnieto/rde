"""Background heartbeat while pipeline workers are in flight."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


ProgressCallback = Callable[[int, int, str], None]


class StageHeartbeat:
    """Emit a live pulse while a non-iterative stage is running."""

    def __init__(
        self,
        callback: ProgressCallback | None,
        *,
        label: str,
        total: int = 0,
        interval_s: float = 5.0,
    ) -> None:
        self._callback = callback
        self._label = label
        self._total = total
        self._interval = interval_s
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the daemon pulse thread and report the stage immediately."""
        if self._callback is None:
            return
        self._callback(0, self._total, f"{self._label}: started")
        self._thread = threading.Thread(
            target=self._loop,
            name="rde-stage-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, detail: str | None = None) -> None:
        """Stop the pulse thread and optionally report completion."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._callback is not None and detail is not None:
            self._callback(self._total, self._total, f"{self._label}: {detail}")

    def _loop(self) -> None:
        from rde.io.console import format_duration

        while not self._stop.wait(self._interval):
            elapsed = format_duration(time.monotonic() - self._started)
            if self._callback is not None:
                self._callback(
                    0,
                    self._total,
                    f"{self._label}: still running ({elapsed} elapsed)",
                )


def throttled_progress(
    callback: ProgressCallback | None,
    *,
    min_interval_s: float = 1.0,
    every_n: int | None = None,
) -> ProgressCallback | None:
    """Throttle count callbacks by count or elapsed wall-clock time."""
    if callback is None:
        return None
    if min_interval_s < 0:
        raise ValueError("min_interval_s must be non-negative")
    if every_n is not None and every_n < 1:
        raise ValueError("every_n must be positive when provided")

    lock = threading.Lock()
    last_done = -1
    last_emit = 0.0

    def _wrapped(done: int, total: int, detail: str) -> None:
        nonlocal last_done, last_emit
        now = time.monotonic()
        with lock:
            terminal = total > 0 and done >= total
            first = last_done < 0
            count_due = every_n is not None and done - last_done >= every_n
            time_due = now - last_emit >= min_interval_s
            if not (first or terminal or count_due or time_due):
                return
            last_done = done
            last_emit = now
        callback(done, total, detail)

    return _wrapped


class ProgressHeartbeat:
    """Emit periodic progress pulses when instances take a long time."""

    def __init__(self, progress: Any, *, total_jobs: int, interval_s: float = 5.0) -> None:
        self._progress = progress
        self._total = total_jobs
        self._interval = interval_s
        self._completed = 0
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._last_done = self._started
        self._thread: threading.Thread | None = None

    def mark_done(self) -> None:
        self._completed += 1
        self._last_done = time.monotonic()

    def start(self) -> None:
        if self._progress is None or not hasattr(self._progress, "on_heartbeat"):
            return
        self._thread = threading.Thread(target=self._loop, name="rde-progress-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        from rde.io.console import format_duration

        while not self._stop.wait(self._interval):
            in_flight = self._total - self._completed
            if in_flight <= 0:
                continue
            elapsed = time.monotonic() - self._started
            since_done = time.monotonic() - self._last_done
            if self._completed == 0:
                detail = (
                    f"first instance running ({format_duration(elapsed)} elapsed; "
                    f"MLX/GPU compile + descriptor panel on cold start)"
                )
            else:
                detail = (
                    f"{in_flight} in flight, last finished {format_duration(since_done)} ago"
                )
            self._progress.on_heartbeat(
                completed=self._completed,
                total=self._total,
                detail=detail,
            )
