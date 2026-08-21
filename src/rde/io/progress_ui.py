"""Bridge RDE console UI to pipeline progress callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from rde.io.console import RdeConsole, TaskProgress, format_duration
from rde.io.events import log_progress
from rde.runtime.progress import InstanceProgress, PipelineProgressCallback


@dataclass
class ConsolePipelineProgress:
    """Live progress bar for a single pipeline run."""

    console: RdeConsole
    label: str = ""
    _task_ctx: Any = field(default=None, repr=False)
    _task: TaskProgress | None = field(default=None, repr=False)

    def on_run_start(self, *, run_id: str, total_jobs: int, size: int, backend: str) -> None:
        self.label = f"N={size}  {backend}"
        if total_jobs == 0:
            self.console.set_campaign_run(note="complete (resume — nothing to do)")
            return
        if backend in {"mlx", "torch_mps", "torch"}:
            warmup = "first instance may be slow (GPU warmup)"
        elif backend == "numpy":
            warmup = "CPU numpy backend"
        else:
            warmup = f"backend={backend}"
        self.console.set_campaign_run(note=f"{total_jobs} instances queued · {warmup}")
        self._task_ctx = self.console.task_progress(self.label, total_jobs)
        self._task = self._task_ctx.__enter__()
        self.console.register_abort_hook(self.abort)

    def abort(self) -> None:
        """Close an in-flight progress bar without marking the run complete."""
        if self._task_ctx is None:
            return
        try:
            self._task_ctx.__exit__(None, None, None)
        except Exception:
            pass
        self._task_ctx = None
        self._task = None
        self.console.set_campaign_run(note="interrupted")

    def on_heartbeat(self, *, completed: int, total: int, detail: str) -> None:
        if self._task is None:
            return
        self._task.update(
            0,
            detail=f"[{completed}/{total}]  {detail}",
            pulse=True,
        )

    def on_instance_done(self, event: InstanceProgress) -> None:
        if self._task is None:
            return
        short_id = event.instance_id[:12]
        detail = (
            f"[{event.completed}/{event.total}]  "
            f"id={short_id}  slices={event.pending_slices}  rows={event.feature_rows}"
        )
        self._task.update(1, detail=detail)

    def on_run_complete(self, *, run_id: str, feature_rows: int, skipped_instances: int) -> None:
        if self._task_ctx is not None:
            self._task_ctx.__exit__(None, None, None)
            self._task_ctx = None
            self._task = None
            self.console.unregister_abort_hook(self.abort)
        skip = f"  skip={skipped_instances}" if skipped_instances else ""
        self.console.set_campaign_run(
            note=f"done · {feature_rows} ledger rows{skip}",
            progress="",
        )


@dataclass
class PlainPipelineProgress:
    """Append-only progress sink for plain and redirected output.

    Instance completions are throttled to a real five-second heartbeat so a
    large campaign remains readable under ``tee`` while still reporting the
    completed work, rate, elapsed time, and ETA.
    """

    run_id: str = ""
    label: str = ""
    _started_at: float = field(default=0.0, init=False, repr=False)
    _last_emit_at: float = field(default=0.0, init=False, repr=False)
    _total: int = field(default=0, init=False, repr=False)

    def on_run_start(self, *, run_id: str, total_jobs: int, size: int, backend: str) -> None:
        self.run_id = run_id
        self.label = f"N={size} {backend}"
        self._started_at = time.monotonic()
        self._last_emit_at = 0.0
        self._total = max(0, int(total_jobs))
        log_progress(
            f"[pipeline {run_id}] {total_jobs} instances queued "
            f"(backend={backend}) elapsed=0s eta=unknown",
        )

    def on_heartbeat(self, *, completed: int, total: int, detail: str) -> None:
        self._emit(completed, total, detail, force=True)

    def on_instance_done(self, event: InstanceProgress) -> None:
        self._emit(
            event.completed,
            event.total,
            f"id={event.instance_id[:12]} slices={event.pending_slices} rows={event.feature_rows}",
        )

    def on_run_complete(self, *, run_id: str, feature_rows: int, skipped_instances: int) -> None:
        skip = f" skip={skipped_instances}" if skipped_instances else ""
        elapsed = self._elapsed()
        log_progress(
            f"[pipeline {run_id}] complete rows={feature_rows}{skip} "
            f"elapsed={format_duration(elapsed)} eta=0s",
        )

    def abort(self) -> None:
        log_progress(f"[pipeline {self.run_id}] interrupted")

    def _elapsed(self) -> float:
        if self._started_at <= 0:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    def _emit(self, completed: int, total: int, detail: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and completed < total and now - self._last_emit_at < 5.0:
            return
        elapsed = self._elapsed()
        rate = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
        eta = (total - completed) / rate if rate > 0 and total > completed else 0.0
        eta_label = format_duration(eta) if rate > 0 else "unknown"
        percent = 100.0 * completed / total if total > 0 else 100.0
        log_progress(
            f"[pipeline {self.run_id}] [{completed}/{total}] {percent:5.1f}% "
            f"elapsed={format_duration(elapsed)} eta={eta_label} {detail}",
        )
        self._last_emit_at = now


@dataclass
class ConsoleCampaignProgress:
    """Campaign-level header + per-size pipeline bars."""

    console: RdeConsole
    campaign_id: str
    total_sizes: int
    _size_index: int = 0
    _current: ConsolePipelineProgress | None = field(default=None, repr=False)

    def begin_size(self, size: int, index: int, run_id: str) -> ConsolePipelineProgress:
        self._size_index = index
        self.console.set_campaign_run(
            replace=True,
            size=size,
            step=f"{index}/{self.total_sizes}",
            run_id=run_id,
            elapsed="",
            eta="",
            progress="",
            note="starting…",
        )
        self._current = ConsolePipelineProgress(self.console)
        return self._current

    def end_size(self, *, feature_rows: int, elapsed_s: float, eta_s: float) -> None:
        self.console.set_campaign_run(
            elapsed=format_duration(elapsed_s),
            eta=format_duration(eta_s) if eta_s > 0 else "",
            note=f"size complete · {feature_rows} ledger rows",
        )

    def abort(self) -> None:
        if self._current is not None:
            self._current.abort()


@dataclass
class PlainCampaignProgress:
    """Campaign-level append-only progress for redirected logs."""

    campaign_id: str
    total_sizes: int

    def begin_size(self, size: int, index: int, run_id: str) -> PlainPipelineProgress:
        log_progress(
            f"[campaign {self.campaign_id}] size={size} ({index}/{self.total_sizes}) "
            f"run_id={run_id} starting",
        )
        return PlainPipelineProgress()

    def end_size(self, *, feature_rows: int, elapsed_s: float, eta_s: float) -> None:
        from rde.io.console import format_duration

        eta = format_duration(eta_s) if eta_s > 0 else "0s"
        log_progress(
            f"[campaign {self.campaign_id}] size complete "
            f"rows={feature_rows} elapsed={format_duration(elapsed_s)} eta={eta}",
        )

    def abort(self) -> None:
        log_progress(f"[campaign {self.campaign_id}] interrupted")
