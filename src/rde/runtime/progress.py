"""Progress callbacks for pipeline / campaign runs."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol


@dataclass
class InstanceProgress:
    run_id: str
    completed: int
    total: int
    feature_rows: int
    instance_id: str
    pending_slices: int


class PipelineProgressCallback(Protocol):
    def on_run_start(self, *, run_id: str, total_jobs: int, size: int, backend: str) -> None: ...

    def on_instance_done(self, event: InstanceProgress) -> None: ...

    def on_run_complete(self, *, run_id: str, feature_rows: int, skipped_instances: int) -> None: ...

    def on_heartbeat(self, *, completed: int, total: int, detail: str) -> None: ...


class NullProgress:
    """No-op progress sink."""

    def on_run_start(self, *, run_id: str, total_jobs: int, size: int, backend: str) -> None:
        pass

    def on_instance_done(self, event: InstanceProgress) -> None:
        pass

    def on_run_complete(self, *, run_id: str, feature_rows: int, skipped_instances: int) -> None:
        pass

    def on_heartbeat(self, *, completed: int, total: int, detail: str) -> None:
        pass


def default_pipeline_progress() -> PipelineProgressCallback:
    """Return the default live sink for a pipeline run.

    TTYs use the RDE console progress bar. Pipes, CI, and ``tee`` use the
    append-only sink so progress remains visible and flushes immediately.
    ``NullProgress`` is still available when a caller explicitly requests
    silence.
    """
    from rde.io.console import get_console, resolve_plain
    from rde.io.progress_ui import ConsolePipelineProgress, PlainPipelineProgress

    if resolve_plain():
        return PlainPipelineProgress()
    return ConsolePipelineProgress(get_console(plain=False))


@dataclass
class DiscoveryProgressReporter:
    """Tee-safe discovery progress with stage counts, rates, and ETA."""

    started_at: float = 0.0
    stage_count: int = 0
    stage_started_at: float = 0.0

    def __post_init__(self) -> None:
        now = time.monotonic()
        if self.started_at <= 0:
            self.started_at = now
        self.stage_started_at = now

    def on_stage(self, name: str) -> None:
        from rde.io.events import log_progress

        self.stage_count += 1
        self.stage_started_at = time.monotonic()
        elapsed = format_duration(time.monotonic() - self.started_at)
        log_progress(
            f"[discovery] stage={self.stage_count} name={name} elapsed={elapsed}",
        )

    def on_progress(self, done: int, total: int, detail: str) -> None:
        elapsed_s = max(0.0, time.monotonic() - self.stage_started_at)
        if total > 0 and done > 0 and elapsed_s > 0:
            rate = done / elapsed_s
            eta = format_duration(max(0.0, (total - done) / rate))
            progress = f"[{done}/{total}] {100.0 * done / total:5.1f}% eta={eta}"
        elif total > 0:
            progress = f"[{done}/{total}] eta=unknown"
        else:
            progress = "[progress] eta=unknown"
        from rde.io.events import log_progress

        elapsed = format_duration(time.monotonic() - self.started_at)
        log_progress(
            f"[discovery] {progress} elapsed={elapsed} {detail}",
        )


def default_discovery_progress() -> DiscoveryProgressReporter:
    """Create the default live reporter for a discovery loop."""
    return DiscoveryProgressReporter()


@dataclass
class ExperimentCampaignProgress:
    """Overall campaign progress with a nested per-size pipeline task."""

    campaign_id: str
    total_sizes: int
    _ui: object | None = None
    _started_at: float = 0.0
    _size_started_at: float = 0.0
    _completed_sizes: int = 0
    _session_ctx: object | None = None

    def __post_init__(self) -> None:
        from rde.io.console import get_console, resolve_plain
        from rde.io.progress_ui import ConsoleCampaignProgress, PlainCampaignProgress

        self._started_at = time.monotonic()
        if resolve_plain():
            self._ui = PlainCampaignProgress(self.campaign_id, self.total_sizes)
        else:
            console = get_console(plain=False)
            self._session_ctx = console.sticky_session()
            self._session_ctx.__enter__()  # type: ignore[attr-defined]
            self._ui = ConsoleCampaignProgress(console, self.campaign_id, self.total_sizes)

    def begin_size(self, size: int, index: int, run_id: str) -> PipelineProgressCallback:
        self._size_started_at = time.monotonic()
        assert self._ui is not None
        return self._ui.begin_size(size, index, run_id)  # type: ignore[no-any-return]

    def end_size(self, *, feature_rows: int, elapsed_s: float | None = None) -> None:
        self._completed_sizes += 1
        elapsed = (
            max(0.0, float(elapsed_s))
            if elapsed_s is not None
            else max(0.0, time.monotonic() - self._size_started_at)
        )
        campaign_elapsed = max(0.0, time.monotonic() - self._started_at)
        average = campaign_elapsed / self._completed_sizes
        eta = average * max(0, self.total_sizes - self._completed_sizes)
        assert self._ui is not None
        self._ui.end_size(
            feature_rows=feature_rows,
            elapsed_s=elapsed,
            eta_s=eta,
        )
        if self._completed_sizes >= self.total_sizes:
            self._close_session()

    def abort(self) -> None:
        if self._ui is not None:
            self._ui.abort()
        self._close_session()

    def _close_session(self) -> None:
        if self._session_ctx is not None:
            self._session_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
            self._session_ctx = None


def default_experiment_progress(
    campaign_id: str,
    total_sizes: int,
) -> ExperimentCampaignProgress:
    """Create the default nested progress sink for experiment scripts."""
    return ExperimentCampaignProgress(campaign_id, total_sizes)


def format_duration(seconds: float) -> str:
    """Local duration formatter for progress reporters without UI imports."""
    if seconds != seconds or seconds < 0:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{int(round(seconds))}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}min")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)
