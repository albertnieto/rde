"""Terminal UI for RDE — colors, banner, live progress (Rich with ANSI fallback)."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

BANNER_TAGLINE = "Representation Discovery Engine"

FOOTER_RULE = "─" * 72

# ANSI palette (Claude Code-ish: muted cyan headers, soft green ok, amber warn)
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"


def _use_plain(explicit_plain: bool | None = None) -> bool:
    if explicit_plain is True:
        return True
    if os.environ.get("RDE_FORCE_UI", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("RDE_PLAIN", "").lower() in {"1", "true", "yes"}:
        return True
    if os.environ.get("NO_COLOR"):
        return True
    return not sys.stdout.isatty()


def resolve_plain(*, flag: bool = False) -> bool:
    """True when --plain, RDE_PLAIN, NO_COLOR, or stdout is not a TTY."""
    return flag or _use_plain(None)


def format_duration(seconds: float) -> str:
    """Human-readable duration: 45.2s, 2min 15s, 1h 5min 3s, 1d 2h 30min."""
    if seconds != seconds or seconds < 0:  # NaN
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{int(round(seconds))}s"
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}min")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def format_rate(count: int, elapsed: float) -> str:
    """Throughput label; uses /min when per-second rate would round to 0."""
    if elapsed <= 0:
        return "warming up"
    if count <= 0:
        return "warming up" if elapsed < 120 else f"0/min ({format_duration(elapsed)} elapsed)"
    per_sec = count / elapsed
    if per_sec >= 0.05:
        return f"{per_sec:.2f}/s"
    per_min = per_sec * 60.0
    if per_min >= 0.05:
        return f"{per_min:.2f}/min"
    per_hr = per_sec * 3600.0
    return f"{per_hr:.2f}/hr"


def _progress_stream():
    """Best stream for live progress: stdout if TTY, else /dev/tty, else stdout."""
    if sys.stdout.isatty():
        return sys.stdout
    if os.environ.get("RDE_FORCE_UI", "").lower() in {"1", "true", "yes"}:
        try:
            return open("/dev/tty", "w", buffering=1)  # noqa: SIM115
        except OSError:
            pass
    return sys.stdout


def _use_log_progress() -> bool:
    """Optional newline progress for log files (RDE_LOG_PROGRESS=1). Default: in-place UI."""
    return os.environ.get("RDE_LOG_PROGRESS", "").lower() in {"1", "true", "yes"}


def _ansi(text: str, code: str, *, plain: bool) -> str:
    if plain:
        return text
    return f"{code}{text}{_RESET}"


@dataclass
class RdeConsole:
    """Styled CLI output; degrades gracefully without Rich or in CI."""

    plain: bool = field(default_factory=lambda: _use_plain(None))
    log_progress: bool = field(default_factory=_use_log_progress)
    _sticky: Any = field(default=None, repr=False)
    _abort_hooks: list[Callable[[], None]] = field(default_factory=list, repr=False)
    _terminal_restored: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._rich = None
        self._progress_rich = None
        if not self.plain:
            try:
                from rich.console import Console as RichConsole

                force = os.environ.get("RDE_FORCE_UI", "").lower() in {"1", "true", "yes"}
                self._rich = RichConsole(
                    highlight=False,
                    soft_wrap=True,
                    force_terminal=force or None,
                )
                prog_file = _progress_stream()
                if prog_file is not sys.stdout or sys.stdout.isatty():
                    self._progress_rich = RichConsole(
                        file=prog_file,
                        highlight=False,
                        soft_wrap=True,
                        force_terminal=True,
                    )
                elif not _use_log_progress():
                    self._progress_rich = self._rich
            except ImportError:
                pass

    def _progress_stream(self) -> Any:
        return _progress_stream()

    def register_abort_hook(self, hook: Callable[[], None]) -> None:
        """Register UI teardown invoked on interrupt or fatal CLI exit."""
        self._abort_hooks.append(hook)

    def unregister_abort_hook(self, hook: Callable[[], None]) -> None:
        try:
            self._abort_hooks.remove(hook)
        except ValueError:
            pass

    def restore_terminal(self) -> None:
        """Run abort hooks and leave the cursor on a fresh line."""
        if self._terminal_restored:
            return
        self._terminal_restored = True
        for hook in reversed(self._abort_hooks):
            try:
                hook()
            except Exception:
                pass
        self._abort_hooks.clear()
        if self._sticky is not None:
            try:
                self._sticky.stop()
            except Exception:
                pass
            self._sticky = None
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except OSError:
            pass

    def cleanup_interrupted(self, *, message: str | None = "Interrupted") -> None:
        """Restore the terminal after Ctrl+C, suspend, or unexpected failure."""
        self.restore_terminal()
        if message:
            self.warn(message)

    @contextmanager
    def sticky_session(self) -> Iterator[None]:
        """Fixed header + in-place progress (Docker-style). Progress goes to TTY."""
        if self.plain or self._rich is None:
            yield
            return
        from rde.io.sticky_ui import StickyUI

        sticky = StickyUI(self, log_progress=self.log_progress)
        self._sticky = sticky

        def _stop_sticky() -> None:
            if sticky._active:
                sticky.stop()
            self._sticky = None

        self.register_abort_hook(_stop_sticky)
        sticky.start()
        try:
            yield
        finally:
            _stop_sticky()
            self.unregister_abort_hook(_stop_sticky)

    def rule(self, title: str) -> None:
        if self._sticky is not None and self._sticky._active:
            return
        if self._rich:
            self._rich.rule(f"[bold cyan]{title}[/bold cyan]")
        else:
            line = "-" * min(72, max(24, len(title) + 4))
            print(_ansi(f"\n{line}", _CYAN, plain=self.plain))
            print(_ansi(f" {title}", _BOLD + _CYAN, plain=self.plain))
            print(_ansi(line, _CYAN, plain=self.plain))

    def banner(self, *, subtitle: str | None = None) -> None:
        from rde.io.pipeline_art import (
            banner_combined_ansi,
            banner_combined_plain,
            banner_combined_rich,
        )

        if self._sticky is not None and self._sticky._active:
            self._sticky.enable_brand(subtitle=subtitle)
            return
        if self._rich:
            self._rich.print(
                banner_combined_rich(0, tagline=BANNER_TAGLINE, status=subtitle)
            )
            self._rich.print(f"[dim]{FOOTER_RULE}[/dim]")
            return
        if self.plain:
            print(banner_combined_plain(0, tagline=BANNER_TAGLINE, status=subtitle))
        else:
            print(banner_combined_ansi(0, tagline=BANNER_TAGLINE, status=subtitle))
        print(_ansi(FOOTER_RULE, _DIM, plain=self.plain))

    def info(self, msg: str) -> None:
        if self._sticky is not None and self._sticky._active:
            self._sticky.set_run_note(msg)
            return
        self._print(msg, style="dim" if self._rich else None, ansi=_DIM)

    def set_campaign_config(self, **fields: Any) -> None:
        if self._sticky is not None and self._sticky._active:
            self._sticky.set_config(**fields)
            return
        for key, value in fields.items():
            self.kv(key, value)

    def set_campaign_run(self, *, replace: bool = False, **fields: Any) -> None:
        if self._sticky is not None and self._sticky._active:
            self._sticky.set_run(replace=replace, **fields)
            return
        parts = "  ".join(f"{k}={v}" for k, v in fields.items())
        self.info(parts)

    def success(self, msg: str) -> None:
        if self._rich:
            self._rich.print(f"[green]{msg}[/green]")
        else:
            print(_ansi(msg, _GREEN, plain=self.plain))

    def warn(self, msg: str) -> None:
        if self._rich:
            self._rich.print(f"[yellow]{msg}[/yellow]")
        else:
            print(_ansi(msg, _YELLOW, plain=self.plain))

    def error(self, msg: str) -> None:
        if self._rich:
            self._rich.print(f"[red]{msg}[/red]")
        else:
            print(_ansi(msg, _RED, plain=self.plain))

    def kv(self, key: str, value: Any, *, key_width: int = 22) -> None:
        if self._sticky is not None and self._sticky._active:
            return
        line = f"{key:<{key_width}} {value}"
        if self._rich:
            self._rich.print(f"[cyan]{key:<{key_width}}[/cyan] {value}")
        else:
            print(_ansi(key.ljust(key_width), _CYAN, plain=self.plain) + str(value))

    def table(self, title: str, columns: list[str], rows: list[list[Any]]) -> None:
        if self._rich:
            from rich.table import Table

            tbl = Table(title=title, show_header=True, header_style="bold cyan", border_style="dim")
            for col in columns:
                tbl.add_column(col)
            for row in rows:
                tbl.add_row(*[str(c) for c in row])
            self._rich.print(tbl)
            return
        print(_ansi(title, _BOLD + _CYAN, plain=self.plain))
        header = "  ".join(columns)
        print(_ansi(header, _BOLD, plain=self.plain))
        for row in rows:
            print("  ".join(str(c) for c in row))

    def _print(self, msg: str, *, style: str | None, ansi: str) -> None:
        if self._rich and style:
            self._rich.print(f"[{style}]{msg}[/{style}]")
        elif self._rich:
            self._rich.print(msg)
        else:
            print(_ansi(msg, ansi, plain=self.plain))

    @contextmanager
    def task_progress(
        self,
        description: str,
        total: int,
        *,
        show_rate: bool = True,
    ) -> Iterator["TaskProgress"]:
        if total <= 0:
            yield TaskProgress(self, description, total, _PlainTracker())
            return
        if self._sticky is not None and self._sticky._active:
            self._sticky.begin_task(description, total)
            tracker = _StickyTracker(self._sticky)
            yield TaskProgress(self, description, total, tracker)
            self._sticky.finish_task()
            return
        log_mode = self.log_progress or _use_log_progress()
        rich_console = self._progress_rich if self._progress_rich and not log_mode else None
        if rich_console:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            progress = Progress(
                TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                BarColumn(bar_width=36, complete_style="cyan", finished_style="green"),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
                console=rich_console,
                refresh_per_second=10,
            )
            with progress:
                task_id = progress.add_task(description, total=total)
                tracker = _RichTracker(progress, task_id)
                tracker.advance(0, detail="starting...")
                yield TaskProgress(self, description, total, tracker)
            return
        if log_mode:
            tracker = _LogLineTracker(description, total, show_rate=show_rate, plain=self.plain)
            tracker.advance(0, detail="starting...")
            yield TaskProgress(self, description, total, tracker)
            return
        tracker = _AnsiTracker(description, total, show_rate=show_rate, plain=self.plain)
        yield TaskProgress(self, description, total, tracker)
        tracker.finish()

    @contextmanager
    def spinner(self, message: str) -> Iterator[None]:
        if self._rich:
            from rich.status import Status

            with Status(f"[cyan]{message}[/cyan]", spinner="dots", console=self._rich):
                yield
            return
        print(_ansi(f"... {message}", _CYAN, plain=self.plain), end="", flush=True)
        try:
            yield
        finally:
            print(_ansi(" done", _GREEN, plain=self.plain))


class _StickyTracker:
    def __init__(self, sticky: Any) -> None:
        self._sticky = sticky

    def advance(self, n: int = 1, *, detail: str | None = None, pulse: bool = False) -> None:
        self._sticky.update_task(n, detail=detail, pulse=pulse)


class _RichTracker:
    def __init__(self, progress: Any, task_id: Any) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, n: int = 1, *, detail: str | None = None, pulse: bool = False) -> None:
        if not pulse:
            self._progress.advance(self._task_id, n)
        if detail is not None:
            self._progress.update(self._task_id, description=detail)


class _AnsiTracker:
    def __init__(self, description: str, total: int, *, show_rate: bool, plain: bool) -> None:
        self.description = description
        self.total = total
        self.show_rate = show_rate
        self.plain = plain
        self.current = 0
        self._t0 = time.monotonic()
        self._render("")

    def advance(self, n: int = 1, *, detail: str | None = None, pulse: bool = False) -> None:
        if pulse:
            self._render(detail or "working...")
            return
        self.current = min(self.total, self.current + n)
        self._render(detail or "")

    def _render(self, detail: str) -> None:
        width = 36
        frac = self.current / self.total if self.total else 1.0
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self._t0
        rate_lbl = format_rate(self.current, elapsed)
        rate_val = self.current / elapsed if elapsed > 0 and self.current else 0.0
        eta = (self.total - self.current) / rate_val if rate_val > 0 else 0.0
        parts = [
            _ansi(self.description, _CYAN, plain=self.plain),
            f"[{bar}] {self.current}/{self.total}",
            f"{frac * 100:5.1f}%",
            format_duration(elapsed),
        ]
        if self.show_rate:
            parts.append(rate_lbl)
            if rate_val > 0:
                parts.append(f"eta {format_duration(eta)}")
        if detail:
            parts.append(_ansi(detail, _DIM, plain=self.plain))
        sys.stdout.write("\r" + "  ".join(parts)[:140].ljust(140))
        sys.stdout.flush()

    def finish(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


class _LogLineTracker:
    """Newline progress for piped stdout (tee, CI logs)."""

    def __init__(self, description: str, total: int, *, show_rate: bool, plain: bool) -> None:
        self.description = description
        self.total = total
        self.show_rate = show_rate
        self.plain = plain
        self.current = 0
        self._t0 = time.monotonic()
        self._last_print = -1
        self._last_emit_t = self._t0

    def advance(self, n: int = 1, *, detail: str | None = None, pulse: bool = False) -> None:
        if pulse:
            now = time.monotonic()
            if now - self._last_emit_t >= 5.0:
                self._emit(detail or "working...", pulse=True)
                self._last_emit_t = now
            return
        self.current = min(self.total, self.current + n)
        if self.current == 0 or self.current == self.total:
            self._emit(detail or "")
            return
        now = time.monotonic()
        if self.current - self._last_print >= 1 or now - self._last_emit_t >= 5.0:
            self._emit(detail or "")
            self._last_print = self.current

    def _emit(self, detail: str, *, pulse: bool = False) -> None:
        width = 24
        frac = self.current / self.total if self.total else 1.0
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self._t0
        rate_lbl = format_rate(self.current, elapsed)
        rate_val = self.current / elapsed if elapsed > 0 and self.current else 0.0
        eta = (self.total - self.current) / rate_val if rate_val > 0 else 0.0
        parts = [
            _ansi(self.description[:32], _CYAN, plain=self.plain),
            f"[{bar}] {self.current}/{self.total}",
            f"{frac * 100:5.1f}%",
            format_duration(elapsed),
        ]
        if self.show_rate:
            parts.append(rate_lbl)
            if rate_val > 0:
                parts.append(f"eta {format_duration(eta)}")
        if detail:
            parts.append(_ansi(detail, _DIM, plain=self.plain))
        print("  ".join(parts), flush=True)
        self._last_emit_t = time.monotonic()


class _PlainTracker:
    def advance(self, n: int = 1, *, detail: str | None = None, pulse: bool = False) -> None:
        pass


@dataclass
class TaskProgress:
    console: RdeConsole
    description: str
    total: int
    _tracker: Any

    def update(self, n: int = 1, *, detail: str | None = None, pulse: bool = False) -> None:
        self._tracker.advance(n, detail=detail, pulse=pulse)

    def set_description(self, detail: str) -> None:
        if hasattr(self._tracker, "advance"):
            self._tracker.advance(0, detail=f"{self.description}  {detail}")


# Module singleton for commands that don't pass console explicitly
_default: RdeConsole | None = None


def get_console(*, plain: bool | None = None) -> RdeConsole:
    global _default
    if plain is not None or _default is None:
        _default = RdeConsole(plain=_use_plain(plain) if plain is not None else _use_plain(None))
    return _default


def set_console(console: RdeConsole) -> None:
    global _default
    _default = console
