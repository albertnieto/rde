"""Sticky header + in-place progress (Docker-style terminal UI)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, TextIO

from rde.io.console import format_duration, format_rate
from rde.io.pipeline_art import BANNER_TAGLINE, banner_combined_rich

# Fixed status table height (lines inside the dim panel)
STATUS_ROWS = 6


@dataclass
class StickyUI:
    """
    Fixed header region with a single updating progress row below.

    Brand panel: 5-row RDE + pipeline/star (animated).
    Status panel: 6-row compressed campaign table (replaced in place).
    """

    console: Any
    log_progress: bool = False
    _stream: TextIO | None = field(default=None, repr=False)
    _rich: Any = field(default=None, repr=False)
    _live: Any = field(default=None, repr=False)
    _config: dict[str, str] = field(default_factory=dict)
    _run: dict[str, str] = field(default_factory=dict)
    _progress: Any = field(default=None, repr=False)
    _task_id: Any = field(default=None, repr=False)
    _desc: str = ""
    _total: int = 0
    _current: int = 0
    _t0: float = 0.0
    _status: str = ""
    _active: bool = False
    _brand: bool = False
    _tagline: str = BANNER_TAGLINE
    _subtitle: str | None = None
    _banner_status: str | None = None
    _anim_frame: int = 0
    _anim_stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _anim_thread: threading.Thread | None = field(default=None, repr=False)

    def start(self) -> None:
        if self._active or self.console.plain:
            return
        self._stream = self.console._progress_stream()  # type: ignore[attr-defined]
        self._rich = self.console._progress_rich or self.console._rich
        if self._rich is None:
            return
        from rich.live import Live

        self._live = Live(
            self._render(),
            console=self._rich,
            refresh_per_second=12,
            transient=False,
        )
        self._live.start()
        self._active = True
        self._start_animation()

    def stop(self) -> None:
        self._stop_animation()
        if not self._active or self._live is None:
            return
        self._live.stop()
        self._live = None
        self._progress = None
        self._task_id = None
        self._active = False
        self._brand = False

    def enable_brand(self, *, subtitle: str | None = None, tagline: str = BANNER_TAGLINE) -> None:
        self._brand = True
        self._subtitle = subtitle
        self._tagline = tagline
        if subtitle:
            self._banner_status = subtitle
        self._refresh()

    def set_banner_status(self, text: str | None) -> None:
        self._banner_status = text
        self._refresh()

    def set_config(self, **fields: Any) -> None:
        """Campaign config block (written once, fixed rows)."""
        self._config = {k: _fmt_val(v) for k, v in fields.items()}
        self._refresh()

    def set_run(self, *, replace: bool = False, **fields: Any) -> None:
        """Current size/run row — replaces prior run state when replace=True."""
        payload = {k: _fmt_val(v) for k, v in fields.items() if v is not None}
        if replace:
            self._run = payload
        else:
            self._run.update(payload)
        self._refresh()

    def set_run_note(self, note: str) -> None:
        self._run["note"] = note[:88]
        self._refresh()

    # Legacy hooks — no-op append; use set_config / set_run instead
    def add_header(self, line: str) -> None:
        self.set_run_note(line)

    def add_rule(self, title: str) -> None:
        pass

    def add_kv(self, key: str, value: Any, *, key_width: int = 22) -> None:
        pass

    def begin_task(self, description: str, total: int) -> None:
        if not self._active:
            return
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._desc = description
        self._total = total
        self._current = 0
        self._t0 = time.monotonic()
        self._status = "starting..."
        self.set_run(note=f"{total} instances · starting…")
        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=32, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=False,
        )
        self._task_id = self._progress.add_task(self._status, total=total)
        self._refresh()
        if self.log_progress:
            self._log_line()

    def update_task(self, n: int = 1, *, detail: str | None = None, pulse: bool = False) -> None:
        if not self._active or self._progress is None or self._task_id is None:
            return
        if not pulse:
            self._current = min(self._total, self._current + n)
            self._progress.advance(self._task_id, n)
        if detail:
            self._status = detail
            self._progress.update(self._task_id, description=detail)
            if pulse:
                self.set_run(note=detail[:88])
            else:
                self.set_run(
                    progress=f"{self._current}/{self._total}",
                    note=detail[:88],
                )
        self._refresh()
        if self.log_progress and (not pulse or detail):
            self._log_line()

    def finish_task(self) -> None:
        self._progress = None
        self._task_id = None
        self._refresh()

    def _status_lines(self) -> list[str]:
        lines: list[str] = []
        c = self._config
        if c:
            cid = c.get("campaign_id", "?")
            dom = c.get("domain", "?")
            lines.append(f"campaign {cid}  ·  {dom}")
            sizes = _fmt_sizes(c.get("sizes", ""))
            lines.append(
                f"sizes {sizes}  ·  n={c.get('n_per_size', '?')}  ·  "
                f"{c.get('generator', '?')}  ·  {c.get('backend', '?')}  ·  w={c.get('workers', '?')}"
            )
        else:
            lines.extend(["", ""])

        lines.append("─" * 62)

        r = self._run
        if r:
            size = r.get("size", "?")
            step = r.get("step", "?")
            run_id = r.get("run_id", "?")
            lines.append(f"N={size}  ({step})  ·  {run_id}")
            timing: list[str] = []
            if r.get("elapsed"):
                timing.append(f"elapsed {r['elapsed']}")
            if r.get("eta"):
                timing.append(f"eta {r['eta']}")
            if r.get("progress"):
                timing.append(r["progress"])
            lines.append("  ·  ".join(timing) if timing else "")
            lines.append(r.get("note", ""))
        else:
            lines.extend(["", "", ""])

        while len(lines) < STATUS_ROWS:
            lines.append("")
        return lines[:STATUS_ROWS]

    def _render_status_panel(self) -> Any:
        from rich.panel import Panel
        from rich.text import Text

        if self.console.plain or self._rich is None:
            return Panel(Text("\n".join(self._status_lines())), border_style="dim", padding=(0, 1))

        t = Text()
        for i, line in enumerate(self._status_lines()):
            if i:
                t.append("\n")
            if i == 0 and self._config:
                t.append("campaign ", style="dim")
                t.append(self._config.get("campaign_id", "?"), style="cyan")
                t.append("  ·  ", style="dim")
                t.append(self._config.get("domain", "?"))
            elif i == 1 and self._config:
                t.append(
                    f"sizes {_fmt_sizes(self._config.get('sizes', ''))}  ·  "
                    f"n={self._config.get('n_per_size', '?')}  ·  "
                    f"{self._config.get('generator', '?')}  ·  "
                    f"{self._config.get('backend', '?')}  ·  "
                    f"w={self._config.get('workers', '?')}",
                    style="dim",
                )
            elif line.startswith("─"):
                t.append(line, style="dim")
            elif i == 3 and self._run:
                t.append(f"N={self._run.get('size', '?')} ", style="bold cyan")
                t.append(f"({self._run.get('step', '?')})  ·  ", style="dim")
                t.append(self._run.get("run_id", "?"))
            elif i == 4 and self._run:
                parts = []
                if self._run.get("elapsed"):
                    parts.append(f"elapsed {self._run['elapsed']}")
                if self._run.get("eta"):
                    parts.append(f"eta {self._run['eta']}")
                if self._run.get("progress"):
                    parts.append(self._run["progress"])
                t.append("  ·  ".join(parts), style="dim")
            elif i == 5 and self._run.get("note"):
                t.append(self._run.get("note", ""), style="dim italic")
            elif line:
                t.append(line)
        return Panel(t, border_style="dim", padding=(0, 1), height=STATUS_ROWS + 2)

    def _start_animation(self) -> None:
        if self._anim_thread is not None:
            return
        self._anim_stop.clear()

        def _loop() -> None:
            while not self._anim_stop.wait(0.12):
                if self._brand and self._live is not None:
                    self._anim_frame += 1
                    self._refresh()

        self._anim_thread = threading.Thread(target=_loop, name="rde-pipeline-anim", daemon=True)
        self._anim_thread.start()

    def _stop_animation(self) -> None:
        self._anim_stop.set()
        if self._anim_thread is not None:
            self._anim_thread.join(timeout=1.0)
            self._anim_thread = None

    def _render(self) -> Any:
        from rich.console import Group
        from rich.panel import Panel

        body: list[Any] = []
        if self._brand:
            body.append(
                Panel(
                    banner_combined_rich(
                        self._anim_frame,
                        plain=self.console.plain,
                        tagline=self._tagline,
                        status=self._banner_status,
                    ),
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
        if self._config or self._run:
            body.append(self._render_status_panel())
        if self._progress is not None:
            body.append(self._progress)
        return Group(*body)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _log_line(self) -> None:
        elapsed = time.monotonic() - self._t0 if self._t0 else 0.0
        frac = self._current / self._total if self._total else 0.0
        width = 24
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        rate = format_rate(self._current, elapsed)
        rate_val = self._current / elapsed if elapsed > 0 and self._current else 0.0
        eta = format_duration((self._total - self._current) / rate_val) if rate_val > 0 else "unknown"
        line = (
            f"{self._desc[:28]}  [{bar}] {self._current}/{self._total}  "
            f"{frac * 100:4.1f}%  {format_duration(elapsed)}  {rate}  eta {eta}"
        )
        if self._status:
            line += f"  {self._status}"
        from rde.io.events import log_progress

        log_progress(line)


def _fmt_val(value: Any) -> str:
    if isinstance(value, list):
        return str(value)
    return str(value)


def _fmt_sizes(raw: str) -> str:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            vals = [int(x.strip()) for x in text.strip("[]").split(",") if x.strip()]
            if len(vals) >= 2:
                return f"{vals[0]}→{vals[-1]} (×{len(vals)})"
            if len(vals) == 1:
                return str(vals[0])
        except ValueError:
            pass
    return text
