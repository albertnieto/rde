"""Combined RDE figlet + pipeline/star (5-row side-by-side, animated)."""

from __future__ import annotations

from typing import Any

# Left: RDE figlet (5 rows, width 22)
RDE_LINES = (
    " ____   ____   _____ ",
    "|  _ \\ |  _ \\ | ____|",
    "| |_) || | | ||  _| ",
    "|  _ < | |_| || |___ ",
    "|_| \\_\\|____/ |_____|",
)

RDE_WIDTH = max(len(line) for line in RDE_LINES)
GAP = 3

# Right: pipeline + Gemini star across 5 rows (same height as RDE)
_PIPELINE = "generate ──► measure ──► discover"

_RIGHT_ROWS = (
    f"{_PIPELINE}  *",
    "                 /|\\",
    "                * | *",
    "                 \\|/",
    "                  *",
)

# Star glyph positions in right block (row, col) -> animation group
_STAR_GROUPS: dict[tuple[int, int], int] = {}
for _c in range(len(_RIGHT_ROWS[1])):
    if _RIGHT_ROWS[1][_c] not in " /|\\":
        continue
    _STAR_GROUPS[(1, _c)] = 1
for _c in range(len(_RIGHT_ROWS[2])):
    ch = _RIGHT_ROWS[2][_c]
    if ch == "|":
        _STAR_GROUPS[(2, _c)] = 3
    elif ch == "*":
        _STAR_GROUPS[(2, _c)] = 2 if _c < len(_RIGHT_ROWS[2]) // 2 else 4
for _c in range(len(_RIGHT_ROWS[3])):
    if _RIGHT_ROWS[3][_c] in "\\|/":
        _STAR_GROUPS[(3, _c)] = 5
for _c in range(len(_RIGHT_ROWS[0])):
    if _RIGHT_ROWS[0][_c] == "*":
        _STAR_GROUPS[(0, _c)] = 0
for _c in range(len(_RIGHT_ROWS[4])):
    if _RIGHT_ROWS[4][_c] == "*":
        _STAR_GROUPS[(4, _c)] = 6

_PALETTE = (
    "bright_blue",
    "blue",
    "cyan",
    "bright_magenta",
    "magenta",
    "bright_white",
)

BANNER_TAGLINE = "Representation Discovery Engine"
BANNER_ROWS = 5


def _style_for(group: int, frame: int, *, plain: bool) -> str | None:
    if plain:
        return None
    highlight = (frame // 2) % 7
    if group == highlight:
        return "bright_white bold"
    return _PALETTE[(frame + group * 2) % len(_PALETTE)]


def _append_right_row(text: Any, row_idx: int, frame: int, *, plain: bool) -> None:
    line = _RIGHT_ROWS[row_idx]
    if row_idx == 0:
        # Pipeline label in cyan; apex star animated
        pipe_end = line.rindex("*") if "*" in line else len(line)
        text.append(line[:pipe_end], style=None if plain else "cyan")
        for col in range(pipe_end, len(line)):
            ch = line[col]
            if ch == " ":
                text.append(ch)
            else:
                style = _style_for(_STAR_GROUPS.get((row_idx, col), 0), frame, plain=plain)
                if style:
                    text.append(ch, style=style)
                else:
                    text.append(ch)
        return
    for col, ch in enumerate(line):
        if ch == " ":
            text.append(ch)
            continue
        group = _STAR_GROUPS.get((row_idx, col), row_idx)
        style = _style_for(group, frame, plain=plain)
        if style:
            text.append(ch, style=style)
        else:
            text.append(ch)


def banner_combined_rich(
    frame: int = 0,
    *,
    plain: bool = False,
    tagline: str | None = BANNER_TAGLINE,
    status: str | None = None,
) -> Any:
    """Five fixed rows: RDE left, pipeline+star right; optional tagline + status below."""
    from rich.text import Text

    block = Text()
    right_w = max(len(r) for r in _RIGHT_ROWS)
    for row_idx in range(BANNER_ROWS):
        if row_idx:
            block.append("\n")
        left = RDE_LINES[row_idx]
        block.append(left, style=None if plain else "bold cyan")
        block.append(" " * max(0, RDE_WIDTH - len(left) + GAP))
        pad = max(0, right_w - len(_RIGHT_ROWS[row_idx]))
        # right-align star block within column
        right_line = _RIGHT_ROWS[row_idx]
        if row_idx > 0:
            block.append(" " * pad)
        _append_right_row(block, row_idx, frame, plain=plain)
    if tagline:
        block.append("\n")
        block.append(tagline, style=None if plain else "dim italic")
    if status:
        block.append("\n")
        block.append(status, style=None if plain else "dim")
    return block


def banner_combined_plain(frame: int = 0, *, tagline: str | None = BANNER_TAGLINE, status: str | None = None) -> str:
    lines: list[str] = []
    right_w = max(len(r) for r in _RIGHT_ROWS)
    for row_idx in range(BANNER_ROWS):
        left = RDE_LINES[row_idx]
        gap = " " * (RDE_WIDTH - len(left) + GAP)
        right = _RIGHT_ROWS[row_idx]
        if row_idx > 0:
            right = right.rjust(right_w)
        lines.append(f"{left}{gap}{right}")
    if tagline:
        lines.append(tagline)
    if status:
        lines.append(status)
    return "\n".join(lines)


def banner_combined_ansi(frame: int = 0, *, tagline: str | None = BANNER_TAGLINE, status: str | None = None) -> str:
    """Fallback ANSI render (single frame)."""
    cyan = "\033[36m"
    dim = "\033[2m"
    reset = "\033[0m"
    lines: list[str] = []
    right_w = max(len(r) for r in _RIGHT_ROWS)
    for row_idx in range(BANNER_ROWS):
        left = RDE_LINES[row_idx]
        gap = " " * (RDE_WIDTH - len(left) + GAP)
        right = _RIGHT_ROWS[row_idx] if row_idx == 0 else _RIGHT_ROWS[row_idx].rjust(right_w)
        lines.append(f"{cyan}{left}{reset}{gap}{right}")
    if tagline:
        lines.append(f"{dim}{tagline}{reset}")
    if status:
        lines.append(f"{dim}{status}{reset}")
    return "\n".join(lines)
