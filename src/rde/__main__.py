"""Module entry: ``python -m rde``."""

from __future__ import annotations

import sys


def _main_entry() -> int:
    from rde.io.shutdown import install_signal_handlers

    install_signal_handlers()
    from rde.cli import main

    return main()


if __name__ == "__main__":
    try:
        raise SystemExit(_main_entry())
    except KeyboardInterrupt:
        try:
            from rde.io.console import get_console

            get_console().cleanup_interrupted(message="Interrupted")
        except Exception:
            print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
