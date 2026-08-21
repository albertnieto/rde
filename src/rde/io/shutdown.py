"""Graceful CLI shutdown: signals, terminal restore, clean exit codes."""

from __future__ import annotations

import multiprocessing
import logging
import os
import signal
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable

from rde.io.console import get_console
from rde.io.events import configure_logging

_interrupt_count = 0
_LOG = logging.getLogger(__name__)


def abort_process_pool(
    pool: ProcessPoolExecutor | None,
    futures: list[Future] | None = None,
) -> None:
    """Cancel pending work and stop worker processes without waiting on in-flight jobs."""
    if pool is None:
        return
    if futures:
        for fut in futures:
            fut.cancel()
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        try:
            pool.shutdown(wait=False)
        except Exception:
            pass
    for child in multiprocessing.active_children():
        try:
            child.terminate()
        except Exception:
            pass


def install_signal_handlers() -> None:
    """Wire SIGINT/SIGTERM/SIGTSTP to restore the terminal before exit/suspend."""

    def _interrupt(signum: int, frame: Any) -> None:
        global _interrupt_count
        _interrupt_count += 1
        if _interrupt_count >= 2:
            get_console().restore_terminal()
            os._exit(130)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupt)

    if hasattr(signal, "SIGTSTP"):

        def _tstp(signum: int, frame: Any) -> None:
            get_console().restore_terminal()
            signal.signal(signal.SIGTSTP, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTSTP)

        signal.signal(signal.SIGTSTP, _tstp)

    if hasattr(signal, "SIGCONT"):

        def _cont(signum: int, frame: Any) -> None:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except OSError:
                pass

        signal.signal(signal.SIGCONT, _cont)


def run_cli_command(func: Callable[[Any], int], args: Any) -> int:
    """Run a subcommand; swallow tracebacks unless RDE_DEBUG is set."""
    configure_logging()
    try:
        return func(args)
    except KeyboardInterrupt:
        get_console().cleanup_interrupted(message="Interrupted")
        return 130
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except OSError:
            pass
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    except Exception as exc:
        get_console().restore_terminal()
        _LOG.exception(
            "cli_command_failed",
            extra={
                "event": "cli_command_failed",
                "fields": {"command": getattr(args, "command", None)},
            },
        )
        if os.environ.get("RDE_DEBUG", "").lower() in {"1", "true", "yes"}:
            raise
        get_console().error(str(exc) or exc.__class__.__name__)
        return 1
