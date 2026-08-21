"""Tests for CLI shutdown helpers."""

from __future__ import annotations

import os

import pytest

from rde.io.shutdown import abort_process_pool, run_cli_command


def test_abort_process_pool_noop_on_none():
    abort_process_pool(None, None)


def test_run_cli_command_keyboard_interrupt():
    def boom(_args):
        raise KeyboardInterrupt

    assert run_cli_command(boom, object()) == 130


def test_run_cli_command_swallows_generic_errors(monkeypatch):
    def boom(_args):
        raise RuntimeError("synthetic cli failure")

    monkeypatch.delenv("RDE_DEBUG", raising=False)
    assert run_cli_command(boom, object()) == 1


def test_run_cli_command_reraises_when_debug_enabled(monkeypatch):
    def boom(_args):
        raise RuntimeError("debug me")

    monkeypatch.setenv("RDE_DEBUG", "1")
    with pytest.raises(RuntimeError, match="debug me"):
        run_cli_command(boom, object())
