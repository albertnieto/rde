"""Tests for ``python -m rde`` entrypoint."""

from __future__ import annotations


def test_main_entry_delegates_to_cli(monkeypatch):
    import rde.cli
    from rde import __main__ as rde_main

    monkeypatch.setattr("rde.io.shutdown.install_signal_handlers", lambda: None)
    monkeypatch.setattr(rde.cli, "main", lambda: 0)
    assert rde_main._main_entry() == 0
