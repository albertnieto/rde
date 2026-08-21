"""Run ledger persistence (JSONL + NPZ)."""

from rde.io.json_util import json_default, utc_now_iso, write_json

__all__ = ["Store", "json_default", "utc_now_iso", "write_json"]


def __getattr__(name: str):
    if name == "Store":
        from rde.io.store import Store

        return Store
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
