"""Re-export shared soft-fail helpers for discovery modules."""

from __future__ import annotations

from rde.runtime.resilience import SOFT_FAIL_EXCEPTIONS, soft_call, soft_map

__all__ = ["SOFT_FAIL_EXCEPTIONS", "soft_call", "soft_map"]
