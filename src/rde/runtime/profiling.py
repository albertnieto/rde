"""Deprecated import path — use ``rde.runtime.config`` instead."""

from __future__ import annotations

import warnings

from rde.runtime.config import StageProfiler, StageTimings

__all__ = ["StageProfiler", "StageTimings"]

warnings.warn(
    "rde.runtime.profiling is deprecated; import StageProfiler from rde.runtime.config",
    DeprecationWarning,
    stacklevel=2,
)
