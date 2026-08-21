"""Tests for runtime resource limits and stage profiling."""

from __future__ import annotations

import os

from rde.runtime.config import ResourceLimits, StageProfiler


def test_resource_limits_from_env():
    env = {
        "RDE_MAX_WORKERS": "4",
        "RDE_RAM_MB": "8192",
        "RDE_NICE": "5",
    }
    old = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        limits = ResourceLimits.from_env()
        assert limits.max_workers == 4
        assert limits.max_ram_mb == 8192
        assert limits.nice_level == 5
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_stage_profiler_accumulates():
    profiler = StageProfiler()
    profiler.begin("materialize")
    profiler.end()
    assert profiler.timings.stages["materialize"] >= 0.0
    assert profiler.timings.counts["materialize"] == 1
    assert "materialize" in profiler.timings.to_dict()["stages"]

    profiler.end()  # empty stack is a no-op
