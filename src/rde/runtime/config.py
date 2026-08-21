"""Resource limits and stage profiling for RDE v0.3."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class ResourceLimits:
    max_workers: int = max(1, (os.cpu_count() or 2) - 2)
    gpu_batch_size: int | None = None
    max_inflight_gpu_batches: int = 2
    max_inflight_descriptor_jobs: int = 16
    max_jsonl_buffer_rows: int = 512
    max_ram_mb: int | None = None
    nice_level: int = 10

    @classmethod
    def from_env(cls) -> ResourceLimits:
        workers = os.environ.get("RDE_MAX_WORKERS")
        ram = os.environ.get("RDE_RAM_MB")
        nice = os.environ.get("RDE_NICE")
        return cls(
            max_workers=int(workers) if workers else max(1, (os.cpu_count() or 2) - 2),
            max_ram_mb=int(ram) if ram else None,
            nice_level=int(nice) if nice else 10,
        )


@dataclass
class StageTimings:
    stages: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"stages": dict(self.stages), "counts": dict(self.counts)}


class StageProfiler:
    def __init__(self) -> None:
        self.timings = StageTimings()
        self._stack: list[tuple[str, float]] = []

    def begin(self, stage: str) -> None:
        self._stack.append((stage, perf_counter()))

    def end(self, stage: str | None = None) -> None:
        if not self._stack:
            return
        name, t0 = self._stack.pop()
        if stage is not None:
            name = stage
        elapsed = perf_counter() - t0
        self.timings.stages[name] = self.timings.stages.get(name, 0.0) + elapsed
        self.timings.counts[name] = self.timings.counts.get(name, 0) + 1
