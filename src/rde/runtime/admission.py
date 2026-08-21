"""Measured admission control for bounded RDE campaigns.

The estimator is intentionally conservative: it never treats the MLX
enumeration crossover table as an end-to-end campaign benchmark.  Callers
provide measured seconds per instance (or a benchmark artifact), and the
returned admission record binds the requested configuration to a stable
fingerprint before execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HARD_WALL_SECONDS = 24.0 * 60.0 * 60.0
DEFAULT_SAFETY_FRACTION = 0.80


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes; Linux reports KiB.
    raw = float(usage.ru_maxrss)
    return raw / (1024.0 * 1024.0) if platform.system() == "Darwin" else raw / 1024.0


@dataclass(frozen=True)
class Admission:
    configuration: dict[str, Any]
    configuration_sha256: str
    benchmark_source: str
    work_units: int
    measured_seconds_per_unit: float
    estimated_seconds: float
    safety_seconds: float
    hard_limit_seconds: float
    within_budget: bool
    estimated_peak_rss_mb: float | None = None
    memory_limit_mb: float | None = None
    estimated_storage_mb: float | None = None
    available_disk_mb: float | None = None
    disk_reserve_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "configuration_sha256": self.configuration_sha256,
            "benchmark_source": self.benchmark_source,
            "work_units": self.work_units,
            "measured_seconds_per_unit": self.measured_seconds_per_unit,
            "estimated_seconds": self.estimated_seconds,
            "safety_seconds": self.safety_seconds,
            "hard_limit_seconds": self.hard_limit_seconds,
            "within_budget": self.within_budget,
            "estimated_peak_rss_mb": self.estimated_peak_rss_mb,
            "memory_limit_mb": self.memory_limit_mb,
            "estimated_storage_mb": self.estimated_storage_mb,
            "available_disk_mb": self.available_disk_mb,
            "disk_reserve_mb": self.disk_reserve_mb,
        }


def load_benchmark(
    path: Path,
    *,
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "seconds_per_unit" not in data and "timings_s" not in data:
        raise ValueError(f"benchmark {path} has no seconds_per_unit or timings_s")
    if expected_contract is not None:
        actual = data.get("contract")
        if not isinstance(actual, dict):
            raise ValueError(f"benchmark {path} has no immutable contract metadata")
        mismatches = {
            key: (expected, actual.get(key))
            for key, expected in expected_contract.items()
            if actual.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"benchmark {path} contract mismatch: {mismatches}")
    return data


def available_disk_mb(path: Path | str) -> float:
    """Return currently available space on the filesystem containing ``path``."""
    return float(shutil.disk_usage(Path(path)).free) / (1024.0 * 1024.0)


def directory_size_bytes(path: Path | str) -> int:
    """Return a streaming filesystem size for a calibration artifact tree."""
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for child in root.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _storage_bytes_per_unit(benchmark: dict[str, Any]) -> float | None:
    raw = benchmark.get("storage_bytes_per_unit")
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("benchmark storage_bytes_per_unit must be finite and non-negative")
    return value


def _seconds_per_unit(benchmark: dict[str, Any]) -> float:
    if "seconds_per_unit" in benchmark:
        value = float(benchmark["seconds_per_unit"])
    else:
        timings = [float(v) for v in benchmark.get("timings_s", {}).values()]
        units = int(benchmark.get("units", 0))
        if not timings or units <= 0:
            raise ValueError("timings_s benchmark requires a positive units field")
        value = max(timings) / units
    if not math.isfinite(value) or value <= 0:
        raise ValueError("benchmark seconds_per_unit must be finite and positive")
    return value


def storage_work_units(
    max_storage_bytes: int,
    benchmark: dict[str, Any],
) -> int:
    """Return the largest positive work-unit budget fitting a byte cap."""
    value = float(max_storage_bytes)
    if (
        not math.isfinite(value)
        or value <= 0
        or value != math.floor(value)
    ):
        raise ValueError("max_storage_bytes must be a positive integer")
    storage_per_unit = _storage_bytes_per_unit(benchmark)
    if storage_per_unit is None or storage_per_unit <= 0:
        raise ValueError(
            "storage_work_units requires a positive benchmark storage_bytes_per_unit"
        )
    units = int(value // storage_per_unit)
    if units < 1:
        raise ValueError(
            "max_storage_bytes is smaller than the measured storage for one work unit"
        )
    return units


def admit_campaign(
    configuration: dict[str, Any],
    *,
    work_units: int,
    benchmark: dict[str, Any],
    benchmark_source: str,
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
    hard_limit_seconds: float = HARD_WALL_SECONDS,
    estimated_peak_rss_mb: float | None = None,
    memory_limit_mb: float | None = None,
    disk_path: Path | str | None = None,
    disk_reserve_mb: float = 0.0,
    require_storage_estimate: bool = False,
) -> Admission:
    if work_units < 0:
        raise ValueError("work_units must be non-negative")
    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")
    per_unit = _seconds_per_unit(benchmark)
    estimated = float(work_units) * per_unit
    safety = estimated / safety_fraction
    memory_ok = (
        memory_limit_mb is None
        or estimated_peak_rss_mb is None
        or estimated_peak_rss_mb <= memory_limit_mb
    )
    storage_per_unit = _storage_bytes_per_unit(benchmark)
    if require_storage_estimate and storage_per_unit is None:
        raise ValueError("campaign requires benchmark storage_bytes_per_unit")
    if require_storage_estimate and storage_per_unit <= 0:
        raise ValueError("campaign requires a positive storage_bytes_per_unit")
    estimated_storage = (
        None
        if storage_per_unit is None
        else float(work_units) * storage_per_unit / (1024.0 * 1024.0)
    )
    available = None if disk_path is None else available_disk_mb(disk_path)
    disk_ok = (
        estimated_storage is None
        or available is None
        or estimated_storage + disk_reserve_mb <= available
    )
    benchmark_admissible = bool(benchmark.get("admissible", True)) and not (
        str(benchmark_source).startswith("heuristic")
    )
    return Admission(
        configuration=dict(configuration),
        configuration_sha256=configuration_fingerprint(configuration),
        benchmark_source=benchmark_source,
        work_units=int(work_units),
        measured_seconds_per_unit=per_unit,
        estimated_seconds=estimated,
        safety_seconds=safety,
        hard_limit_seconds=hard_limit_seconds,
        within_budget=(
            safety <= hard_limit_seconds
            and memory_ok
            and disk_ok
            and benchmark_admissible
        ),
        estimated_peak_rss_mb=estimated_peak_rss_mb,
        memory_limit_mb=memory_limit_mb,
        estimated_storage_mb=estimated_storage,
        available_disk_mb=available,
        disk_reserve_mb=disk_reserve_mb,
    )


def write_admission(
    path: Path,
    admission: Admission,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **admission.to_dict(),
        **(extra or {}),
        "created_at_unix": time.time(),
        "pid": os.getpid(),
        "host": platform.node(),
        "effective_peak_rss_mb": peak_rss_mb(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_admission(admission: Admission) -> None:
    if not admission.within_budget:
        reasons: list[str] = []
        if admission.benchmark_source.startswith("heuristic"):
            reasons.append("benchmark is heuristic/non-admissible")
        if admission.safety_seconds > admission.hard_limit_seconds:
            reasons.append(
                f"{admission.safety_seconds / 3600:.2f} h projected "
                f"(hard limit {admission.hard_limit_seconds / 3600:.2f} h)"
            )
        if (
            admission.estimated_storage_mb is not None
            and admission.available_disk_mb is not None
            and admission.estimated_storage_mb + (admission.disk_reserve_mb or 0.0)
            > admission.available_disk_mb
        ):
            reasons.append(
                f"{admission.estimated_storage_mb:.0f} MB storage plus reserve "
                f"exceeds {admission.available_disk_mb:.0f} MB free"
            )
        if (
            admission.estimated_peak_rss_mb is not None
            and admission.memory_limit_mb is not None
            and admission.estimated_peak_rss_mb > admission.memory_limit_mb
        ):
            reasons.append(
                f"{admission.estimated_peak_rss_mb:.0f} MB RSS exceeds "
                f"{admission.memory_limit_mb:.0f} MB limit"
            )
        raise RuntimeError(
            "campaign rejected by measured admission: "
            + ("; ".join(reasons) if reasons else "resource contract failed")
        )
