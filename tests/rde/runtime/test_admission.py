from __future__ import annotations

import json
from pathlib import Path

import pytest

from rde.runtime.admission import (
    admit_campaign,
    configuration_fingerprint,
    load_benchmark,
    require_admission,
    storage_work_units,
    write_admission,
)


def test_measured_admission_binds_configuration_and_budget(tmp_path: Path):
    configuration = {"sizes": [2, 3], "backend": "numpy"}
    admission = admit_campaign(
        configuration,
        work_units=10,
        benchmark={"seconds_per_unit": 2.0},
        benchmark_source="benchmark.json",
        hard_limit_seconds=30.0,
    )
    assert admission.configuration_sha256 == configuration_fingerprint(configuration)
    assert admission.estimated_seconds == 20.0
    assert admission.safety_seconds == 25.0
    assert admission.within_budget
    path = tmp_path / "admission.json"
    write_admission(path, admission)
    assert json.loads(path.read_text())["configuration_sha256"] == admission.configuration_sha256


def test_admission_rejects_oversized_campaign():
    admission = admit_campaign(
        {"backend": "numpy"},
        work_units=100,
        benchmark={"seconds_per_unit": 2.0},
        benchmark_source="benchmark.json",
        hard_limit_seconds=30.0,
    )
    assert not admission.within_budget
    with pytest.raises(RuntimeError, match="rejected"):
        require_admission(admission)


def test_storage_batch_admission_uses_session_work_units():
    benchmark = {
        "seconds_per_unit": 2.0,
        "storage_bytes_per_unit": 10.0,
    }
    session_units = storage_work_units(60, benchmark)
    session = admit_campaign(
        {"total_instances": 1_000_000, "max_storage_gb": 6},
        work_units=session_units,
        benchmark=benchmark,
        benchmark_source="benchmark.json",
        hard_limit_seconds=20.0,
    )
    full = admit_campaign(
        {"total_instances": 1_000_000, "max_storage_gb": 6},
        work_units=1_000_000,
        benchmark=benchmark,
        benchmark_source="benchmark.json",
        hard_limit_seconds=20.0,
    )
    assert session.work_units == 6
    assert session.estimated_storage_mb < 1
    assert session.within_budget
    assert not full.within_budget


def test_admission_requires_matching_benchmark_contract(tmp_path: Path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(
            {
                "seconds_per_unit": 1.0,
                "storage_bytes_per_unit": 10.0,
                "contract": {"profile": "phase1-wide"},
            }
        )
    )
    assert load_benchmark(path, expected_contract={"profile": "phase1-wide"})
    with pytest.raises(ValueError, match="contract mismatch"):
        load_benchmark(path, expected_contract={"profile": "other"})


def test_admission_rejects_storage_over_free_space(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "rde.runtime.admission.available_disk_mb",
        lambda _path: 1.0,
    )
    admission = admit_campaign(
        {"profile": "wide", "requires_storage_estimate": True},
        work_units=100,
        benchmark={"seconds_per_unit": 0.01, "storage_bytes_per_unit": 1_000_000},
        benchmark_source="benchmark.json",
        disk_path=tmp_path,
        disk_reserve_mb=1.0,
        require_storage_estimate=True,
    )
    assert not admission.within_budget
    with pytest.raises(RuntimeError, match="storage"):
        require_admission(admission)


def test_heuristic_benchmark_is_never_admitted():
    admission = admit_campaign(
        {"profile": "orientation"},
        work_units=1,
        benchmark={"seconds_per_unit": 1.0},
        benchmark_source="heuristic-not-admissible",
    )
    assert not admission.within_budget
    with pytest.raises(RuntimeError, match="heuristic"):
        require_admission(admission)
