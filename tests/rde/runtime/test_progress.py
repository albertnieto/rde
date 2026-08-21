"""Regression tests for continuous RDE progress reporting."""

from __future__ import annotations

import threading

from rde.core.registry import Registry
from rde.discovery.loop import DiscoveryReport, run_discovery
from rde.discovery.symbolic import discover_equation
from rde.discovery.walsh_synthesis import run_walsh_synthesis
from rde.features import register_builtin_descriptors, register_builtin_metrics
from rde.runtime.campaign import CampaignConfig, run_campaign
from rde.runtime.heartbeat import StageHeartbeat, throttled_progress
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.runtime.progress import InstanceProgress, NullProgress
from rde.testing import SyntheticPolyDomain


def test_stage_heartbeat_emits_while_stage_is_blocked():
    pulses: list[str] = []
    pulse_seen = threading.Event()

    def callback(_done: int, _total: int, detail: str) -> None:
        pulses.append(detail)
        if "still running" in detail:
            pulse_seen.set()

    heartbeat = StageHeartbeat(callback, label="long stage", interval_s=0.01)
    heartbeat.start()
    try:
        assert pulse_seen.wait(1.0)
    finally:
        heartbeat.stop(detail="complete")

    assert pulses[0] == "long stage: started"
    assert pulses[-1] == "long stage: complete"


def test_throttled_progress_reports_first_count_terminal_and_threshold():
    events: list[tuple[int, int, str]] = []
    report = throttled_progress(
        lambda done, total, detail: events.append((done, total, detail)),
        min_interval_s=60.0,
        every_n=2,
    )
    assert report is not None

    report(0, 10, "first")
    report(1, 10, "held")
    report(2, 10, "threshold")
    report(10, 10, "complete")

    assert [done for done, _total, _detail in events] == [0, 2, 10]


def test_plain_campaign_keeps_instance_progress(tmp_path, capsys):
    registry = Registry()
    registry.register_domain(SyntheticPolyDomain())
    register_builtin_descriptors(registry)
    register_builtin_metrics(registry)

    run_campaign(
        CampaignConfig(
            domain_id="synthetic_poly",
            sizes=[4],
            n_per_size=2,
            seed_base=0,
            indices=[1, 2],
            store_root=tmp_path,
            campaign_id="plain_progress",
            compute_backend="numpy",
            workers=1,
            plain_output=True,
        ),
        registry=registry,
    )

    output = capsys.readouterr().out
    assert "[pipeline plain_progress_n4] 2 instances queued" in output
    assert "[pipeline plain_progress_n4] [1/2]" in output
    assert "[pipeline plain_progress_n4] complete" in output


def test_pipeline_defaults_to_live_plain_progress(tmp_path, capsys):
    registry = Registry()
    registry.register_domain(SyntheticPolyDomain())
    register_builtin_descriptors(registry)
    register_builtin_metrics(registry)

    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=2,
            size=4,
            indices=[1],
            store_root=tmp_path,
            run_id="default_progress",
            compute_backend="numpy",
        ),
        registry=registry,
    )

    output = capsys.readouterr().out
    assert "[pipeline default_progress] 2 instances queued" in output
    assert "[pipeline default_progress] [1/2]" in output
    assert "elapsed=" in output
    assert "eta=" in output


def test_pipeline_null_progress_is_explicitly_silent(tmp_path, capsys):
    registry = Registry()
    registry.register_domain(SyntheticPolyDomain())
    register_builtin_descriptors(registry)
    register_builtin_metrics(registry)

    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=1,
            size=4,
            indices=[1],
            store_root=tmp_path,
            run_id="null_progress",
            compute_backend="numpy",
        ),
        registry=registry,
        progress=NullProgress(),
    )

    assert "[pipeline null_progress]" not in capsys.readouterr().out


def test_discovery_defaults_to_live_progress(monkeypatch, tmp_path, capsys):
    import rde.discovery.loop as loop

    def fake_body(*_args, **kwargs):
        kwargs["on_stage"]("fake stage")
        kwargs["on_progress"](1, 2, "candidate batch")
        return DiscoveryReport(run_id="fake", target="target", n_rows=2)

    monkeypatch.setattr(loop, "_run_discovery_body", fake_body)
    run_discovery("fake", tmp_path, target="target")

    output = capsys.readouterr().out
    assert "[discovery] stage=1 name=fake stage" in output
    assert "[1/2]" in output
    assert "eta=" in output


def test_plain_pipeline_progress_reports_eta(capsys):
    from rde.io.progress_ui import PlainPipelineProgress

    progress = PlainPipelineProgress()
    progress.on_run_start(run_id="eta", total_jobs=2, size=4, backend="numpy")
    progress.on_instance_done(
        InstanceProgress(
            run_id="eta",
            completed=1,
            total=2,
            feature_rows=1,
            instance_id="instance",
            pending_slices=0,
        )
    )

    assert "eta=" in capsys.readouterr().out


def test_symbolic_engine_progress_callback_is_live():
    rows = [{"x": float(i), "target": float(2 * i + 1)} for i in range(20)]
    progress: list[tuple[int, int, str]] = []

    discover_equation(
        rows,
        "target",
        features=["x"],
        use_pysr=False,
        use_operon=False,
        use_physics_templates=False,
        use_native_gp=True,
        gp_generations=2,
        gp_population=8,
        eval_backend="numpy",
        on_progress=lambda done, total, detail: progress.append((done, total, detail)),
    )

    assert any("symbolic native_gp: started" in detail for _done, _total, detail in progress)
    assert any("backend=numpy" in detail for _done, _total, detail in progress)


def test_walsh_and_dreamcoder_progress_callback_is_live():
    rows = [
        {
            "fourier.w0": float(i % 2),
            "fourier.w1": float((i // 2) % 2),
            "target": float(i % 2),
        }
        for i in range(20)
    ]
    progress: list[tuple[int, int, str]] = []

    run_walsh_synthesis(
        rows,
        "target",
        ["fourier.w0", "fourier.w1"],
        max_support=1,
        max_candidates=10,
        gp_generations=2,
        gp_population=8,
        eval_backend="numpy",
        on_progress=lambda done, total, detail: progress.append((done, total, detail)),
    )

    assert progress
    assert any("backend=numpy" in detail for _done, _total, detail in progress)
    assert any(done == total for done, total, _detail in progress if total > 0)
