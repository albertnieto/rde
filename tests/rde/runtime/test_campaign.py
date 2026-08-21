"""Campaign, export, resume, and CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rde.analyze.query import flatten_features, summarize_run
import rde.runtime.campaign as campaign_runtime
from rde.runtime.campaign import CampaignConfig, run_campaign
from rde.runtime.pipeline import RunConfig, StorageBudgetExceeded, run_pipeline
from rde.io.seal import seal_run
from rde.io.store import Store, iter_jsonl
from tests.rde.helpers import toy_registry


def test_export_features_csv(tmp_path: Path):
    reg = toy_registry()
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=2,
            size=8,
            seed=0,
            indices=[1, 2],
            store_root=tmp_path,
            run_id="export_test",
        ),
        registry=reg,
    )
    store = Store(tmp_path)
    out = store.export_features_csv("export_test", tmp_path / "out.csv")
    store.close()
    text = out.read_text(encoding="utf-8")
    assert "representation_complexity" in text
    assert "family_index" in text


def test_resume_skips_completed_instances(tmp_path: Path):
    reg = toy_registry()
    config = RunConfig(
        domain_id="synthetic_poly",
        n_instances=3,
        size=8,
        seed=99,
        indices=[1],
        store_root=tmp_path,
        run_id="resume_test",
    )
    first = run_pipeline(config, registry=reg)
    assert first.n_skipped_instances == 0

    second = run_pipeline(RunConfig(**{**config.__dict__, "resume": True}), registry=reg)
    assert second.n_skipped_instances == 3


def test_resume_backfills_missing_instance_metadata(tmp_path: Path):
    reg = toy_registry()
    config = RunConfig(
        domain_id="synthetic_poly",
        n_instances=2,
        size=8,
        seed=77,
        indices=[1],
        store_root=tmp_path,
        run_id="resume_backfill_test",
    )
    run_pipeline(config, registry=reg)
    store = Store(tmp_path)
    run_dir = store.run_dir("resume_backfill_test")
    instances = list(iter_jsonl(run_dir / "instances.jsonl"))
    (run_dir / "instances.jsonl").write_text("", encoding="utf-8")
    store.close()

    resumed = run_pipeline(RunConfig(**{**config.__dict__, "resume": True}), registry=reg)
    assert resumed.n_skipped_instances == 2

    store = Store(tmp_path)
    assert store.run_stats("resume_backfill_test")["instances_lines"] == 2
    restored = list(iter_jsonl(store.run_dir("resume_backfill_test") / "instances.jsonl"))
    store.close()
    assert {row["instance_id"] for row in restored} == {
        row["instance_id"] for row in instances
    }


def test_resume_rejects_changed_configuration(tmp_path: Path):
    reg = toy_registry()
    config = RunConfig(
        domain_id="synthetic_poly",
        n_instances=2,
        size=8,
        seed=99,
        indices=[1],
        store_root=tmp_path,
        run_id="resume_config_test",
    )
    run_pipeline(config, registry=reg)
    changed = RunConfig(**{**config.__dict__, "seed": 100, "resume": True})
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        run_pipeline(changed, registry=reg)


def test_storage_budget_resume_does_not_duplicate_feature_keys(tmp_path: Path):
    reg = toy_registry()
    config = RunConfig(
        domain_id="synthetic_poly",
        n_instances=3,
        size=8,
        seed=101,
        indices=[1, 2],
        store_root=tmp_path,
        run_id="storage_resume_test",
        max_storage_bytes=10_000,
    )
    with pytest.raises(StorageBudgetExceeded) as error:
        run_pipeline(config, registry=reg)
    assert error.value.completed_jobs == 1
    assert error.value.new_feature_rows == 2

    resumed = run_pipeline(
        RunConfig(**{**config.__dict__, "resume": True, "max_storage_bytes": None}),
        registry=reg,
    )
    assert resumed.n_new_feature_rows > 0
    store = Store(tmp_path)
    assert len(store.completed_feature_keys("storage_resume_test")) == 6
    assert len(store.read_features("storage_resume_test")) == 6
    store.close()


def test_campaign_storage_batch_journal_and_resume(tmp_path: Path):
    reg = toy_registry()
    partial = run_campaign(
        CampaignConfig(
            domain_id="synthetic_poly",
            sizes=[8],
            n_per_size=3,
            indices=[1, 2],
            store_root=tmp_path,
            campaign_id="storage_batch",
            resume=False,
            strict=True,
            compute_backend="numpy",
            max_storage_bytes=10_000,
        ),
        registry=reg,
    )
    assert partial.batch_summary["stop_reason"] == "storage_budget"
    assert partial.batch_summary["instances_completed_this_session"] == 1
    assert partial.batch_summary["bytes_written_this_session"] >= 1
    journal = tmp_path / "campaigns" / "storage_batch" / "batches.jsonl"
    assert journal.exists()

    complete = run_campaign(
        CampaignConfig(
            domain_id="synthetic_poly",
            sizes=[8],
            n_per_size=3,
            indices=[1, 2],
            store_root=tmp_path,
            campaign_id="storage_batch",
            resume=True,
            strict=True,
            compute_backend="numpy",
        ),
        registry=reg,
    )
    assert complete.batch_summary["stop_reason"] == "complete"
    store = Store(tmp_path)
    assert len(store.completed_feature_keys("storage_batch_n8")) == 6
    store.close()


def test_campaign_seal_batch_cleans_operational_artifacts_and_resumes(
    tmp_path: Path,
):
    reg = toy_registry()
    config = CampaignConfig(
        domain_id="synthetic_poly",
        sizes=[4],
        n_per_size=2,
        indices=[1, 2],
        store_root=tmp_path,
        campaign_id="sealed_campaign",
        resume=False,
        strict=True,
        compute_backend="numpy",
        seal_batches=True,
    )
    first = run_campaign(config, registry=reg)
    run_dir = tmp_path / "runs" / "sealed_campaign_n4"
    assert first.batch_summary["sealed"] is True
    assert first.batch_summary["seal_records"][0]["status"] == "sealed"
    assert not (run_dir / "completion.sqlite3").exists()
    assert not (run_dir / "arrays").exists()
    assert (run_dir / "sealed" / "features.parquet").exists()
    assert not (run_dir / "features.jsonl").exists()

    resumed = run_campaign(
        CampaignConfig(**{**config.__dict__, "resume": True}),
        registry=reg,
    )
    assert resumed.run_results[0].n_skipped_instances == 2
    assert len(flatten_features("sealed_campaign_n4", tmp_path)) == 4


def test_pipeline_refuses_resume_false_on_sealed_run(tmp_path: Path):
    pytest.importorskip("pyarrow")
    reg = toy_registry()
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=2,
            size=4,
            seed=7,
            indices=[1, 2],
            store_root=tmp_path,
            run_id="sealed_guard",
        ),
        registry=reg,
    )
    seal_run(tmp_path, "sealed_guard")
    with pytest.raises(ValueError, match="sealed"):
        run_pipeline(
            RunConfig(
                domain_id="synthetic_poly",
                n_instances=2,
                size=4,
                seed=7,
                indices=[1, 2],
                store_root=tmp_path,
                run_id="sealed_guard",
                resume=False,
            ),
            registry=reg,
        )


def test_campaign_multi_size(tmp_path: Path):
    reg = toy_registry()
    result = run_campaign(
        CampaignConfig(
            domain_id="synthetic_poly",
            sizes=[4, 8],
            n_per_size=2,
            seed_base=0,
            indices=[1, 2],
            store_root=tmp_path,
            campaign_id="camp1",
            resume=False,
        ),
        registry=reg,
    )
    assert len(result.run_results) == 2
    summary = summarize_run("camp1_n4", tmp_path)
    assert summary["n_feature_rows"] == 4


@pytest.mark.parametrize(
    ("strict", "max_wall_seconds"),
    [(True, None), (False, 60.0)],
)
def test_campaign_propagates_strict_or_deadline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strict: bool,
    max_wall_seconds: float | None,
):
    def fail_run(*args, **kwargs):
        del args, kwargs
        raise TimeoutError("synthetic deadline") if max_wall_seconds else ValueError(
            "synthetic strict failure"
        )

    monkeypatch.setattr(campaign_runtime, "run_pipeline", fail_run)
    with pytest.raises(TimeoutError if max_wall_seconds else ValueError):
        run_campaign(
            CampaignConfig(
                domain_id="synthetic_poly",
                sizes=[4],
                n_per_size=1,
                store_root=tmp_path,
                campaign_id="failure_policy",
                plain_output=True,
                strict=strict,
                max_wall_seconds=max_wall_seconds,
            )
        )


def test_flatten_features(tmp_path: Path):
    reg = toy_registry()
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=1,
            size=4,
            seed=0,
            indices=[1],
            store_root=tmp_path,
            run_id="flat",
        ),
        registry=reg,
    )
    rows = flatten_features("flat", tmp_path)
    assert len(rows) == 1
    assert "metric.representation_complexity" in rows[0]


def test_run_cli_command_keyboard_interrupt():
    from rde.io.console import RdeConsole, set_console
    from rde.io.shutdown import run_cli_command

    set_console(RdeConsole(plain=True))

    def _boom(_args):
        raise KeyboardInterrupt

    assert run_cli_command(_boom, object()) == 130


def test_abort_process_pool_returns_promptly():
    import time
    from concurrent.futures import ProcessPoolExecutor

    from rde.io.shutdown import abort_process_pool

    def _slow_worker() -> int:
        import time

        time.sleep(60)
        return 1

    pool = ProcessPoolExecutor(max_workers=1)
    fut = pool.submit(_slow_worker)
    time.sleep(0.3)
    t0 = time.monotonic()
    abort_process_pool(pool, [fut])
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"abort_process_pool blocked for {elapsed:.1f}s"


def test_cli_sigint_exits_cleanly():
    import os
    import signal
    import subprocess
    import sys
    import textwrap
    import time

    script = textwrap.dedent(
        """
        import time
        from rde.io.console import RdeConsole, set_console
        from rde.io.shutdown import install_signal_handlers, run_cli_command

        install_signal_handlers()
        set_console(RdeConsole(plain=True))

        def work(_args):
            print("running", flush=True)
            while True:
                time.sleep(0.1)

        raise SystemExit(run_cli_command(work, None))
        """
    )
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        line = proc.stdout.readline() if proc.stdout else ""
        if "running" in line:
            break
        time.sleep(0.05)
    proc.send_signal(signal.SIGINT)
    try:
        stdout, stderr = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(f"SIGINT handler did not exit within 15s: stdout={stdout!r} stderr={stderr!r}")
    assert proc.returncode == 130, (proc.returncode, stdout, stderr)
    assert "Traceback" not in stderr
    assert "Traceback" not in stdout
    assert "Interrupted" in stdout or "Interrupted" in stderr


def test_cli_run_subprocess(tmp_path: Path):
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "rde",
        "run",
        "--domain",
        "synthetic_poly",
        "--n-instances",
        "1",
        "--size",
        "4",
        "--indices",
        "1",
        "--store-root",
        str(tmp_path),
        "--run-id",
        "cli_run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert proc.returncode == 0, proc.stderr
    assert "run_id=cli_run" in proc.stdout
