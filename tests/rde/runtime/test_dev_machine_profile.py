"""Runtime dev-machine detection for backend policy."""

from __future__ import annotations

import platform

import pytest

from rde.backends.resolve import dev_machine_profile, format_dev_machine_profile


def test_dev_machine_profile_matches_platform():
    profile = dev_machine_profile()
    assert profile.platform_machine == platform.machine().lower()
    if profile.platform_machine == "arm64":
        assert profile.profile_id == "apple_silicon_mac"
        assert profile.rde_gpu_primary == "mlx"
    elif profile.platform_machine in {"x86_64", "i386"}:
        assert profile.profile_id == "intel_mac"
        assert profile.rde_gpu_primary is None


def test_format_dev_machine_profile_json_roundtrip():
    text = format_dev_machine_profile(as_json=True)
    assert '"profile_id"' in text
    assert '"platform_machine"' in text


@pytest.mark.skipif(platform.machine().lower() != "arm64", reason="Apple Silicon only")
def test_arm64_profile_expects_mlx_when_installed():
    profile = dev_machine_profile()
    assert profile.profile_id == "apple_silicon_mac"
    if profile.mlx_usable:
        assert profile.default_expr_backend == "mlx"
