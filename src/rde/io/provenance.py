"""Environment and source provenance captured in RDE run manifests."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_revision(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def collect_provenance(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Collect stable execution metadata without importing optional backends."""
    root = Path(repo_root or Path.cwd()).resolve()
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "qiskit", "torch", "mlx", "pyarrow", "sympy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "git_sha": _git_revision(root),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd().resolve()),
        "packages": packages,
    }
