"""Regression tests for the core/plugin distribution boundary."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _manifest(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_core_and_adapter_manifests_are_distinct():
    core = _manifest(ROOT / "src" / "rde" / "pyproject.toml")
    adapters = _manifest(ROOT / "src" / "rde_domains" / "pyproject.toml")

    assert core["project"]["name"] == "rde"
    assert adapters["project"]["name"] == "rde-domains"
    assert "rde.domains" not in core.get("project", {}).get("entry-points", {})
    assert adapters["project"]["entry-points"]["rde.domains"]


def test_core_manifest_has_no_adapter_dependency():
    core = _manifest(ROOT / "src" / "rde" / "pyproject.toml")
    dependencies = core["project"]["dependencies"]

    assert all("rde-domains" not in dependency for dependency in dependencies)
    assert all("qpfa" not in dependency.lower() for dependency in dependencies)


def test_core_tests_do_not_import_adapter_modules():
    violations: list[str] = []
    test_root = ROOT / "tests" / "rde"
    for py_file in sorted(test_root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(
                module == "rde_domains"
                or module.startswith("rde_domains.")
                or module == "qpfa"
                or module.startswith("qpfa.")
                for module in modules
            ):
                violations.append(f"{py_file.relative_to(ROOT)}:{node.lineno}")

    assert violations == [], "core tests must not import adapter modules:\n" + "\n".join(violations)
