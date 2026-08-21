"""Ensure the standalone RDE core has no adapter-package imports."""

from __future__ import annotations

import ast
from pathlib import Path


RDE_ROOT = Path(__file__).resolve().parents[3] / "src" / "rde"


def test_rde_core_does_not_import_rde_domains():
    violations: list[str] = []
    for py_file in sorted(RDE_ROOT.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                if module == "rde_domains" or module.startswith("rde_domains."):
                    rel = py_file.relative_to(RDE_ROOT.parent.parent)
                    violations.append(f"{rel}:{node.lineno} {module}")

    assert violations == [], "rde core must not import rde_domains:\n" + "\n".join(violations)
