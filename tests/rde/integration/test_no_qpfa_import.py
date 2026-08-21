"""Ensure src/rde never imports qpfa (domain independence)."""

from __future__ import annotations

import ast
from pathlib import Path


RDE_ROOT = Path(__file__).resolve().parents[3] / "src" / "rde"


def _collect_imports(path: Path) -> list[tuple[str, int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno, "import"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            found.append((module, node.lineno, "from"))
    return found


def test_rde_package_does_not_import_qpfa():
    violations: list[str] = []
    for py_file in sorted(RDE_ROOT.rglob("*.py")):
        for module, lineno, kind in _collect_imports(py_file):
            if module == "qpfa" or module.startswith("qpfa."):
                rel = py_file.relative_to(RDE_ROOT.parent.parent)
                violations.append(f"{rel}:{lineno} {kind} {module}")

    assert violations == [], "rde must not import qpfa:\n" + "\n".join(violations)
