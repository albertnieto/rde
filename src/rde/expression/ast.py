"""Safe expression trees over descriptor columns (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OpKind = Literal["const", "var", "add", "sub", "mul", "div", "neg", "abs", "sqrt", "log", "exp", "pow"]

_UNARY = frozenset({"neg", "abs", "sqrt", "log", "exp"})
_BINARY = frozenset({"add", "sub", "mul", "div", "pow"})
_BINARY_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


@dataclass(frozen=True)
class Expr:
    """Small AST for metric candidate formulas."""

    kind: OpKind
    value: float | str | None = None
    left: Expr | None = None
    right: Expr | None = None

    @staticmethod
    def var(name: str) -> Expr:
        return Expr("var", value=name)

    @staticmethod
    def const(value: float) -> Expr:
        return Expr("const", value=float(value))

    def __str__(self) -> str:
        """Render without recursive Python call frames (deep GP trees)."""
        cache: dict[int, str] = {}
        stack: list[tuple[Expr, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            node_id = id(node)
            if node_id in cache:
                continue
            if node.kind == "const":
                cache[node_id] = f"{node.value:g}"
                continue
            if node.kind == "var":
                cache[node_id] = str(node.value)
                continue
            if not expanded:
                stack.append((node, True))
                if node.kind in _UNARY:
                    if node.left is not None:
                        stack.append((node.left, False))
                elif node.left is not None and node.right is not None:
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                continue
            if node.kind == "neg":
                cache[node_id] = f"(-{cache[id(node.left)]})" if node.left else "(-?)"
            elif node.kind in {"abs", "sqrt", "log", "exp"}:
                arg = cache[id(node.left)] if node.left else "?"
                cache[node_id] = f"{node.kind}({arg})"
            elif node.kind == "pow":
                cache[node_id] = f"({cache[id(node.left)]} ** {cache[id(node.right)]})"
            else:
                op = _BINARY_OP[node.kind]
                cache[node_id] = f"({cache[id(node.left)]} {op} {cache[id(node.right)]})"
        return cache[id(self)]

    def depth(self) -> int:
        """Tree depth without recursive Python call frames."""
        cache: dict[int, int] = {}
        stack: list[tuple[Expr, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            node_id = id(node)
            if node_id in cache:
                continue
            if node.kind in {"const", "var"}:
                cache[node_id] = 1
                continue
            if not expanded:
                stack.append((node, True))
                if node.kind in _UNARY:
                    if node.left is not None:
                        stack.append((node.left, False))
                elif node.left is not None and node.right is not None:
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                continue
            if node.kind in _UNARY:
                cache[node_id] = 1 + (cache.get(id(node.left), 0) if node.left else 0)
            elif node.left is not None and node.right is not None:
                cache[node_id] = 1 + max(cache[id(node.left)], cache[id(node.right)])
            else:
                cache[node_id] = 1
        return cache[id(self)]

    def size(self) -> int:
        """Node count without recursive Python call frames."""
        cache: dict[int, int] = {}
        stack: list[tuple[Expr, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            node_id = id(node)
            if node_id in cache:
                continue
            if node.kind in {"const", "var"}:
                cache[node_id] = 1
                continue
            if not expanded:
                stack.append((node, True))
                if node.kind in _UNARY:
                    if node.left is not None:
                        stack.append((node.left, False))
                elif node.left is not None and node.right is not None:
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                continue
            if node.kind in _UNARY:
                cache[node_id] = 1 + (cache.get(id(node.left), 0) if node.left else 0)
            elif node.left is not None and node.right is not None:
                cache[node_id] = 1 + cache[id(node.left)] + cache[id(node.right)]
            else:
                cache[node_id] = 1
        return cache[id(self)]
