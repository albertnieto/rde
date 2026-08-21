"""Shared AST dispatch for expression evaluation across numeric backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rde.expression.ast import Expr


class ExprNumericBackend(Protocol):
    def const(self, value: float, n: int) -> Any: ...
    def var(self, key: str, env: dict[str, Any]) -> Any: ...
    def neg(self, x: Any) -> Any: ...
    def abs(self, x: Any) -> Any: ...
    def sqrt(self, x: Any) -> Any: ...
    def log(self, x: Any) -> Any: ...
    def exp(self, x: Any) -> Any: ...
    def add(self, a: Any, b: Any) -> Any: ...
    def sub(self, a: Any, b: Any) -> Any: ...
    def mul(self, a: Any, b: Any) -> Any: ...
    def div(self, num: Any, denom: Any) -> Any: ...
    def pow(self, base: Any, expn: Any) -> Any: ...
    def env_len(self, env: dict[str, Any]) -> int: ...


Instruction = tuple[str, float | str | None]

_UNARY = frozenset({"neg", "abs", "sqrt", "log", "exp"})
_BINARY = frozenset({"add", "sub", "mul", "div", "pow"})


@dataclass(frozen=True)
class CompiledExpr:
    """Post-order instruction tape for an expression tree."""

    instructions: tuple[Instruction, ...]


def _compile_expr_iterative(root: Expr) -> tuple[Instruction, ...]:
    """Lower an Expr tree without recursive hashing or recursive Python calls.

    ``functools.lru_cache`` on frozen nested ``Expr`` trees hashes the whole
    AST first; pathological GP/DreamCoder depths then raise RecursionError
    before compilation starts. An explicit post-order stack stays O(depth)
    in heap frames and O(nodes) in work.
    """
    cache: dict[int, tuple[Instruction, ...]] = {}
    stack: list[tuple[Expr, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        node_id = id(node)
        if node_id in cache:
            continue
        if node.kind == "const":
            cache[node_id] = (("const", float(node.value)),)
            continue
        if node.kind == "var":
            cache[node_id] = (("var", str(node.value)),)
            continue
        if not expanded:
            stack.append((node, True))
            if node.kind in _UNARY:
                if node.left is None:
                    raise ValueError(f"Unary expression {node.kind!r} has no operand")
                stack.append((node.left, False))
            elif node.kind in _BINARY:
                if node.left is None or node.right is None:
                    raise ValueError(f"Binary expression {node.kind!r} has a missing operand")
                stack.append((node.right, False))
                stack.append((node.left, False))
            else:
                raise ValueError(f"Unknown expression kind: {node.kind!r}")
            continue
        if node.kind in _UNARY:
            assert node.left is not None
            cache[node_id] = cache[id(node.left)] + ((node.kind, None),)
        else:
            assert node.left is not None and node.right is not None
            cache[node_id] = (
                cache[id(node.left)] + cache[id(node.right)] + ((node.kind, None),)
            )
    return cache[id(root)]


def compile_expr(expr: Expr) -> CompiledExpr:
    """Lower an ``Expr`` tree to a flat post-order instruction tape."""
    return CompiledExpr(_compile_expr_iterative(expr))


def evaluate_compiled_expr(
    compiled: CompiledExpr,
    env: dict[str, Any],
    backend: ExprNumericBackend,
) -> Any:
    """Interpret a compiled tape without recursive Python AST dispatch."""
    stack: list[Any] = []
    n = backend.env_len(env) if env else 1
    for opcode, payload in compiled.instructions:
        if opcode == "const":
            stack.append(backend.const(float(payload), n))
        elif opcode == "var":
            key = str(payload)
            if key not in env:
                raise KeyError(f"Unknown variable: {key!r}")
            stack.append(backend.var(key, env))
        elif opcode in _UNARY:
            if not stack:
                raise ValueError(f"Unary expression {opcode!r} has no operand")
            value = stack.pop()
            stack.append(getattr(backend, opcode)(value))
        elif opcode in _BINARY:
            if len(stack) < 2:
                raise ValueError(f"Binary expression {opcode!r} has missing operands")
            right = stack.pop()
            left = stack.pop()
            stack.append(getattr(backend, opcode)(left, right))
        else:
            raise ValueError(f"Unknown expression kind: {opcode!r}")
    if len(stack) != 1:
        raise ValueError("Compiled expression did not produce exactly one value")
    return stack[0]


def evaluate_expr_with_backend(expr: Expr, env: dict[str, Any], backend: ExprNumericBackend) -> Any:
    """Evaluate an expression through its memoized iterative instruction tape."""
    return evaluate_compiled_expr(compile_expr(expr), env, backend)
