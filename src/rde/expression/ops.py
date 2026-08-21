"""Expression operator registry (Phase 3 DSL)."""

from __future__ import annotations

UNARY_OPS: tuple[str, ...] = ("abs", "sqrt", "log", "exp", "neg")
BINARY_OPS: tuple[str, ...] = ("add", "sub", "mul", "div", "pow")

# Safe depth limits for enumeration (avoid exp(exp(...)) blow-up).
MAX_EXP_DEPTH: int = 1
MAX_LOG_DEPTH: int = 2
