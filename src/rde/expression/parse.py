"""Parse ``str(Expr)`` back into an expression tree (for seed re-scoring)."""

from __future__ import annotations

from rde.expression.ast import Expr


class _ParseError(ValueError):
    pass


class _Parser:
    def __init__(self, source: str) -> None:
        self._src = source.strip()
        self._i = 0

    def parse(self) -> Expr:
        if not self._src:
            raise _ParseError("empty expression")
        expr = self._parse_expr()
        self._skip_ws()
        if self._i < len(self._src):
            raise _ParseError(f"unexpected trailing input at {self._src[self._i:]!r}")
        return expr

    def _skip_ws(self) -> None:
        while self._i < len(self._src) and self._src[self._i].isspace():
            self._i += 1

    def _peek(self) -> str:
        self._skip_ws()
        return self._src[self._i] if self._i < len(self._src) else ""

    def _consume(self, expected: str | None = None) -> str:
        self._skip_ws()
        if self._i >= len(self._src):
            raise _ParseError("unexpected end of input")
        ch = self._src[self._i]
        if expected is not None and ch != expected:
            raise _ParseError(f"expected {expected!r}, got {ch!r}")
        self._i += 1
        return ch

    def _parse_expr(self) -> Expr:
        left = self._parse_term()
        while True:
            self._skip_ws()
            if self._peek() == "+":
                self._consume("+")
                left = Expr("add", left=left, right=self._parse_term())
            elif self._peek() == "-":
                self._consume("-")
                left = Expr("sub", left=left, right=self._parse_term())
            else:
                break
        return left

    def _parse_term(self) -> Expr:
        left = self._parse_factor()
        while True:
            self._skip_ws()
            if self._peek() == "*":
                self._consume("*")
                left = Expr("mul", left=left, right=self._parse_factor())
            elif self._peek() == "/":
                self._consume("/")
                left = Expr("div", left=left, right=self._parse_factor())
            else:
                break
        return left

    def _parse_factor(self) -> Expr:
        self._skip_ws()
        ch = self._peek()
        if ch == "(":
            self._consume("(")
            if self._peek() == "-":
                self._consume("-")
                inner = self._parse_expr()
                self._consume(")")
                return Expr("neg", left=inner)
            inner = self._parse_expr()
            self._skip_ws()
            if self._peek() == "**":
                self._consume("*")
                self._consume("*")
                right = self._parse_factor()
                self._consume(")")
                return Expr("pow", left=inner, right=right)
            self._consume(")")
            return inner
        if ch == "-":
            self._consume("-")
            return Expr("neg", left=self._parse_factor())
        return self._parse_atom()

    def _parse_atom(self) -> Expr:
        self._skip_ws()
        ch = self._peek()
        if ch == "(":
            return self._parse_factor()
        if ch.isdigit() or ch == ".":
            return self._parse_number()
        if ch.isalpha() or ch in {"_", "."}:
            return self._parse_call_or_var()
        raise _ParseError(f"unexpected character {ch!r}")

    def _parse_number(self) -> Expr:
        start = self._i
        while self._i < len(self._src) and (self._src[self._i].isdigit() or self._src[self._i] in {".", "e", "E", "+", "-"}):
            self._i += 1
        text = self._src[start : self._i]
        try:
            return Expr.const(float(text))
        except ValueError as exc:
            raise _ParseError(f"invalid number {text!r}") from exc

    def _parse_call_or_var(self) -> Expr:
        start = self._i
        while self._i < len(self._src) and (self._src[self._i].isalnum() or self._src[self._i] in {"_", "."}):
            self._i += 1
        name = self._src[start : self._i]
        self._skip_ws()
        if self._peek() == "(":
            self._consume("(")
            arg = self._parse_expr()
            self._consume(")")
            if name in {"abs", "sqrt", "log", "exp"}:
                return Expr(name, left=arg)  # type: ignore[arg-type]
            raise _ParseError(f"unknown unary function {name!r}")
        return Expr.var(name)


def parse_expr(source: str) -> Expr:
    """Parse the string form produced by ``str(Expr)``."""
    return _Parser(source).parse()
