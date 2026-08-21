"""Brute-force size limits for exact enumeration domains."""

from __future__ import annotations

DEFAULT_MAX_BRUTEFORCE_N = 14


class SizeLimitExceeded(ValueError):
    """Raised when instance size exceeds configured brute-force limit."""

    def __init__(self, size: int, max_size: int, *, state_space: int | None = None) -> None:
        self.size = size
        self.max_size = max_size
        self.state_space = state_space
        detail = f"size N={size} exceeds max_bruteforce_n={max_size}"
        if state_space is not None:
            detail += f" (state space 2^{size}={state_space})"
        super().__init__(detail)


def validate_bruteforce_size(size: int, max_size: int | None) -> None:
    """Fail fast before O(2^N) work."""
    if max_size is None:
        return
    if size > max_size:
        raise SizeLimitExceeded(size, max_size, state_space=2**size)
