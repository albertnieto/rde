"""Collision grouping on a query tape. No planted secret, no family label."""

from __future__ import annotations

from collections import defaultdict

from rde.core.protocols import QueryTape


def collision_groups(tape: QueryTape) -> list[list[int]]:
    """Distinct query points that share a label, groups of size ≥ 2."""
    labels: dict[int, list[int]] = defaultdict(list)
    xs = tape.xs.tolist()
    ys = tape.ys.tolist()
    for x, y in zip(xs, ys):
        labels[int(y)].append(int(x))
    groups: list[list[int]] = []
    for members in labels.values():
        unique = sorted(set(members))
        if len(unique) >= 2:
            groups.append(unique)
    return groups
