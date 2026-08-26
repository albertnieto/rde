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


def _hamming_distance(left: int, right: int) -> int:
    return (int(left) ^ int(right)).bit_count()


def near_collision_groups(tape: QueryTape, *, radius: int = 4) -> list[list[int]]:
    """Group distinct query points whose 63-bit labels are Hamming-near.

    This is deliberately an observation-layer primitive, not a family
    branch: every recovery program receives the same ``QueryTape`` and can
    choose to read exact or near label agreement.  The index uses the
    pigeonhole principle.  With a radius of four, two labels within the
    radius share at least one of five fixed bit blocks, so only points in the
    same block bucket need an exact ``bit_count`` check.  For random labels
    this is expected linear work in the tape length, unlike an all-pairs
    distance scan.

    The current HSP adapter emits 63-bit labels and is scoped through
    ``n_bits <= 24``.  The fixed five-way partition is consequently part of
    this adapter's declared finite-horizon resource model, rather than an
    asymptotic claim about arbitrary-width label encodings.
    """
    radius = int(radius)
    if radius < 0 or radius > 4:
        raise ValueError("near-collision radius must be in [0, 4]")
    key = f"near_collision_groups:r{radius}"
    cached = tape.cache.get(key)
    if cached is not None:
        return cached

    xs = [int(x) for x in tape.xs.tolist()]
    ys = [int(y) for y in tape.ys.tolist()]
    parent = list(range(len(xs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        left, right = find(i), find(j)
        if left != right:
            parent[right] = left

    # Radius r needs r+1 blocks.  The 63 label bits split into nearly equal
    # blocks whose widths sum exactly to 63.
    n_blocks = radius + 1
    widths = [63 // n_blocks + (1 if i < (63 % n_blocks) else 0) for i in range(n_blocks)]
    offset = 0
    for block, width in enumerate(widths):
        mask = (1 << width) - 1
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, label in enumerate(ys):
            bucket = (block, (label >> offset) & mask)
            for prior in buckets[bucket]:
                if _hamming_distance(label, ys[prior]) <= radius:
                    union(i, prior)
            buckets[bucket].append(i)
        offset += width

    components: dict[int, set[int]] = defaultdict(set)
    for i, x in enumerate(xs):
        components[find(i)].add(x)
    groups = [sorted(members) for members in components.values() if len(members) >= 2]
    tape.cache[key] = groups
    return groups
