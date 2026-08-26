"""MAP-Elites-style diversity archive.

Complements `rde.representation.pareto`'s dominance/frontier ranking: the
Pareto frontier answers "which candidates are not strictly worse than any
other on every tracked objective," which can still collapse candidates that
occupy genuinely different *behavioral* niches down to whichever wins on
those objectives. This module keeps the best-fitness candidate *per
behavior bucket* instead, so "many valid, differently-shaped witnesses" have
a principled home rather than being pruned by dominance alone.

Domain-agnostic: works on any candidate type -- a `Representation`, a
`rde.recovery.search_space.RecoveryChainResult`, a
`rde.substrate.program.Program` -- as long as the caller supplies a
`descriptor_fn` (candidate -> a fixed-length behavior-descriptor tuple) and
a `fitness_fn` (candidate -> a scalar to maximize within each bucket).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

CandidateT = TypeVar("CandidateT")

DescriptorFn = Callable[[CandidateT], Sequence[float]]
FitnessFn = Callable[[CandidateT], float]


@dataclass(frozen=True)
class Elite(Generic[CandidateT]):
    """The best-fitness candidate observed in one behavior-descriptor bucket."""

    bucket: tuple[int, ...]
    candidate: CandidateT
    descriptor: tuple[float, ...]
    fitness: float


class EliteArchive(Generic[CandidateT]):
    """Buckets a behavior-descriptor space at a fixed per-axis resolution.

    `resolution[i]` is the bucket width along descriptor axis `i` -- a
    descriptor value `v` falls in bucket `floor(v / resolution[i])`. Only
    the highest-fitness candidate per bucket is kept (`add` only replaces
    the incumbent when the new candidate's fitness is strictly greater),
    matching MAP-Elites' "keep the best per niche" rule. Insertion order
    does not affect the final archive except on an exact fitness tie, where
    the first-inserted candidate wins and a later, equal-fitness challenger
    is discarded.
    """

    def __init__(self, resolution: Sequence[float]):
        if not resolution:
            raise ValueError("resolution must have at least one axis")
        if any(r <= 0 for r in resolution):
            raise ValueError("resolution entries must be positive")
        self._resolution = tuple(resolution)
        self._elites: dict[tuple[int, ...], Elite[CandidateT]] = {}

    def _bucket_of(self, descriptor: Sequence[float]) -> tuple[int, ...]:
        if len(descriptor) != len(self._resolution):
            raise ValueError(
                f"descriptor has {len(descriptor)} axes, resolution has {len(self._resolution)}"
            )
        return tuple(int(v // r) for v, r in zip(descriptor, self._resolution))

    def add(self, candidate: CandidateT, descriptor: Sequence[float], fitness: float) -> bool:
        """Insert `candidate`. Returns whether it became the bucket's elite."""
        bucket = self._bucket_of(descriptor)
        incumbent = self._elites.get(bucket)
        if incumbent is not None and incumbent.fitness >= fitness:
            return False
        self._elites[bucket] = Elite(
            bucket=bucket,
            candidate=candidate,
            descriptor=tuple(float(v) for v in descriptor),
            fitness=float(fitness),
        )
        return True

    @property
    def elites(self) -> list[Elite[CandidateT]]:
        return list(self._elites.values())

    def __len__(self) -> int:
        return len(self._elites)


def archive_candidates(
    candidates: Sequence[CandidateT],
    descriptor_fn: DescriptorFn,
    fitness_fn: FitnessFn,
    resolution: Sequence[float],
) -> EliteArchive:
    """Bucket every candidate in `candidates` into a fresh `EliteArchive`."""
    archive: EliteArchive = EliteArchive(resolution)
    for candidate in candidates:
        archive.add(candidate, descriptor_fn(candidate), fitness_fn(candidate))
    return archive
