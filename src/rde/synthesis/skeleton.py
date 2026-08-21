"""Algorithm skeletons and the verb vocabulary they are named from.

A skeleton is the "one level above sums vs division vs Boolean logic" search
object: a decomposition strategy plus a combine step, not yet compiled to
arithmetic or gates. Each skeleton names itself using a small fixed verb
vocabulary so search output stays legible to a human reviewer — the verbs are
documentation of intent, the ``Recurrence`` is what actually gets solved and
verified.
"""

from __future__ import annotations

from dataclasses import dataclass

from rde.synthesis.recurrence import Recurrence

# Fixed algorithmic-verb vocabulary (deliberately small — Mode 1's
# "invent metrics, don't hand-write every one" principle applied one
# level up: invent *skeletons* from this vocabulary rather than
# hand-writing every algorithm). See
# rde/docs/methodology.md.
VERBS = (
    "evaluate",
    "compare",
    "select",
    "partition",
    "filter",
    "update",
    "merge",
    "decompose",
    "iterate",
    "aggregate",
)


@dataclass(frozen=True)
class AlgorithmSkeleton:
    """One candidate algorithm shape: a name, its verbs, and its recurrence."""

    name: str
    verbs: tuple[str, ...]
    recurrence: Recurrence

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "verbs": list(self.verbs),
            "recurrence_shape": self.recurrence.shape,
            "recurrence": self.recurrence.describe(),
        }


def base_skeleton(base_exponent: float, base: float = 2.0) -> AlgorithmSkeleton:
    """The trivial "solve directly" skeleton — brute force at declared cost."""
    from rde.synthesis.recurrence import CostClass

    rec = Recurrence(shape="base", base_cost=CostClass(kind="exp", exp_base=base, exp_rate=base_exponent))
    return AlgorithmSkeleton(name="brute_force", verbs=("evaluate", "compare", "select"), recurrence=rec)


def flat_skeleton(combine_degree: float) -> AlgorithmSkeleton:
    """decompose(instance) -> n independent leaves; evaluate each; aggregate."""
    rec = Recurrence(shape="flat", combine_degree=combine_degree)
    return AlgorithmSkeleton(
        name=f"flat_decompose_d{combine_degree:g}",
        verbs=("decompose", "evaluate", "aggregate"),
        recurrence=rec,
    )


def divide_skeleton(branches: int, divisor: float, combine_degree: float) -> AlgorithmSkeleton:
    """decompose(instance, branches) recursively on n/divisor-size parts; merge.

    `SynthesisDomain.decompose_divide(instance, branches)` is a disjoint
    *partition* into exactly `branches` equal-size pieces (see the protocol
    docstring) — it has no hook for the redundant/overlapping subproblems a
    Karatsuba-style `branches != divisor` shape would need. Executable
    catalogs (`default_skeleton_catalog`) therefore only ever build these
    with `divisor == branches`; a caller constructing one with
    `divisor != branches` directly gets a symbolically-solved recurrence
    that `rde.synthesis.search` cannot execute or verify against any current
    domain — the same documented gap as the `subtract` shape.
    """
    rec = Recurrence(shape="divide", branches=branches, divisor=divisor, combine_degree=combine_degree)
    return AlgorithmSkeleton(
        name=f"divide_a{branches}_b{divisor:g}_d{combine_degree:g}",
        verbs=("decompose", "iterate", "merge", "compare"),
        recurrence=rec,
    )


def subtract_skeleton(branches: int, shrink: int, combine_degree: float) -> AlgorithmSkeleton:
    """decompose(instance, branches) recursively on n-shrink-size parts; merge."""
    rec = Recurrence(shape="subtract", branches=branches, shrink=shrink, combine_degree=combine_degree)
    return AlgorithmSkeleton(
        name=f"subtract_a{branches}_c{shrink}_d{combine_degree:g}",
        verbs=("decompose", "iterate", "merge", "filter"),
        recurrence=rec,
    )


def default_skeleton_catalog(
    *,
    base_exponent: float = 1.0,
    combine_degrees: tuple[float, ...] = (0.0, 1.0, 2.0),
    divide_branches: tuple[int, ...] = (2, 3, 4),
    subtract_branches: tuple[int, ...] = (1, 2),
) -> list[AlgorithmSkeleton]:
    """A small, bounded enumeration of candidate skeletons.

    This is intentionally not an unbounded search: it mirrors the project's
    "don't search arbitrary arithmetic expressions everywhere" rule (the
    reverse-engineering brief's point 4) by fixing a handful of named,
    classically-meaningful shapes (flat separable decomposition, constant-
    branching divide-and-conquer, peeling recursion) rather than enumerating
    every possible (a, b, d) triple. `divide` entries always set
    `divisor == branches` (see `divide_skeleton`'s docstring for why).
    """
    catalog: list[AlgorithmSkeleton] = [base_skeleton(base_exponent)]
    for d in combine_degrees:
        catalog.append(flat_skeleton(d))
    for a in divide_branches:
        for d in combine_degrees:
            catalog.append(divide_skeleton(a, float(a), d))
    for a in subtract_branches:
        for d in combine_degrees:
            catalog.append(subtract_skeleton(a, shrink=1, combine_degree=d))
    return catalog
