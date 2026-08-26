"""Core protocols for the Representation Discovery Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from rde.core.instance import InstanceRecord


class QueryIntent(str, Enum):
    """What a metric is intended to support (succinctness vs query power)."""

    EVALUATE = "evaluate"
    OVERLAP = "overlap"
    UPDATE = "update"
    COMPRESS = "compress"
    RANK = "rank"


@dataclass(frozen=True)
class ResourceModel:
    """Declared resource scaling for a representation candidate (v0.2)."""

    classical_time: str
    memory: str
    gates: str
    depth: str
    shots: str
    precision: str

    def to_dict(self) -> dict[str, str]:
        return {
            "classical_time": self.classical_time,
            "memory": self.memory,
            "gates": self.gates,
            "depth": self.depth,
            "shots": self.shots,
            "precision": self.precision,
        }


@runtime_checkable
class FamilySlice(Protocol):
    """One element of a parameterized family (e.g. trajectory index r)."""

    @property
    def values(self) -> np.ndarray:
        """Primary array payload (vector, matrix, or coefficient table)."""
        ...

    @property
    def index(self) -> int:
        """Discrete family parameter."""
        ...

    @property
    def kind(self) -> str:
        """Semantic label for the payload (domain-defined string)."""
        ...


class SimpleFamilySlice:
    """Default FamilySlice implementation."""

    def __init__(self, values: np.ndarray, index: int, kind: str = "vector") -> None:
        self._values = np.asarray(values)
        self._index = index
        self._kind = kind

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def index(self) -> int:
        return self._index

    @property
    def kind(self) -> str:
        return self._kind


@runtime_checkable
class Domain(Protocol):
    """Adapter that samples P and measures Z (methodology PZXESO).

    Science stack is `rde/docs/methodology.md`. This protocol
    is the code hook — not a second ontology.

    Instance cache: if `materialize` and `primitive_features` share an
    expensive primitive, implement `prepare_instance(instance, *, indices=None)
    -> dict` so the worker computes it once and passes `cache=` into both.
    `primitive_features` must expose raw arrays the forward catalog can sweep,
    not only scalars from one hand-authored mechanism.
    """

    @property
    def domain_id(self) -> str:
        ...

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        """Create n instances of the given size (sample P)."""
        ...

    def materialize(self, instance: InstanceRecord, index: int) -> SimpleFamilySlice:
        """Materialize one family element at discrete index.

        Accept `cache=` when `prepare_instance` exists; do not re-enumerate a
        cached landscape, query sample, or factorial tour list.
        """
        ...

    def primitive_features(self, instance: InstanceRecord) -> dict[str, float | np.ndarray]:
        """Domain-specific raw quantities (inputs to descriptors/metrics).

        Accept `cache=` when `prepare_instance` exists. Return named arrays
        (`D`, `costs`, `diff_profile`, …) so the generic catalog can run.
        """
        ...


DescriptorFn = Callable[..., dict[str, float]]


@runtime_checkable
class Descriptor(Protocol):
    """Domain-agnostic feature extractor operating on arrays."""

    @property
    def name(self) -> str:
        ...

    def compute(
        self,
        instance: InstanceRecord,
        slice_: SimpleFamilySlice | None = None,
        array: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        ...


MetricFn = Callable[[InstanceRecord, SimpleFamilySlice, dict[str, Any]], float]


@runtime_checkable
class Metric(Protocol):
    """Scored quantity derived from descriptors / family slices."""

    @property
    def name(self) -> str:
        ...

    @property
    def intent(self) -> QueryIntent:
        ...

    def score(
        self,
        instance: InstanceRecord,
        slice_: SimpleFamilySlice,
        descriptors: dict[str, Any],
    ) -> float:
        ...


@dataclass(frozen=True)
class SynthesisSolution:
    """One candidate solution to a synthesis-domain instance (or sub-instance)."""

    assignment: Any
    cost: float


@runtime_checkable
class SynthesisDomain(Protocol):
    """Domain-agnostic contract for reverse/backward algorithm synthesis (Mode 2).

    Where ``Domain`` answers "what object are we studying," ``SynthesisDomain``
    answers "what structural moves can an algorithm skeleton legally make on
    this object." It exists so ``rde.synthesis`` can search *backward* from a
    declared resource target (see ``rde synthesize``) instead of only fitting
    representations to data already generated for a fixed object, without
    ``rde.synthesis`` itself knowing anything about any concrete domain.

    Every method must be a genuine structural operation on the instance —
    never a shortcut that looks at, or is derived from, the answer
    (``brute_force``'s own output). A domain that let ``decompose_*`` or
    ``combine`` peek at ``brute_force`` would smuggle the answer into the
    "discovered" skeleton; see ``rde/docs/methodology.md`` §8
    and THEOREM-001 for why that is rejected project-wide, not just here.
    """

    @property
    def domain_id(self) -> str:
        ...

    def size_of(self, instance: Any) -> int:
        """The structural size parameter n used by the recurrence solver."""
        ...

    def brute_force(self, instance: Any) -> SynthesisSolution:
        """Exact optimum via exhaustive search. Reference oracle only — never
        called by a candidate skeleton above the domain's base-case threshold."""
        ...

    def decompose_flat(self, instance: Any) -> list[Any] | None:
        """Split into n independent O(1)-size leaves in one step, or None if
        this instance does not admit that structure."""
        ...

    def decompose_divide(self, instance: Any, branches: int) -> list[Any] | None:
        """Split into `branches` roughly-equal-size sub-instances (recursive
        divide-and-conquer shape), or None if unsupported at this size/branch
        count."""
        ...

    def combine(self, instance: Any, sub_solutions: list[SynthesisSolution]) -> SynthesisSolution:
        """Merge sub-solutions into a solution for `instance`."""
        ...

    def cost(self, instance: Any, solution: SynthesisSolution) -> float:
        """Objective value of `solution` on `instance` (lower is better)."""
        ...


@dataclass(frozen=True)
class QueryTape:
    """Bounded oracle transcript. Extractors may read only this object."""

    xs: Any  # np.ndarray[int]
    ys: Any  # np.ndarray[int]
    budget: int
    modulus: int
    n_bits: int
    # Per-tape derived structures shared by a recovery catalog.  A catalog can
    # contain many programs that read the same relation (for example, the
    # Hamming-near label relation); recomputing it for every candidate would
    # turn a polynomial recovery search into needless repeated work.
    cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@runtime_checkable
class RecoveryProtocol(Protocol):
    """Mode 2 extractor: map a query tape to a candidate hidden object.

    Must not receive the planted secret, family label, or generator name.
    Returning ``None`` means abstain (no structure claimed).
    """

    protocol_id: str

    def extract(self, tape: QueryTape) -> Any:
        ...


@runtime_checkable
class RecoveryDomain(Protocol):
    """Mode 2 scoring surface for query-bounded recovery of planted structure.

    Distinct from ``SynthesisDomain`` (divide / combine recurrences). Same
    mode: start from ``C_target`` (query budget + exact recovery) and ask
    whether an extractor meets it. ``planted`` / ``match`` are scoring-only
    and must never be passed into ``RecoveryProtocol.extract``.
    """

    @property
    def domain_id(self) -> str:
        ...

    def size_of(self, instance: Any) -> int:
        ...

    def family_of(self, instance: Any) -> str:
        ...

    def draw_tape(self, instance: Any, rng: Any) -> QueryTape:
        ...

    def planted(self, instance: Any) -> Any:
        """Hidden object used only to score extractors. Never given to them."""
        ...

    def match(self, recovered: Any, planted: Any) -> bool:
        ...


class CallableDescriptor:
    """Wrap a function as a Descriptor."""

    def __init__(self, name: str, fn: DescriptorFn) -> None:
        self._name = name
        self._fn = fn
        self._accepts_context = "context" in inspect.signature(fn).parameters

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        instance: InstanceRecord,
        slice_: SimpleFamilySlice | None = None,
        array: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if self._accepts_context:
            return self._fn(instance, slice_, array, context=context)
        return self._fn(instance, slice_, array)


class CallableMetric:
    """Wrap a function as a Metric."""

    def __init__(self, name: str, intent: QueryIntent, fn: MetricFn) -> None:
        self._name = name
        self._intent = intent
        self._fn = fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def intent(self) -> QueryIntent:
        return self._intent

    def score(
        self,
        instance: InstanceRecord,
        slice_: SimpleFamilySlice,
        descriptors: dict[str, Any],
    ) -> float:
        return self._fn(instance, slice_, descriptors)
