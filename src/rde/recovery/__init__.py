"""Mode 2 query-tape recovery — extract planted structure under a budget.

This is not a third science mode. It is the HSP-shaped surface of reverse
synthesis: ``C_target`` is a poly(n) query budget plus exact recovery of a
hidden object, and search is over ``RecoveryProtocol`` extractors that see
only a ``QueryTape``. Divide/combine recurrences stay in ``rde.synthesis``.
"""

from rde.core.protocols import QueryTape, RecoveryDomain, RecoveryProtocol
from rde.recovery.extractors import (
    AdditiveGcdExtractor,
    AdditiveSumExtractor,
    XorCollisionExtractor,
    default_extractor_catalog,
)
from rde.recovery.programs import (
    ConfidentCollisionProgram,
    GroupClosureProgram,
    PairCombine,
    enumerate_recovery_programs,
)
from rde.recovery.search import RecoveryReport, RecoveryRow, evaluate_protocols
from rde.recovery.search_space import (
    RecoveryChainResult,
    enumerate_recovery_chains,
    search_recovery_chains,
)
from rde.recovery.campaign import ProtocolSearchVerdict, assess_protocol_search

__all__ = [
    "AdditiveGcdExtractor",
    "AdditiveSumExtractor",
    "ConfidentCollisionProgram",
    "GroupClosureProgram",
    "PairCombine",
    "QueryTape",
    "RecoveryChainResult",
    "RecoveryDomain",
    "RecoveryProtocol",
    "RecoveryReport",
    "RecoveryRow",
    "ProtocolSearchVerdict",
    "XorCollisionExtractor",
    "assess_protocol_search",
    "default_extractor_catalog",
    "enumerate_recovery_chains",
    "enumerate_recovery_programs",
    "evaluate_protocols",
    "search_recovery_chains",
]
