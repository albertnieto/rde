"""An opt-in, ontology-free program-execution substrate.

`rde.representation.program_search` and `rde.recovery.search_space` search
*typed* candidate spaces (a fixed grammar of primitives, a fixed catalog of
collision-algebra programs) -- deliberately, since exhaustive search over a
small typed grammar is tractable and auditable (see each module's
docstring). This package is the complementary, deliberately minimal
alternative: candidates are bare instruction sequences (`program.Program`)
for a tiny deterministic stack machine (`vm.execute`), enumerated by brute
force (`enumeration.enumerate_programs`) with no attached mathematical
meaning -- whatever a program "means" comes only from a caller-supplied
verifier via `rde.search.holdout_search.search_with_holdout`, exactly like
every other candidate space in this codebase.

Core-only: no `rde_domains` import, same boundary as everywhere else in
`src/rde/`. See `rde/testing/vm_toy.py` for an end-to-end demonstration.
"""

from __future__ import annotations

from rde.substrate.enumeration import enumerate_programs
from rde.substrate.program import OPCODES, Instruction, Program
from rde.substrate.vm import ExecutionTrace, ResourceExceeded, execute

__all__ = [
    "OPCODES",
    "ExecutionTrace",
    "Instruction",
    "Program",
    "ResourceExceeded",
    "enumerate_programs",
    "execute",
]
