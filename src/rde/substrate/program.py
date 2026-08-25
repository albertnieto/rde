"""`Program`: the candidate type for `rde.substrate.vm`.

An opaque, hashable sequence of instructions -- deliberately no
mathematical meaning attached (no `Vector`/`Function`/`Generator` type
anywhere in this package): whatever behavior a `Program` has comes from
executing it (`rde.substrate.vm.execute`) and comparing that behavior
against a caller-supplied verifier, same as every other candidate type
`rde.search` operates on.
"""

from __future__ import annotations

from dataclasses import dataclass

OPCODES = (
    "PUSH",
    "POP",
    "DUP",
    "SWAP",
    "ADD",
    "SUB",
    "MUL",
    "MOD",
    "JMP",
    "JZ",
    "LOAD",
    "STORE",
    "HALT",
)


@dataclass(frozen=True)
class Instruction:
    opcode: str
    operand: int = 0

    def __post_init__(self) -> None:
        if self.opcode not in OPCODES:
            raise ValueError(f"unknown opcode: {self.opcode!r}")


@dataclass(frozen=True)
class Program:
    """An ordered, fixed instruction sequence -- the substrate's candidate type."""

    instructions: tuple[Instruction, ...]

    @property
    def program_id(self) -> str:
        """Stable textual id -- the `candidate_id` `rde.search` keys results on."""
        body = ",".join(f"{ins.opcode}:{ins.operand}" for ins in self.instructions)
        return f"prog[{body}]"

    def __len__(self) -> int:
        return len(self.instructions)
