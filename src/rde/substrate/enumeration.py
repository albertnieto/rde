"""Bounded brute-force enumeration of `rde.substrate.program.Program`s.

Deliberately the simplest possible candidate source for `rde.search` --
exhaustive over `{opcodes} x {operand range}` up to a fixed instruction
count, no mutation/crossover/learned proposal. This is a genuine instance of
an ontology-free candidate space (candidates are bare instruction
sequences, not typed grammar primitives), offered as an *alternative*, not
a replacement, to `rde.representation.program_search`'s typed-grammar
enumeration -- both plug into the same
`rde.search.holdout_search.search_with_holdout` engine, so which one a
caller uses is a runtime choice, not an architectural commitment.

Candidate count grows as roughly `O((n_opcodes * operand_range) **
max_length)`, so `max_length` and `operand_range` must stay small -- this
does not scale past short programs, and is not meant to; see this module's
tests for the sizes actually exercised.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from rde.substrate.program import OPCODES, Instruction, Program

_OPERAND_OPCODES = {"PUSH", "JMP", "JZ", "LOAD", "STORE"}


def _instruction_choices(operand_range: Sequence[int]) -> list[Instruction]:
    choices: list[Instruction] = []
    for opcode in OPCODES:
        if opcode in _OPERAND_OPCODES:
            choices.extend(Instruction(opcode, operand) for operand in operand_range)
        else:
            choices.append(Instruction(opcode))
    return choices


def enumerate_programs(
    max_length: int,
    *,
    operand_range: Sequence[int] = range(0, 4),
    require_halt: bool = True,
) -> Iterator[Program]:
    """Every program of length 1..`max_length` over a bounded instruction/operand alphabet.

    `require_halt`, when true (the default), only yields programs whose
    final instruction is `HALT` -- `rde.substrate.vm.execute` treats falling
    off the end the same as an explicit `HALT`, so this only trims
    otherwise-duplicate candidates (`X` and `X, HALT` behave identically),
    not the reachable behavior space.
    """
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    choices = _instruction_choices(operand_range)

    def _extend(prefix: tuple[Instruction, ...]) -> Iterator[Program]:
        if not require_halt or (prefix and prefix[-1].opcode == "HALT"):
            yield Program(instructions=prefix)
        if len(prefix) >= max_length:
            return
        for instruction in choices:
            yield from _extend(prefix + (instruction,))

    yield from _extend(())
