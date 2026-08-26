"""Tests for the deterministic, resource-bounded stack machine (`rde.substrate.vm`)."""

from __future__ import annotations

import pytest

from rde.substrate.program import Instruction, Program
from rde.substrate.vm import ResourceExceeded, execute


def _prog(*ops: tuple[str, int]) -> Program:
    return Program(instructions=tuple(Instruction(op, operand) for op, operand in ops))


def test_empty_program_returns_input_unchanged():
    program = Program(instructions=())
    trace = execute(program, 7)
    assert trace.output == 7
    assert trace.halted is True


def test_halt_returns_top_of_stack():
    program = _prog(("HALT", 0))
    trace = execute(program, 42)
    assert trace.output == 42


def test_push_add_computes_offset():
    program = _prog(("PUSH", 3), ("ADD", 0), ("HALT", 0))
    trace = execute(program, 5)
    assert trace.output == 8


def test_sub_and_mul_and_mod():
    assert execute(_prog(("PUSH", 3), ("SUB", 0)), 10).output == 7
    assert execute(_prog(("PUSH", 3), ("MUL", 0)), 4).output == 12
    assert execute(_prog(("PUSH", 3), ("MOD", 0)), 10).output == 1


def test_mod_by_zero_raises():
    with pytest.raises(ResourceExceeded):
        execute(_prog(("PUSH", 0), ("MOD", 0)), 10)


def test_dup_and_swap():
    program = _prog(("DUP", 0), ("ADD", 0))
    assert execute(program, 6).output == 12

    program = _prog(("PUSH", 1), ("SWAP", 0), ("SUB", 0))
    # stack [10,1] -> swap -> [1,10] (top=10) -> SUB pops b=10 (top), a=1 -> a-b = 1-10.
    assert execute(program, 10).output == -9


def test_pop_on_empty_stack_raises():
    with pytest.raises(ResourceExceeded):
        execute(_prog(("POP", 0), ("POP", 0)), 1)


def test_dup_on_empty_stack_raises():
    with pytest.raises(ResourceExceeded):
        execute(_prog(("POP", 0), ("DUP", 0)), 1)


def test_store_and_load_round_trip_through_memory():
    program = _prog(("STORE", 0), ("PUSH", 1), ("LOAD", 0), ("ADD", 0))
    assert execute(program, 99).output == 100


def test_memory_index_out_of_bounds_raises():
    with pytest.raises(ResourceExceeded):
        execute(_prog(("STORE", 1000)), 1, max_memory=4)


def test_jump_out_of_bounds_raises():
    with pytest.raises(ResourceExceeded):
        execute(_prog(("JMP", 999)), 1)


def test_conditional_jump_skips_when_zero():
    # PUSH 0; JZ 4 (-> HALT); PUSH 1; ADD; HALT
    program = _prog(("PUSH", 0), ("JZ", 4), ("PUSH", 1), ("ADD", 0), ("HALT", 0))
    trace = execute(program, 5)
    assert trace.output == 5  # jumped straight to HALT, skipping the ADD


def test_infinite_loop_raises_resource_exceeded_not_hangs():
    program = _prog(("JMP", 0))
    with pytest.raises(ResourceExceeded):
        execute(program, 1, max_steps=100)


def test_execution_is_deterministic():
    program = _prog(("PUSH", 2), ("MUL", 0), ("HALT", 0))
    first = execute(program, 21)
    second = execute(program, 21)
    assert first == second
