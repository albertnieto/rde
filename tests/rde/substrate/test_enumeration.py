"""Tests for bounded brute-force program enumeration and the VM-toy end-to-end demo."""

from __future__ import annotations

import pytest

from rde.substrate.enumeration import enumerate_programs
from rde.substrate.program import Instruction, Program
from rde.substrate.vm import execute
from rde.testing.vm_toy import make_batches, rediscover_offset


def test_rejects_non_positive_max_length():
    with pytest.raises(ValueError):
        list(enumerate_programs(0))


def test_every_yielded_program_ends_in_halt_by_default():
    programs = list(enumerate_programs(2, operand_range=range(0, 2)))
    assert programs  # non-empty
    for program in programs:
        assert program.instructions[-1].opcode == "HALT"


def test_require_halt_false_yields_every_prefix_too():
    with_halt = list(enumerate_programs(2, operand_range=range(0, 2), require_halt=True))
    without = list(enumerate_programs(2, operand_range=range(0, 2), require_halt=False))
    assert len(without) > len(with_halt)


def test_require_halt_false_never_yields_the_empty_program():
    # "Every program of length 1..max_length" is the documented contract --
    # a length-0 program is not a valid candidate under either setting.
    programs = list(enumerate_programs(2, operand_range=range(0, 1), require_halt=False))
    assert all(len(p) >= 1 for p in programs)


def test_identity_program_is_reachable_at_length_one():
    programs = list(enumerate_programs(1, operand_range=range(0, 2)))
    ids = {p.program_id for p in programs}
    identity = Program(instructions=(Instruction("HALT", 0),))
    assert identity.program_id in ids


def test_enumeration_is_a_generator_not_a_list_eagerly_built():
    gen = enumerate_programs(2, operand_range=range(0, 2))
    first = next(gen)
    assert isinstance(first, Program)


def test_make_batches_produces_offset_function_pairs():
    train, holdout = make_batches(3, xs_train=(0, 1), xs_holdout=(10,))
    assert train == ((0, 3), (1, 4))
    assert holdout == ((10, 13),)


def test_rediscover_offset_finds_a_verified_program_end_to_end():
    program = rediscover_offset(2, xs_train=(0, 1, 2), xs_holdout=(10, 11), max_length=3)
    assert program is not None
    # The returned program must actually compute f(x) = x + 2 on fresh inputs
    # never used during search -- the real claim `search_with_holdout` makes.
    for x in (5, 20, 100):
        assert execute(program, x).output == x + 2


def test_rediscover_offset_returns_none_when_offset_outside_search_space():
    # offset=99 cannot be reached by any PUSH within operand_range=range(0,4).
    program = rediscover_offset(99, max_length=3, operand_range=range(0, 4))
    assert program is None


def test_empty_batch_never_vacuously_verifies_every_candidate():
    # Zero evidence must not count as proof -- an empty train/holdout batch
    # should reject every candidate, not let the untested identity program
    # "win" as the shortest apparently-correct answer.
    program = rediscover_offset(5, xs_train=(), xs_holdout=(), max_length=2, operand_range=range(0, 2))
    assert program is None
