"""A tiny, deterministic, resource-bounded stack machine.

Deliberately minimal -- this is a *baseline* candidate execution substrate
for `rde.search`, not a claim it scales past small instruction counts (see
`rde.substrate.enumeration`'s module docstring for the same honest framing
`rde.representation.program_search` already applies to its own exhaustive,
not heuristic, search over typed primitives). `execute` is total and
deterministic: it either halts within `max_steps` or raises
`ResourceExceeded` -- it never loops forever and never silently truncates.
"""

from __future__ import annotations

from dataclasses import dataclass

from rde.substrate.program import Program


class ResourceExceeded(Exception):
    """`execute` hit a resource limit, or an instruction was underspecified, before halting."""


@dataclass(frozen=True)
class ExecutionTrace:
    output: int
    steps_used: int
    halted: bool


def execute(
    program: Program,
    input_value: int,
    *,
    max_steps: int = 1024,
    max_memory: int = 64,
) -> ExecutionTrace:
    """Run `program` on `input_value`, deterministically, within bounded resources.

    Stack starts as `[input_value]`; memory is a fixed-size zero-initialized
    integer array of length `max_memory`. `HALT` (or falling off the end of
    the instruction list) stops execution and returns the top of stack (0
    if empty) as `output`. Any instruction that would exceed `max_steps`,
    index memory or jump out of bounds, pop an empty stack, or divide/mod by
    zero raises `ResourceExceeded` -- an under-specified program is a failed
    candidate for the caller's verifier to drop, not a crash to catch
    elsewhere.
    """
    stack: list[int] = [int(input_value)]
    memory = [0] * max_memory
    pc = 0
    steps = 0
    instructions = program.instructions
    n = len(instructions)

    def _pop() -> int:
        if not stack:
            raise ResourceExceeded("pop on empty stack")
        return stack.pop()

    def _check_target(target: int) -> None:
        if not (0 <= target <= n):
            raise ResourceExceeded(f"jump target out of bounds: {target}")

    while pc < n:
        if steps >= max_steps:
            raise ResourceExceeded(f"exceeded max_steps={max_steps}")
        steps += 1
        ins = instructions[pc]
        op = ins.opcode
        if op == "HALT":
            break
        elif op == "PUSH":
            stack.append(int(ins.operand))
        elif op == "POP":
            _pop()
        elif op == "DUP":
            if not stack:
                raise ResourceExceeded("dup on empty stack")
            stack.append(stack[-1])
        elif op == "SWAP":
            if len(stack) < 2:
                raise ResourceExceeded("swap on stack of size < 2")
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif op == "ADD":
            b, a = _pop(), _pop()
            stack.append(a + b)
        elif op == "SUB":
            b, a = _pop(), _pop()
            stack.append(a - b)
        elif op == "MUL":
            b, a = _pop(), _pop()
            stack.append(a * b)
        elif op == "MOD":
            b, a = _pop(), _pop()
            if b == 0:
                raise ResourceExceeded("mod by zero")
            stack.append(a % b)
        elif op == "JMP":
            _check_target(ins.operand)
            pc = ins.operand
            continue
        elif op == "JZ":
            top = _pop()
            if top == 0:
                _check_target(ins.operand)
                pc = ins.operand
                continue
        elif op == "LOAD":
            if not (0 <= ins.operand < max_memory):
                raise ResourceExceeded(f"memory index out of bounds: {ins.operand}")
            stack.append(memory[ins.operand])
        elif op == "STORE":
            if not (0 <= ins.operand < max_memory):
                raise ResourceExceeded(f"memory index out of bounds: {ins.operand}")
            memory[ins.operand] = _pop()
        else:  # pragma: no cover - unreachable, Instruction validates opcode
            raise AssertionError(f"unhandled opcode {op!r}")
        pc += 1

    output = stack[-1] if stack else 0
    return ExecutionTrace(output=output, steps_used=steps, halted=True)
