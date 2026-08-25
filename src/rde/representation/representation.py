"""`Representation`: a formal encoding of an object type.

A representation is not just a feature-extraction function. It declares its
carrier, its encode/decode pair, whether it claims to be exact, what it
preserves, and (optionally) how to measure distance between two decoded
objects and the description complexity of an encoded value.

The only structural requirement of an exact representation is the roundtrip
law `decode(encode(x)) == x`; see `rde.representation.equivalence` for how
that is checked, and `rde.representation.certificate` for recording the
result.

`input_carrier_kind`/`carrier_kind` type a representation's encode input and
decode input by shape/dtype family (`"real_vector"`, `"complex_vector"`,
`"matrix"`, `"sorted_pair"`, ...) — both default to `"real_vector"`, correct
for every `grammar.py` primitive except the few that produce something else
(`dft`/`dft_full` -> `"complex_vector"`, `matrix_reshape` -> `"matrix"`,
`sorted_permutation` -> `"sorted_pair"`). `layered.py` uses these to compose
a second-stage `Representation` onto a first-stage one only when the second
stage's `input_carrier_kind` actually matches the first stage's
`carrier_kind` — see that module for why this typing is what makes deeper
composition meaningful (unlike flat composition over `grammar.py`'s
single-object-type primitives, which `search.py` documents as vacuous).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def default_distance(a: Any, b: Any) -> float:
    """Max-abs-difference distance for array-like objects.

    Domains with non-numeric carriers (graphs, symbolic expressions, ...)
    must supply their own `distance` — this default only handles the
    array-like case Phase 1's reference representations use.
    """
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


@dataclass(frozen=True)
class Representation:
    """One way of encoding objects of `object_type`.

    `encode` and `decode` operate on raw object values (e.g. a numpy array),
    not on `Object` records — `Object` is identity/metadata, this is the
    carrier-level mapping.
    """

    representation_id: str
    object_type: str
    carrier: str
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]
    exact: bool = True
    invariants: tuple[str, ...] = ()
    distance: Callable[[Any, Any], float] = field(default=default_distance)
    complexity: Callable[[Any], float] | None = None
    input_carrier_kind: str = "real_vector"
    carrier_kind: str = "real_vector"
