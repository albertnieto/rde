"""`rde.representation` applied to real tsp distance data.

Domain-integration gap closure: `rde.representation`'s grammar/search/Pareto
machinery previously only ran on synthetic NumPy vectors built for its own
tests. This module runs it on the actual pairwise-distance profile of
generated Euclidean TSP instances (`components.random_euclidean_points` +
`components.distance_matrix`) — real geometric domain data, not a
surrogate built to make the demo work. Core (`rde.representation`) still
never imports `rde_domains`; this module lives on the domain side and
imports core, which is the direction the architecture already allows.

The object handed to `rde.representation` is the *upper-triangular* distance
vector (`n * (n-1) // 2` entries), not the raw `(n, n)` distance matrix —
that vector is what has a fixed length across independently generated
instances of the same `n_cities`, matching `rde.representation`'s
fixed-`N`-per-batch requirement. The full symmetric, zero-diagonal matrix is
recoverable from it, but recovering it is not implemented here — this
module only needs the flat profile as an object for representation search,
not a round-trip back to a `TspSynthesisDomain` instance.

Exploratory tooling, not a `DomainContract`-integrated experiment: this does
not add predictor columns to any tsp domain's feature contract and does not
run the leak-audit / held-out-family / preregistered-decision-rule discovery
loop `docs/experiment-playbook.md` requires of a "real experiment". It
answers a narrower, honest question — "does representation search find
anything on this domain's actual geometry" — not "here is a validated
discovery about TSP".
"""

from __future__ import annotations

import numpy as np

from rde.representation import SearchCandidate, rank_representations
from rde_domains.tsp.components import distance_matrix, random_euclidean_points


def upper_triangular_distances(D: np.ndarray) -> np.ndarray:
    """The `n * (n-1) / 2` pairwise distances of a symmetric, zero-diagonal distance matrix."""
    n = D.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return D[rows, cols]


def distance_profile_batch(
    rng: np.random.Generator, *, n_cities: int, batch_size: int = 16
) -> np.ndarray:
    """Stack `batch_size` independent Euclidean TSP instances' distance profiles.

    Each row has length `n_cities * (n_cities - 1) // 2`, fixed for a given
    `n_cities`. Generating `batch_size` independent random instances is a
    small (caller-bounded), not per-sample-data, Python loop; each
    instance's own `distance_matrix` computation is itself a vectorized
    pairwise-distance call.
    """
    profiles = [
        upper_triangular_distances(distance_matrix(random_euclidean_points(rng, n_cities)))
        for _ in range(batch_size)
    ]
    return np.stack(profiles, axis=0)


def rank_distance_profile_representations(
    rng: np.random.Generator,
    *,
    n_cities: int,
    batch_size: int = 16,
    tolerance: float = 1e-6,
) -> list[SearchCandidate]:
    """Rank `rde.representation`'s grammar against real Euclidean TSP distance profiles."""
    batch = distance_profile_batch(rng, n_cities=n_cities, batch_size=batch_size)
    n = batch.shape[1]
    return rank_representations(batch, n=n, tolerance=tolerance)
