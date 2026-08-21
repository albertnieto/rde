"""Lazy streaming descriptor template catalog.

The broad catalog is deliberately made from real evaluator families.  Older
versions reached a nominal 100k budget by appending ``massive_pad`` records
that the evaluator could never execute; those records are no longer emitted.
"""

from __future__ import annotations

import warnings
from typing import Iterator

from rde.descriptor_gen.enumerate import enumerate_descriptor_templates
from rde.descriptor_gen.spec import DescriptorTemplate
from rde.descriptor_gen.compute import template_family_executable

# The default is a bounded, reproducible catalog rather than an invitation to
# allocate an unbounded list.  Every one of these templates has an evaluator.
MASSIVE_CATALOG_TARGET = 100_000
MASSIVE_CATALOG_VERSION = "massive-v2"


def _autocorr_templates() -> Iterator[DescriptorTemplate]:
    for lag in range(1, 65):
        yield DescriptorTemplate(
            template_id=f"massive.autocorr.lag_{lag}",
            family="massive_autocorr",
            key=f"gen.massive.autocorr.lag_{lag}",
            params={"kind": "autocorr", "lag": lag},
            description=f"Autocorrelation lag {lag}",
        )


def _walsh_band_templates() -> Iterator[DescriptorTemplate]:
    for lo in range(0, 64):
        for hi in range(lo + 1, min(lo + 9, 65)):
            yield DescriptorTemplate(
                template_id=f"massive.walsh.{lo}_{hi}",
                family="massive_walsh",
                key=f"gen.massive.walsh.{lo}_{hi}",
                params={"lo": lo, "hi": hi},
                description=f"Walsh band [{lo},{hi}) energy",
            )


def _matrix_operator_templates() -> Iterator[DescriptorTemplate]:
    for k in range(2, 13):
        yield DescriptorTemplate(
            template_id=f"massive.matrix.trace_power_{k}",
            family="massive_matrix",
            key=f"massive.matrix.trace_power_{k}",
            params={"kind": "trace_power", "k": k},
            description=f"Matrix trace power k={k}",
        )
    for i in range(7):
        for j in range(7):
            yield DescriptorTemplate(
                template_id=f"massive.svd.ratio_{i}_{j}",
                family="massive_spectral",
                key=f"gen.massive.svd.ratio_{i}_{j}",
                params={"kind": "sv_ratio", "i": i, "j": j},
                description=f"SVD ratio s[{i}]/s[{j}]",
            )


def _massive_quantile_templates() -> Iterator[DescriptorTemplate]:
    """A large executable grid of distinct order-statistic descriptors.

    Quantile evaluation is a genuine reduction over the input array.  The
    dense rational grid is useful for stress-testing catalog and null handling
    without inventing fake descriptors; callers still receive the exact
    ``q`` parameter in the provenance record.
    """
    denominator = MASSIVE_CATALOG_TARGET + 1
    for numerator in range(1, denominator):
        q_label = f"{numerator:06d}"
        yield DescriptorTemplate(
            template_id=f"massive.stats.quantile_{q_label}",
            family="massive_stats",
            key=f"gen.massive.stats.quantile_{q_label}",
            params={
                "kind": "quantile",
                "q": numerator / denominator,
                "grid_numerator": numerator,
                "grid_denominator": denominator,
                "catalog_version": MASSIVE_CATALOG_VERSION,
            },
            description=f"Massive quantile q={numerator}/{denominator}",
        )


def enumerate_massive_templates(
    *,
    max_templates: int | None = None,
    pad_to_target: bool = False,
) -> list[DescriptorTemplate]:
    """Return a bounded list of executable templates.

    ``pad_to_target`` remains accepted for source compatibility with old
    callers, but is intentionally ignored: padding is not a valid descriptor
    family and cannot be used to reach the broad-search budget.
    """
    if max_templates is not None and max_templates < 0:
        raise ValueError("max_templates must be non-negative or None")
    cap = max_templates if max_templates is not None else MASSIVE_CATALOG_TARGET
    if cap == 0:
        return []
    out: list[DescriptorTemplate] = []
    streams = (
        enumerate_descriptor_templates(),
        _autocorr_templates(),
        _walsh_band_templates(),
        _matrix_operator_templates(),
        _massive_quantile_templates(),
    )
    seen: set[str] = set()
    seen_canonical: set[str] = set()
    for stream in streams:
        for template in stream:
            if template.template_id in seen:
                continue
            if not template_family_executable(template.family):
                continue
            if template.canonical_id in seen_canonical:
                continue
            seen.add(template.template_id)
            seen_canonical.add(template.canonical_id)
            out.append(template)
            if len(out) >= cap:
                return out
    return out


def executable_massive_catalog_size(*, max_templates: int | None = None) -> int:
    """Count templates that ``evaluate_template`` can actually compute."""
    return len(enumerate_massive_templates(max_templates=max_templates, pad_to_target=False))


def massive_catalog_size(*, pad_to_target: bool = False) -> int:
    """Return the actual executable count; legacy padding is ignored."""
    return len(enumerate_massive_templates(pad_to_target=pad_to_target))
