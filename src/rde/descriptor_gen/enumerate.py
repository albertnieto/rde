"""Expanded parameterized descriptor template catalog."""

from __future__ import annotations

import math
from typing import Iterator

from rde.descriptor_gen.spec import DescriptorTemplate


def _spectral_templates() -> Iterator[DescriptorTemplate]:
    for i in range(7):
        for j in range(i + 1, 7):
            yield DescriptorTemplate(
                template_id=f"spectral.sv_ratio_{i}_{j}",
                family="spectral",
                key=f"gen.spectral.sv_ratio_{i}_{j}",
                params={"kind": "sv_ratio", "i": i, "j": j},
                description=f"Singular-value ratio s[{i}]/s[{j}]",
            )
    for k in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64):
        yield DescriptorTemplate(
            template_id=f"spectral.top_k_energy_{k}",
            family="spectral",
            key=f"gen.spectral.top_k_energy_{k}",
            params={"kind": "top_k_energy", "k": k},
            description=f"Top-{k} squared singular-value energy fraction",
        )
    for p in (1.0, 2.0, 4.0, 8.0):
        label = str(int(p))
        yield DescriptorTemplate(
            template_id=f"spectral.p_norm_{label}",
            family="spectral",
            key=f"gen.spectral.p_norm_{label}",
            params={"kind": "p_norm", "p": p},
            description=f"L{label} norm of singular-value profile",
        )
    yield DescriptorTemplate(
        template_id="spectral.effective_rank",
        family="spectral",
        key="gen.spectral.effective_rank",
        params={"kind": "effective_rank"},
        description="Participation-ratio effective rank of singular values",
    )
    yield DescriptorTemplate(
        template_id="spectral.gap",
        family="spectral",
        key="gen.spectral.gap",
        params={"kind": "gap"},
        description="Top-two singular-value gap",
    )


def _stats_templates() -> Iterator[DescriptorTemplate]:
    for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        q_label = str(q).replace(".", "p")
        yield DescriptorTemplate(
            template_id=f"stats.quantile_{q_label}",
            family="stats",
            key=f"gen.stats.quantile_{q_label}",
            params={"kind": "quantile", "q": q},
            description=f"Quantile q={q}",
        )
    for frac in (0.01, 0.05, 0.1, 0.15, 0.2, 0.25):
        frac_label = str(frac).replace(".", "p")
        yield DescriptorTemplate(
            template_id=f"stats.trimmed_mean_{frac_label}",
            family="stats",
            key=f"gen.stats.trimmed_mean_{frac_label}",
            params={"kind": "trimmed_mean", "frac": frac},
            description=f"Mean after trimming {frac} tails",
        )
    for bins in (4, 8, 16, 32, 64, 128):
        yield DescriptorTemplate(
            template_id=f"stats.hist_entropy_{bins}",
            family="stats",
            key=f"gen.stats.hist_entropy_{bins}",
            params={"kind": "histogram_entropy", "bins": bins},
            description=f"Histogram entropy with {bins} bins",
        )
    for kind in ("mad", "iqr"):
        yield DescriptorTemplate(
            template_id=f"stats.{kind}",
            family="stats",
            key=f"gen.stats.{kind}",
            params={"kind": kind},
            description=f"Stats {kind}",
        )


def _norm_templates() -> Iterator[DescriptorTemplate]:
    for p in (1.0, 2.0, 4.0, 8.0):
        yield DescriptorTemplate(
            template_id=f"norm.p_{int(p)}",
            family="norm",
            key=f"gen.norm.p_{int(p)}",
            params={"p": p},
            description=f"Vector L{int(p)} norm",
        )


def _fourier_templates() -> Iterator[DescriptorTemplate]:
    bands = (
        (0, 1),
        (1, 2),
        (2, 4),
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (0, 2),
        (0, 4),
        (2, 8),
        (4, 16),
        (8, 32),
        (0, 8),
        (0, 16),
        (0, 32),
    )
    for lo, hi in bands:
        yield DescriptorTemplate(
            template_id=f"fourier.band_{lo}_{hi}",
            family="fourier",
            key=f"gen.fourier.band_{lo}_{hi}",
            params={"lo": lo, "hi": hi},
            description=f"Walsh coefficient energy fraction in [{lo}, {hi})",
        )


def _autocorr_templates() -> Iterator[DescriptorTemplate]:
    for lag in range(1, 33):
        yield DescriptorTemplate(
            template_id=f"autocorr.lag_{lag}",
            family="autocorr",
            key=f"gen.autocorr.lag_{lag}",
            params={"lag": lag},
            description=f"Autocorrelation at lag {lag}",
        )


def _compress_templates() -> Iterator[DescriptorTemplate]:
    for kind in ("gzip_ratio", "lzma_ratio", "zlib_ratio"):
        yield DescriptorTemplate(
            template_id=f"compress.{kind}",
            family="compress",
            key=f"gen.compress.{kind}",
            params={"kind": kind},
            description=f"Compression ratio ({kind})",
        )


def _matrix_templates() -> Iterator[DescriptorTemplate]:
    for k in (2, 3, 4, 5, 6):
        yield DescriptorTemplate(
            template_id=f"matrix.trace_power_{k}",
            family="matrix",
            key=f"gen.matrix.trace_power_{k}",
            params={"kind": "trace_power", "k": k},
            description=f"Matrix trace power tr(Q^{k})",
        )
    yield DescriptorTemplate(
        template_id="matrix.fro_norm",
        family="matrix",
        key="gen.matrix.fro_norm",
        params={"kind": "fro_norm"},
        description="Matrix Frobenius norm",
    )
    yield DescriptorTemplate(
        template_id="matrix.off_diag_ratio",
        family="matrix",
        key="gen.matrix.off_diag_ratio",
        params={"kind": "off_diag_ratio"},
        description="Off-diagonal Frobenius mass fraction",
    )


def enumerate_descriptor_templates(*, max_templates: int | None = None) -> list[DescriptorTemplate]:
    """Return the full parameterized descriptor generator catalog."""
    out: list[DescriptorTemplate] = []
    sources = (
        _spectral_templates,
        _stats_templates,
        _norm_templates,
        _fourier_templates,
        _autocorr_templates,
        _compress_templates,
        _matrix_templates,
    )
    for source in sources:
        for template in source():
            out.append(template)
            if max_templates is not None and len(out) >= max_templates:
                return out
    return out


def default_pipeline_templates() -> list[DescriptorTemplate]:
    """Templates computed on every pipeline run."""
    return enumerate_descriptor_templates()


def catalog_size() -> int:
    return len(enumerate_descriptor_templates())
