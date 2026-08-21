"""Evaluate parameterized descriptor templates on array payloads."""

from __future__ import annotations

import math

import numpy as np

from rde.descriptor_gen.spec import DescriptorTemplate
from rde.features.fourier import walsh_hadamard
from rde.features.spectral import effective_rank
from rde.features.stats import shannon_entropy_histogram

_MASSIVE_FAMILY_ALIASES = {
    "massive_walsh": "fourier",
    "massive_spectral": "spectral",
    "massive_matrix": "matrix",
    "massive_autocorr": "autocorr",
    "massive_stats": "stats",
}
_UNSUPPORTED_MASSIVE_FAMILIES = frozenset(
    {"massive_alg", "massive_trig", "massive_operator", "massive_pad"}
)


def template_family_executable(family: str) -> bool:
    """True when ``evaluate_template`` can compute this family (not a stub/no-op)."""
    if family in _UNSUPPORTED_MASSIVE_FAMILIES:
        return False
    if family in _MASSIVE_FAMILY_ALIASES:
        return True
    return family in {
        "spectral",
        "stats",
        "norm",
        "fourier",
        "autocorr",
        "compress",
        "matrix",
    }


def _walsh_coeffs(vec: np.ndarray, cache: dict[int, np.ndarray] | None) -> np.ndarray:
    if cache is None:
        return walsh_hadamard(vec)
    key = id(vec)
    cached = cache.get(key)
    if cached is None:
        cached = walsh_hadamard(vec)
        cache[key] = cached
    return cached


def _as_1d(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=float).ravel()


def _singular_profile(
    array: np.ndarray,
    cache: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if cache is not None:
        key = id(arr)
        cached = cache.get(key)
        if cached is not None:
            return cached
    if arr.ndim == 1:
        profile = np.sort(np.abs(_as_1d(arr)))[::-1]
    elif arr.ndim == 2:
        profile = np.linalg.svd(arr, compute_uv=False)
    else:
        profile = _singular_profile(arr.ravel(), cache)
    if cache is not None:
        cache[id(arr)] = profile
    return profile


def _safe_ratio(num: float, den: float, eps: float = 1e-15) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= eps:
        return float("nan")
    return float(num / den)


def _vector_p_norm(values: np.ndarray, p: float) -> float:
    arr = np.abs(_as_1d(values))
    if arr.size == 0:
        return 0.0
    if p == math.inf:
        return float(np.max(arr))
    if p <= 0:
        return float("nan")
    return float(np.sum(arr**p) ** (1.0 / p))


def evaluate_template(
    template: DescriptorTemplate,
    array: np.ndarray | None,
    *,
    walsh_cache: dict[int, np.ndarray] | None = None,
    spectral_cache: dict[int, np.ndarray] | None = None,
    sorted_cache: dict[int, np.ndarray] | None = None,
) -> float:
    """Compute one template value; returns NaN when undefined."""
    if array is None:
        return float("nan")
    arr = np.asarray(array, dtype=float)
    if arr.size == 0:
        return float("nan")

    family = template.family
    if family in _UNSUPPORTED_MASSIVE_FAMILIES:
        return float("nan")
    family = _MASSIVE_FAMILY_ALIASES.get(family, family)
    params = template.params

    if family == "spectral":
        s = _singular_profile(arr, spectral_cache)
        s = s[s > 1e-15]
        if s.size == 0:
            return float("nan")
        kind = params["kind"]
        if kind == "sv_ratio":
            i, j = int(params["i"]), int(params["j"])
            if i >= s.size or j >= s.size:
                return float("nan")
            return _safe_ratio(float(s[i]), float(s[j]))
        if kind == "top_k_energy":
            k = int(params["k"])
            k = max(1, min(k, s.size))
            s2 = s**2
            return _safe_ratio(float(np.sum(s2[:k])), float(np.sum(s2)))
        if kind == "p_norm":
            return _vector_p_norm(s, float(params["p"]))
        if kind == "effective_rank":
            return effective_rank(s)
        if kind == "gap":
            if s.size < 2:
                return float("nan")
            return float(s[0] - s[1])

    if family == "stats":
        vec = _as_1d(arr)
        kind = params["kind"]
        if kind == "quantile":
            if sorted_cache is not None:
                key = id(arr)
                sorted_vec = sorted_cache.get(key)
                if sorted_vec is None:
                    sorted_vec = np.sort(vec)
                    sorted_cache[key] = sorted_vec
                return float(np.quantile(sorted_vec, float(params["q"])))
            return float(np.quantile(vec, float(params["q"])))
        if kind == "trimmed_mean":
            frac = float(params["frac"])
            if not 0.0 <= frac < 0.5:
                return float("nan")
            lo = int(np.floor(frac * vec.size))
            hi = int(np.ceil((1.0 - frac) * vec.size))
            trimmed = np.sort(vec)[lo:hi]
            if trimmed.size == 0:
                return float("nan")
            return float(np.mean(trimmed))
        if kind == "histogram_entropy":
            return shannon_entropy_histogram(vec, bins=int(params["bins"]))
        if kind == "mad":
            med = float(np.median(vec))
            return float(np.median(np.abs(vec - med)))
        if kind == "iqr":
            q1, q3 = np.quantile(vec, [0.25, 0.75])
            return float(q3 - q1)

    if family == "norm":
        return _vector_p_norm(arr, float(params["p"]))

    if family == "fourier":
        vec = _as_1d(arr)
        n = vec.size
        if n == 0 or (n & (n - 1)) != 0:
            return float("nan")
        pm1 = vec
        if np.all((vec >= 0) & (vec <= 1)):
            pm1 = 2.0 * vec - 1.0
        coeffs = _walsh_coeffs(pm1, walsh_cache)
        abs_c = np.abs(coeffs)
        total = float(np.sum(abs_c))
        if total <= 0:
            return float("nan")
        lo = int(params["lo"])
        hi = int(params["hi"])
        lo = max(0, min(lo, n))
        hi = max(lo + 1, min(hi, n))
        return _safe_ratio(float(np.sum(abs_c[lo:hi])), total)

    if family == "autocorr":
        from rde.features.numeric import safe_pearson_r

        vec = _as_1d(arr)
        lag = int(params["lag"])
        if lag < 1 or vec.size <= lag + 1:
            return float("nan")
        return safe_pearson_r(vec[:-lag], vec[lag:], min_count=2)

    if family == "compress":
        vec = _as_1d(arr)
        raw = vec.tobytes()
        raw_b = float(len(raw) * 8)
        if raw_b <= 0:
            return float("nan")
        import gzip
        import io
        import lzma

        kind = params["kind"]
        if kind == "gzip_ratio":
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                gz.write(raw)
            return float(len(buf.getvalue()) * 8 / raw_b)
        if kind == "lzma_ratio":
            return float(len(lzma.compress(raw)) * 8 / raw_b)
        if kind == "zlib_ratio":
            import zlib

            return float(len(zlib.compress(raw)) * 8 / raw_b)

    if family == "matrix":
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return float("nan")
        mat = arr.astype(float)
        kind = params["kind"]
        if kind == "trace_power":
            k = int(params["k"])
            return float(np.trace(np.linalg.matrix_power(mat, k)))
        if kind == "fro_norm":
            return float(np.linalg.norm(mat, ord="fro"))
        if kind == "off_diag_ratio":
            n = mat.shape[0]
            off = mat[~np.eye(n, dtype=bool)]
            return _safe_ratio(float(np.linalg.norm(off)), float(np.linalg.norm(mat)))

    return float("nan")


def evaluate_templates(
    templates: list[DescriptorTemplate],
    array: np.ndarray | None,
) -> dict[str, float]:
    """Evaluate many templates at once."""
    out: dict[str, float] = {}
    walsh_cache: dict[int, np.ndarray] = {}
    spectral_cache: dict[int, np.ndarray] = {}
    sorted_cache: dict[int, np.ndarray] = {}
    for template in templates:
        out[template.key] = evaluate_template(
            template,
            array,
            walsh_cache=walsh_cache,
            spectral_cache=spectral_cache,
            sorted_cache=sorted_cache,
        )
    return out
