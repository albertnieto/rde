"""Compression-proxy descriptors (native gzip/lzma/zlib/bz2 only, v0.3)."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import lzma
import zlib

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice

COMPRESS_VERSION = "v0.3_native_only"


def _gzip_bits(data: bytes) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(data)
    return len(buf.getvalue()) * 8


def _lzma_bits(data: bytes) -> int:
    return len(lzma.compress(data)) * 8


def _zlib_bits(data: bytes) -> int:
    return len(zlib.compress(data)) * 8


def _bz2_bits(data: bytes) -> int:
    return len(bz2.compress(data)) * 8


def _quantize_canonical(arr: np.ndarray) -> bytes:
    """Fixed-point quantization for reproducible byte payloads."""
    a = np.asarray(arr, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return b""
    scale = max(float(np.max(np.abs(finite))), 1e-12)
    q = np.round(finite / scale * 1e6).astype("<i4")
    return q.tobytes()


def _kolmogorov_ensemble_bits(raw_b: float, compressors: list[float]) -> float:
    if raw_b <= 0:
        return float("nan")
    return float(min(compressors) / raw_b)


def descriptor(
    _instance: InstanceRecord,
    _slice_: SimpleFamilySlice | None,
    array: np.ndarray | None,
) -> dict[str, float]:
    if array is None:
        return {}
    raw = _quantize_canonical(array)
    gzip_b = float(_gzip_bits(raw))
    lzma_b = float(_lzma_bits(raw))
    zlib_b = float(_zlib_bits(raw))
    bz2_b = float(_bz2_bits(raw))
    raw_b = float(len(raw) * 8)
    bits = [gzip_b, lzma_b, zlib_b, bz2_b]
    kolmogorov = _kolmogorov_ensemble_bits(raw_b, bits)
    return {
        "compress.version": float(
            int.from_bytes(hashlib.sha256(COMPRESS_VERSION.encode("utf-8")).digest()[:4], "little")
            % 1000
        ),
        "compress.raw_bytes": raw_b,
        "compress.gzip_bits": gzip_b,
        "compress.lzma_bits": lzma_b,
        "compress.zlib_bits": zlib_b,
        "compress.bz2_bits": bz2_b,
        "compress.mdl_proxy": min(bits),
        "compress.kolmogorov_proxy": kolmogorov,
        "compress.ratio_gzip": gzip_b / raw_b if raw_b > 0 else float("nan"),
        "compress.ratio_zlib": zlib_b / raw_b if raw_b > 0 else float("nan"),
        "compress.ratio_bz2": bz2_b / raw_b if raw_b > 0 else float("nan"),
    }
