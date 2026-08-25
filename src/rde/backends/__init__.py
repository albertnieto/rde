"""Compute backend registry and selection."""

from __future__ import annotations

from rde.backends.base import ComputeBackend
from rde.backends.crossover_table import (
    backend_for_size_and_batch,
    load_crossover_table,
    recommended_batch_size,
    write_crossover_table,
)
from rde.backends.numpy_backend import NumpyBackend
from rde.backends.resolve import (
    DevMachineProfile,
    GpuBackendError,
    compute_backend_choices,
    compute_backend_for_size,
    default_compute_backend,
    default_expr_backend,
    dev_machine_profile,
    expr_backend_choices,
    format_dev_machine_profile,
    is_gpu_compute_backend,
    is_gpu_expr_backend,
    mlx_usable,
    MLX_GPU_CROSSOVER_N,
    recommended_compute_batch_size,
    require_mlx,
    resolve_compute_backend,
    resolve_expr_backend,
    usable_backends,
)


def get_compute_backend(name: str | None = None, *, size: int | None = None) -> ComputeBackend:
    key = resolve_compute_backend(name, size=size)
    if key == "numpy":
        return NumpyBackend()
    if key == "mlx":
        from rde.backends.mlx_backend import MlxBackend

        return MlxBackend()
    if key == "torch_mps":
        from rde.backends.torch_mps_backend import TorchMpsBackend

        return TorchMpsBackend(device="mps", strict_device=True)
    if key == "torch_cpu":
        from rde.backends.torch_mps_backend import TorchMpsBackend

        return TorchMpsBackend(device="cpu")
    raise ValueError(f"Unknown compute backend: {key!r}")


def available_backends(*, include_auto: bool = True) -> list[str]:
    choices = [b for b in compute_backend_choices() if b != "auto"]
    if include_auto:
        return ["auto", *choices]
    return choices


def default_backend() -> str:
    """Auto-select numpy vs mlx from problem size at run time."""
    return default_compute_backend()


__all__ = [
    "ComputeBackend",
    "DevMachineProfile",
    "GpuBackendError",
    "MLX_GPU_CROSSOVER_N",
    "available_backends",
    "backend_for_size_and_batch",
    "compute_backend_choices",
    "compute_backend_for_size",
    "default_backend",
    "default_compute_backend",
    "default_expr_backend",
    "dev_machine_profile",
    "expr_backend_choices",
    "format_dev_machine_profile",
    "get_compute_backend",
    "is_gpu_compute_backend",
    "is_gpu_expr_backend",
    "load_crossover_table",
    "mlx_usable",
    "recommended_batch_size",
    "recommended_compute_batch_size",
    "require_mlx",
    "resolve_compute_backend",
    "resolve_expr_backend",
    "usable_backends",
    "write_crossover_table",
]
