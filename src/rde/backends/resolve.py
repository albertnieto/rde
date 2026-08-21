"""Single source of truth for compute / expression backend selection.

On Apple Silicon, **MLX is the primary GPU path** for this project (pipeline
enumeration, expression DSL, discovery/GP). PyTorch MPS exists only as an
explicit compatibility opt-in — never as the default and never as the target
of generic ``mps`` / ``gpu`` aliases.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

_LOG = logging.getLogger(__name__)

ComputeBackendName = Literal["numpy", "mlx", "torch_mps", "torch_cpu", "auto"]
ExprBackendName = Literal["numpy", "mlx", "torch", "torch_mps"]

# Benchmark crossover: MLX wins full_instance only at N >= 14 on Apple Silicon.
MLX_GPU_CROSSOVER_N = 14

_COMPUTE_BACKENDS: tuple[str, ...] = ("numpy", "mlx", "torch_mps", "torch_cpu", "auto")
_EXPR_BACKENDS: tuple[str, ...] = ("numpy", "mlx", "torch", "torch_mps")
_GPU_ALIASES: frozenset[str] = frozenset({"mps", "gpu", "apple", "metal", "accelerator"})
_GPU_COMPUTE_BACKENDS: frozenset[str] = frozenset({"mlx", "torch_mps"})
_GPU_EXPR_BACKENDS: frozenset[str] = frozenset({"mlx", "torch", "torch_mps"})


class GpuBackendError(RuntimeError):
    """Raised when a GPU backend was requested but is unavailable."""


def mlx_usable() -> bool:
    try:
        from rde.expression.mlx_eval import mlx_runtime_available

        return mlx_runtime_available()
    except ImportError:
        return False


@dataclass(frozen=True)
class DevMachineProfile:
    """Runtime profile for the machine executing this process.

    Agents must call :func:`dev_machine_profile` (or ``rde machine-profile``)
    at session start. Do not assume Intel vs Apple Silicon from prose in
    ``CLAUDE.md`` — the repo is developed on both.
    """

    platform_system: str
    platform_machine: str
    profile_id: str
    mlx_usable: bool
    default_compute_backend: str
    default_expr_backend: str
    rde_gpu_primary: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dev_machine_profile() -> DevMachineProfile:
    """Detect the active dev machine and recommended RDE backends."""
    machine = platform.machine().lower()
    system = sys.platform
    usable = mlx_usable()
    if system == "darwin" and machine == "arm64":
        profile_id = "apple_silicon_mac"
        gpu_primary = "mlx"
    elif system == "darwin" and machine in {"x86_64", "i386"}:
        profile_id = "intel_mac"
        gpu_primary = None
    else:
        profile_id = "other"
        gpu_primary = "mlx" if usable else None
    return DevMachineProfile(
        platform_system=system,
        platform_machine=machine,
        profile_id=profile_id,
        mlx_usable=usable,
        default_compute_backend=default_compute_backend(),
        default_expr_backend=default_expr_backend(),
        rde_gpu_primary=gpu_primary,
    )


def format_dev_machine_profile(*, as_json: bool = False) -> str:
    profile = dev_machine_profile()
    if as_json:
        return json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n"
    lines = [
        f"profile_id={profile.profile_id}",
        f"platform={profile.platform_system} machine={profile.platform_machine}",
        f"mlx_usable={profile.mlx_usable}",
        f"default_compute_backend={profile.default_compute_backend}",
        f"default_expr_backend={profile.default_expr_backend}",
        f"rde_gpu_primary={profile.rde_gpu_primary or 'none'}",
    ]
    if profile.profile_id == "apple_silicon_mac":
        lines.append("policy=use --backend mlx (or auto) for RDE compute on this machine")
    elif profile.profile_id == "intel_mac":
        lines.append("policy=use --backend numpy on this machine; MLX is not installed by design")
    return "\n".join(lines) + "\n"


def require_mlx(*, context: str = "MLX GPU backend") -> None:
    """Raise if MLX/Metal is not usable."""
    if mlx_usable():
        return
    raise GpuBackendError(
        f"{context} requires MLX with Metal GPU access. "
        "Install mlx on Apple Silicon or pass an explicit CPU backend (numpy / torch_cpu)."
    )


def is_gpu_compute_backend(name: str) -> bool:
    return _normalize_key(name) in _GPU_COMPUTE_BACKENDS | _GPU_ALIASES


def is_gpu_expr_backend(name: str) -> bool:
    key = _normalize_key(name)
    return key in _GPU_EXPR_BACKENDS | _GPU_ALIASES


def default_compute_backend() -> ComputeBackendName:
    """Size-agnostic default: auto-select per problem size at run time."""
    return "auto"  # type: ignore[return-value]


def compute_backend_for_size(size: int, *, batch_size: int = 1) -> ComputeBackendName:
    """Pick numpy vs mlx from calibrated (N, B) table or size-only fallback."""
    from rde.backends.crossover_table import backend_for_size_and_batch

    choice = backend_for_size_and_batch(size, batch_size)
    if choice == "mlx" and not mlx_usable():
        _LOG.warning(
            "compute backend fallback: requested=auto recommended=mlx effective=numpy "
            "reason=mlx_unusable size=%d batch_size=%d",
            size,
            batch_size,
        )
        return "numpy"  # type: ignore[return-value]
    return choice  # type: ignore[return-value]


def recommended_compute_batch_size(size: int, *, prefer: str | None = None) -> int:
    from rde.backends.crossover_table import recommended_batch_size

    if prefer is None and mlx_usable():
        prefer = "mlx"
    return recommended_batch_size(size, prefer=prefer)


def default_expr_backend() -> ExprBackendName:
    return "mlx" if mlx_usable() else "numpy"  # type: ignore[return-value]


def compute_backend_choices() -> list[str]:
    return list(_COMPUTE_BACKENDS)


def expr_backend_choices() -> list[str]:
    return list(_EXPR_BACKENDS)


def _normalize_key(name: str) -> str:
    return name.lower().strip()


def _resolve_gpu_alias_to_mlx() -> ComputeBackendName:
    require_mlx(context="GPU compute backend alias (gpu/mps/metal)")
    return "mlx"  # type: ignore[return-value]


def resolve_compute_backend(
    name: str | None,
    *,
    size: int | None = None,
    batch_size: int = 1,
) -> ComputeBackendName:
    """Resolve pipeline/compute backend name.

    When ``name`` is ``None`` or ``auto``, pick numpy vs mlx from ``size`` and
    ``batch_size`` when provided; otherwise fall back to mlx when Metal is available.
    """
    if name is None:
        if size is not None:
            return compute_backend_for_size(size, batch_size=batch_size)
        return "mlx" if mlx_usable() else "numpy"  # type: ignore[return-value]
    key = _normalize_key(name)
    if key == "auto":
        if size is not None:
            return compute_backend_for_size(size, batch_size=batch_size)
        return "mlx" if mlx_usable() else "numpy"  # type: ignore[return-value]
    if key in _GPU_ALIASES:
        return _resolve_gpu_alias_to_mlx()
    if key == "mlx":
        require_mlx(context="compute backend mlx")
        return "mlx"  # type: ignore[return-value]
    if key in {"torch", "torch_mps"}:
        return "torch_mps"
    if key in {"torch_cpu", "cpu"}:
        return "torch_cpu"
    if key in _COMPUTE_BACKENDS:
        resolved = key  # type: ignore[assignment]
        if resolved == "mlx":
            require_mlx(context="compute backend mlx")
        return resolved  # type: ignore[return-value]
    if size is not None:
        return compute_backend_for_size(size, batch_size=batch_size)
    return "mlx" if mlx_usable() else "numpy"  # type: ignore[return-value]


def resolve_expr_backend(name: str | None) -> ExprBackendName:
    """Resolve expression-ranking / GP eval backend name."""
    if name is None:
        return default_expr_backend()
    key = _normalize_key(name)
    if key in _GPU_ALIASES:
        require_mlx(context="GPU expression backend alias (gpu/mps/metal)")
        return "mlx"  # type: ignore[return-value]
    if key == "mlx":
        require_mlx(context="expression backend mlx")
        return "mlx"  # type: ignore[return-value]
    if key in _EXPR_BACKENDS:
        return key  # type: ignore[return-value]
    if key in {"torch_cpu", "cpu"}:
        return "torch"
    if key in {"torch", "torch_mps"}:
        return "torch_mps"
    return default_expr_backend()
