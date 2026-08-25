"""PyTorch MPS/CPU backend for batched real-valued enumeration."""

from __future__ import annotations

import math

import numpy as np

from rde.backends.base import FullInstanceBatch, full_instance_batch_fallback
from rde.backends.resolve import GpuBackendError


class TorchDeviceUnavailable(GpuBackendError):
    """Raised when a requested torch device is not available."""


def _torch_device(requested: str, *, strict: bool = False):
    import torch

    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if strict:
            raise TorchDeviceUnavailable("PyTorch MPS requested but torch.backends.mps.is_available() is False")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if strict:
            raise TorchDeviceUnavailable("PyTorch CUDA requested but torch.cuda.is_available() is False")
    if strict and requested in {"mps", "cuda"}:
        raise TorchDeviceUnavailable(f"PyTorch device {requested!r} unavailable")
    return torch.device("cpu")


def _torch_float_dtype(device):
    import torch

    # MPS does not support float64; CPU/CUDA use float64 for NumPy parity.
    return torch.float32 if device.type == "mps" else torch.float64


def _as_numpy(tensor) -> np.ndarray:
    """Host copy without per-element Python (torch 2.2 + NumPy 2.x workaround)."""
    import torch

    t = tensor.detach().cpu().contiguous().to(torch.float64).view(-1)
    try:
        return t.numpy(force=True).astype(float, copy=False).reshape(tensor.shape)
    except (RuntimeError, TypeError):
        # torch's Tensor.numpy() bridge is a C-API call into NumPy's array
        # struct; a torch wheel built against NumPy 1.x's ABI raises here
        # under NumPy 2.x (no newer torch wheel exists for Intel macOS to
        # upgrade past). Read the tensor's own memory directly via its
        # data pointer instead -- a bulk `ctypes` copy, not the broken
        # bridge, and still O(n) bytes, not a Python-level per-element loop.
        import ctypes

        nbytes = t.numel() * t.element_size()
        buf = (ctypes.c_byte * nbytes).from_address(t.data_ptr())
        raw = bytes(buf)
        return np.frombuffer(raw, dtype=np.float64).copy().reshape(tensor.shape)


def _assignment_matrix(n: int, device, dtype):
    """Build (2^n, n) assignment matrix on CPU, then move to compute device."""
    import torch

    nums = torch.arange(2**n, dtype=torch.int64)
    bits = torch.arange(n, dtype=torch.int64)
    x = ((nums.unsqueeze(1) >> bits.unsqueeze(0)) & 1).to(dtype=dtype, device=device)
    return x


class TorchMpsBackend:
    name = "torch_mps"

    def __init__(self, device: str = "mps", *, strict_device: bool = False) -> None:
        self._requested = device
        self._strict_device = strict_device
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._device = _torch_device(self._requested, strict=self._strict_device)
        return self._device

    @property
    def device_name(self) -> str:
        return str(self.device)

    def all_costs(self, Q: np.ndarray) -> np.ndarray:
        import torch

        q = np.asarray(Q, dtype=float)
        n = q.shape[0]
        dtype = _torch_float_dtype(self.device)
        x = _assignment_matrix(n, self.device, dtype)
        q_t = torch.tensor(q, device=self.device, dtype=dtype)
        costs = torch.sum(x * (x @ q_t), dim=1)
        return _as_numpy(costs)

    def bound_normalized_weights(
        self, Q: np.ndarray, *, costs: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        import torch

        n = Q.shape[0]
        qm = float(np.max(np.abs(Q)))
        if qm == 0:
            ones = np.ones(2**n)
            return ones, ones, 0.0, 0.0, math.pi / 2
        a = math.pi / (2 * n**2 * qm)
        b = math.pi / 2
        if costs is None:
            costs = self.all_costs(Q)
        dtype = _torch_float_dtype(self.device)
        theta = torch.tensor((a * costs + b) / 2.0, device=self.device, dtype=dtype)
        w_amp = _as_numpy(torch.cos(theta))
        return w_amp, w_amp**2, qm, a, b

    def trajectory_states(self, w_amp: np.ndarray, indices: list[int]) -> dict[int, np.ndarray]:
        import torch

        if not indices:
            return {}
        dtype = _torch_float_dtype(self.device)
        w = torch.tensor(w_amp, device=self.device, dtype=dtype)
        rs = torch.tensor(indices, device=self.device, dtype=dtype)
        powered = w.unsqueeze(0) ** rs.unsqueeze(1)
        norms = torch.linalg.norm(powered, dim=1, keepdim=True)
        norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        powered = powered / norms
        arr = _as_numpy(powered)
        return {r: arr[i] for i, r in enumerate(indices)}

    def full_instance_batch(self, Qs: np.ndarray, indices: list[int]) -> FullInstanceBatch:
        return full_instance_batch_fallback(self, Qs, indices)
