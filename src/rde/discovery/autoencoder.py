"""Nonlinear autoencoders for features and instance Q (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from rde.backends.resolve import GpuBackendError, mlx_usable
from rde.discovery.checkpoint import save_latent_checkpoint
from rde.discovery.datasets import InstanceQBatch, load_instance_q_batch
from rde.discovery.impute import impute_matrix
from rde.features.numeric import safe_pearson_r


def latent_column_names(prefix: str, k: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(k)]


@dataclass
class AutoencoderResult:
    hidden_dim: int
    reconstruction_error: float
    train_r_squared: float
    target_correlation: float
    method: str = "random_features"
    latent_codes: np.ndarray = field(default_factory=lambda: np.array([]))
    latent_columns: list[str] = field(default_factory=list)
    checkpoint_name: str = "feature_ae"


@dataclass
class InstanceQResult:
    n_instances: int
    q_dim: int
    hidden_dim: int
    reconstruction_error: float
    train_r_squared: float
    target_correlation: float
    target_correlations: dict[str, float]
    method: str
    instance_ids: list[str] = field(default_factory=list)
    latent_codes: np.ndarray = field(default_factory=lambda: np.array([]))
    checkpoint_name: str = "instance_q_ae"


def random_feature_autoencoder(
    X: np.ndarray,
    *,
    target: np.ndarray | None = None,
    hidden_dim: int = 16,
    seed: int = 0,
) -> AutoencoderResult:
    """1-hidden-layer tanh autoencoder via random features + ridge decode."""
    X_clean = impute_matrix(X)
    mu = X_clean.mean(axis=0, keepdims=True)
    X_centered = X_clean - mu
    rng = np.random.default_rng(seed)
    k = min(hidden_dim, X_centered.shape[1], max(1, X_centered.shape[0] // 2))
    W = rng.normal(scale=0.5, size=(X_centered.shape[1], k))
    H = np.tanh(X_centered @ W)
    # Decode only from the latent representation.  Including X_centered in
    # this design makes reconstruction an identity shortcut and invalidates
    # the bottleneck diagnostic.
    design = np.hstack([np.ones((H.shape[0], 1)), H])
    reg = 1e-3 * np.eye(design.shape[1])
    coef = np.linalg.solve(design.T @ design + reg, design.T @ X_clean)
    recon = design @ coef
    mse = float(np.mean((X_clean - recon) ** 2))

    train_r2 = float("nan")
    target_corr = float("nan")
    if target is not None:
        target_arr = np.asarray(target, dtype=float)
        target_mask = np.isfinite(target_arr)
        t = target_arr[target_mask]
        if t.size < 3 or float(np.std(t)) <= 0.0:
            t = np.array([], dtype=float)
        # Predict target from hidden codes
        H_aug = np.hstack([np.ones((H.shape[0], 1)), H])
        H_fit = H_aug[target_mask] if t.size else np.empty((0, H_aug.shape[1]))
        if t.size == 0:
            return AutoencoderResult(
                hidden_dim=k,
                reconstruction_error=mse,
                train_r_squared=float("nan"),
                target_correlation=float("nan"),
                latent_codes=H,
            )
        reg_t = 1e-3 * np.eye(H_aug.shape[1])
        beta = np.linalg.solve(H_fit.T @ H_fit + reg_t, H_fit.T @ t)
        pred = H_fit @ beta
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        train_r2 = float(1.0 - np.sum((t - pred) ** 2) / ss_tot)
        if np.std(pred) > 0 and np.std(t) > 0:
            target_corr = safe_pearson_r(pred, t)

    return AutoencoderResult(
        hidden_dim=k,
        reconstruction_error=mse,
        train_r_squared=train_r2,
        target_correlation=target_corr,
        method="random_features",
        latent_codes=H,
        latent_columns=latent_column_names("latent.ae", k),
    )


def torch_mlp_autoencoder(
    X: np.ndarray,
    *,
    target: np.ndarray | None = None,
    hidden_dim: int = 16,
    epochs: int = 40,
    seed: int = 0,
    on_epoch: Callable[[int, int], None] | None = None,
    require_gpu: bool = False,
) -> AutoencoderResult | None:
    """Small torch MLP autoencoder; returns None if torch unavailable."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        if require_gpu:
            raise GpuBackendError("torch autoencoder required but torch is not installed") from exc
        return None

    X_clean = impute_matrix(X)
    mu = X_clean.mean(axis=0, keepdims=True)
    X_centered = X_clean - mu
    n, d = X_centered.shape
    k = min(hidden_dim, d, max(1, n // 2))
    if k < 1 or n < 4:
        return None

    torch.manual_seed(seed)
    from rde.backends.torch_mps_backend import _torch_device

    if require_gpu:
        device = _torch_device("mps", strict=True)
    else:
        device = _torch_device("cpu", strict=False)
        if device.type == "cpu" and torch.backends.mps.is_available():
            device = _torch_device("mps", strict=False)
        elif device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
    if require_gpu and device.type == "cpu":
        raise GpuBackendError("torch GPU autoencoder required but no MPS/CUDA device is available")
    xt = torch.tensor(X_centered, dtype=torch.float32, device=device)

    class _AE(nn.Module):
        def __init__(self, in_dim: int, latent: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(in_dim, latent), nn.Tanh())
            self.decoder = nn.Sequential(nn.Linear(latent, in_dim))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            return z, self.decoder(z)

    model = _AE(d, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        _, recon = model(xt)
        loss = loss_fn(recon, xt)
        loss.backward()
        opt.step()
        if on_epoch is not None:
            on_epoch(epoch + 1, epochs)

    model.eval()
    with torch.no_grad():
        z, recon = model(xt)
        mse = float(loss_fn(recon, xt).item())

    train_r2 = float("nan")
    target_corr = float("nan")
    if target is not None:
        target_arr = np.asarray(target, dtype=float)
        target_mask = np.isfinite(target_arr)
        t = target_arr[target_mask]
        if t.size < 3 or float(np.std(t)) <= 0.0:
            t = np.array([], dtype=float)
        z_np = z.cpu().numpy()
        H_aug = np.hstack([np.ones((z_np.shape[0], 1)), z_np])
        H_fit = H_aug[target_mask] if t.size else np.empty((0, H_aug.shape[1]))
        if not t.size:
            return AutoencoderResult(
                hidden_dim=k,
                reconstruction_error=mse,
                train_r_squared=float("nan"),
                target_correlation=float("nan"),
                method="torch_mlp",
                latent_codes=z_np,
                latent_columns=latent_column_names("latent.ae", k),
            )
        reg_t = 1e-3 * np.eye(H_aug.shape[1])
        beta = np.linalg.solve(H_fit.T @ H_fit + reg_t, H_fit.T @ t)
        pred = H_fit @ beta
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        train_r2 = float(1.0 - np.sum((t - pred) ** 2) / ss_tot)
        if np.std(pred) > 0 and np.std(t) > 0:
            target_corr = safe_pearson_r(pred, t)

    z_np = z.cpu().numpy()
    return AutoencoderResult(
        hidden_dim=k,
        reconstruction_error=mse,
        train_r_squared=train_r2,
        target_correlation=target_corr,
        method="torch_mlp",
        latent_codes=z_np,
        latent_columns=latent_column_names("latent.ae", k),
    )


def mlx_mlp_autoencoder(
    X: np.ndarray,
    *,
    target: np.ndarray | None = None,
    hidden_dim: int = 16,
    epochs: int = 40,
    seed: int = 0,
    on_epoch: Callable[[int, int], None] | None = None,
    require_gpu: bool = False,
) -> AutoencoderResult | None:
    """Small MLX MLP autoencoder; returns None if MLX unavailable."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError as exc:
        if require_gpu:
            raise GpuBackendError("MLX autoencoder required but mlx is not installed") from exc
        return None

    X_clean = impute_matrix(X)
    mu = X_clean.mean(axis=0, keepdims=True)
    X_centered = X_clean - mu
    n, d = X_centered.shape
    k = min(hidden_dim, d, max(1, n // 2))
    if k < 1 or n < 4:
        return None

    mx.random.seed(seed)

    class _AE(nn.Module):
        def __init__(self, in_dim: int, latent: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(in_dim, latent), nn.Tanh())
            self.decoder = nn.Sequential(nn.Linear(latent, in_dim))

        def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
            z = self.encoder(x)
            return z, self.decoder(z)

    model = _AE(d, k)
    opt = optim.Adam(learning_rate=1e-2)
    loss_fn = nn.losses.mse_loss
    x_mx = mx.array(X_centered, dtype=mx.float32)

    def train_loss(model: _AE) -> mx.array:
        _, recon = model(x_mx)
        return loss_fn(recon, x_mx)

    loss_and_grad_fn = mx.value_and_grad(train_loss)

    for epoch in range(epochs):
        loss, grads = loss_and_grad_fn(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if on_epoch is not None:
            on_epoch(epoch + 1, epochs)

    z_mx, recon_mx = model(x_mx)
    mse = float(mx.mean((recon_mx - x_mx) ** 2).item())
    z_np = np.array(z_mx)

    train_r2 = float("nan")
    target_corr = float("nan")
    if target is not None:
        target_arr = np.asarray(target, dtype=float)
        target_mask = np.isfinite(target_arr)
        t = target_arr[target_mask]
        if t.size < 3 or float(np.std(t)) <= 0.0:
            t = np.array([], dtype=float)
        H_aug = np.hstack([np.ones((z_np.shape[0], 1)), z_np])
        H_fit = H_aug[target_mask] if t.size else np.empty((0, H_aug.shape[1]))
        if not t.size:
            return AutoencoderResult(
                hidden_dim=k,
                reconstruction_error=mse,
                train_r_squared=float("nan"),
                target_correlation=float("nan"),
                method="mlx_mlp",
                latent_codes=z_np,
                latent_columns=latent_column_names("latent.ae", k),
            )
        reg_t = 1e-3 * np.eye(H_aug.shape[1])
        beta = np.linalg.solve(H_fit.T @ H_fit + reg_t, H_fit.T @ t)
        pred = H_fit @ beta
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        train_r2 = float(1.0 - np.sum((t - pred) ** 2) / ss_tot)
        if np.std(pred) > 0 and np.std(t) > 0:
            target_corr = safe_pearson_r(pred, t)

    return AutoencoderResult(
        hidden_dim=k,
        reconstruction_error=mse,
        train_r_squared=train_r2,
        target_correlation=target_corr,
        method="mlx_mlp",
        latent_codes=z_np,
        latent_columns=latent_column_names("latent.ae", k),
    )


def discover_autoencoder(
    X: np.ndarray,
    *,
    target: np.ndarray | None = None,
    hidden_dim: int = 16,
    prefer_torch: bool = False,
    prefer_mlx: bool | None = None,
    require_gpu: bool = False,
    on_epoch: Callable[[int, int], None] | None = None,
) -> AutoencoderResult:
    """Use MLX by default when available; retain random features as CPU fallback."""
    if prefer_mlx is None:
        prefer_mlx = mlx_usable()
    if prefer_mlx:
        mlx_result = mlx_mlp_autoencoder(
            X,
            target=target,
            hidden_dim=hidden_dim,
            on_epoch=on_epoch,
            require_gpu=require_gpu,
        )
        if mlx_result is not None:
            return mlx_result
        if require_gpu:
            raise GpuBackendError("MLX autoencoder required but training failed or Metal is unavailable")
    if prefer_torch:
        torch_result = torch_mlp_autoencoder(
            X,
            target=target,
            hidden_dim=hidden_dim,
            on_epoch=on_epoch,
            require_gpu=require_gpu,
        )
        if torch_result is not None:
            return torch_result
        if require_gpu:
            raise GpuBackendError("torch autoencoder required but training failed or GPU is unavailable")
    if require_gpu:
        raise GpuBackendError("GPU autoencoder required but neither MLX nor torch path succeeded")
    return random_feature_autoencoder(X, target=target, hidden_dim=hidden_dim)


def discover_autoencoder_with_checkpoint(
    run_id: str,
    store_root: Path | str,
    *,
    target: np.ndarray,
    X: np.ndarray,
    hidden_dim: int = 16,
    prefer_torch: bool = False,
    prefer_mlx: bool | None = None,
    require_gpu: bool = False,
    on_epoch: Callable[[int, int], None] | None = None,
    checkpoint: bool = True,
) -> AutoencoderResult:
    result = discover_autoencoder(
        X,
        target=target,
        hidden_dim=hidden_dim,
        prefer_torch=prefer_torch,
        prefer_mlx=prefer_mlx,
        require_gpu=require_gpu,
        on_epoch=on_epoch,
    )
    if checkpoint and result.latent_codes.size:
        save_latent_checkpoint(
            store_root,
            run_id,
            result.checkpoint_name,
            arrays={"latent_codes": result.latent_codes},
            metadata={
                "method": result.method,
                "hidden_dim": result.hidden_dim,
                "reconstruction_error": result.reconstruction_error,
                "train_r_squared": result.train_r_squared,
                "target_correlation": result.target_correlation,
            },
        )
    return result


def discover_instance_q_autoencoder(
    run_id: str,
    store_root: Path | str,
    *,
    target: str = "metric.representation_complexity",
    hidden_dim: int = 8,
    prefer_torch: bool = False,
    prefer_mlx: bool | None = None,
    require_gpu: bool = False,
    checkpoint: bool = True,
    on_epoch: Callable[[int, int], None] | None = None,
) -> InstanceQResult:
    """Instance-level Q autoencoder: flat upper-tri Q → latent; correlate with mean complexity."""
    batch = load_instance_q_batch(run_id, store_root, target=target)
    if batch.q_vectors.shape[0] < 4:
        return InstanceQResult(0, 0, 0, float("nan"), float("nan"), float("nan"), {}, "none")

    ae = discover_autoencoder(
        batch.q_vectors,
        target=batch.targets,
        hidden_dim=hidden_dim,
        prefer_torch=prefer_torch,
        prefer_mlx=prefer_mlx,
        require_gpu=require_gpu,
        on_epoch=on_epoch,
    )
    corrs: dict[str, float] = {}
    for i in range(ae.hidden_dim):
        col = ae.latent_codes[:, i]
        if np.std(col) > 0 and np.std(batch.targets) > 0:
            corrs[f"latent_q_{i}"] = safe_pearson_r(col, batch.targets)

    result = InstanceQResult(
        n_instances=int(batch.q_vectors.shape[0]),
        q_dim=int(batch.q_vectors.shape[1]),
        hidden_dim=ae.hidden_dim,
        reconstruction_error=ae.reconstruction_error,
        train_r_squared=ae.train_r_squared,
        target_correlation=ae.target_correlation,
        target_correlations=corrs,
        method=ae.method,
        instance_ids=list(batch.instance_ids),
        latent_codes=ae.latent_codes,
    )
    if checkpoint and ae.latent_codes.size:
        save_latent_checkpoint(
            store_root,
            run_id,
            result.checkpoint_name,
            arrays={"latent_codes": ae.latent_codes, "q_vectors": batch.q_vectors},
            metadata={
                "method": ae.method,
                "n_instances": result.n_instances,
                "q_dim": result.q_dim,
                "target": target,
                "target_correlations": corrs,
            },
        )
    return result
