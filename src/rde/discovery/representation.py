"""Encoder/decoder (Q, r) → z_r → Ψ_r with inspectable latents (ship 6, G4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rde.discovery.checkpoint import save_latent_checkpoint
from rde.discovery.datasets import RepresentationBatch, load_representation_batch
from rde.discovery.impute import impute_matrix
from rde.discovery.latent import (
    correlate_latent_descriptors,
    ridge_predict_columns,
    ridge_predict_multi,
)
from rde.core.protocols import ResourceModel
from rde.core.resources import _DEFAULT_RESOURCE
from rde.features.numeric import safe_pearson_r


class RepresentationCandidate:
    """Input-computable representation with explicit resource model (v0.3)."""

    access_model: str = "polynomial_input"

    def fit(self, rows: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def encode_from_input(self, q: np.ndarray, r: int) -> np.ndarray:
        raise NotImplementedError

    def decode(self, code: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, code: np.ndarray, r: int) -> np.ndarray:
        raise NotImplementedError

    def resource_model(self) -> ResourceModel:
        return _DEFAULT_RESOURCE


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return X / norms


@dataclass
class RepresentationResult:
    """Level-4 coordinate system: low-dim z_r encodes (Q, r) and decodes to Ψ_r."""

    method: str
    n_samples: int
    q_dim: int
    state_dim: int
    latent_dim: int
    state_reconstruction_r2: float
    state_reconstruction_error: float
    latent_update_r2: float
    target_correlation: float
    g4_met: bool
    latent_columns: list[str] = field(default_factory=list)
    latent_descriptor_hits: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_name: str = "representation_ae"


def discover_representation_encoder_decoder(
    run_id: str,
    store_root: Path | str,
    *,
    target: str = "metric.representation_complexity",
    hidden_dim: int = 16,
    seed: int = 0,
    checkpoint: bool = True,
    rows: list[dict[str, Any]] | None = None,
    batch: RepresentationBatch | None = None,
    compute_backend: str | None = None,
) -> RepresentationResult:
    """Random-feature encoder on [Q, log r, size] → z_r; ridge decoder z_r → Ψ_r."""
    if batch is None:
        batch = load_representation_batch(run_id, store_root, target=target)
    empty = RepresentationResult(
        method="random_features_qr",
        n_samples=0,
        q_dim=0,
        state_dim=0,
        latent_dim=0,
        state_reconstruction_r2=float("nan"),
        state_reconstruction_error=float("nan"),
        latent_update_r2=float("nan"),
        target_correlation=float("nan"),
        g4_met=False,
    )
    if batch.n_samples < 4 or batch.state_dim < 1:
        return empty

    q = impute_matrix(batch.q_vectors)
    log_r = np.log1p(batch.r_indices.astype(float)).reshape(-1, 1)
    size_norm = (batch.sizes.astype(float) / max(float(batch.sizes.max()), 1.0)).reshape(-1, 1)
    enc_in = np.hstack([q, log_r, size_norm])
    backend = (compute_backend or "numpy").lower()
    if backend not in {"numpy", "mlx"}:
        raise ValueError(f"Unsupported representation backend: {compute_backend!r}")
    states = impute_matrix(batch.state_vectors)

    rng = np.random.default_rng(seed)
    k = min(hidden_dim, enc_in.shape[1], max(1, enc_in.shape[0] // 3))
    W = rng.normal(scale=0.5, size=(enc_in.shape[1], k))
    enc_input = impute_matrix(enc_in)
    if backend == "mlx":
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError("MLX representation discovery requires the mlx package") from exc

        enc_m = mx.array(enc_input, dtype=mx.float32)
        states_m = mx.array(states, dtype=mx.float32)
        state_norms_m = mx.sqrt(mx.sum(states_m * states_m, axis=1, keepdims=True))
        state_norms_m = mx.where(state_norms_m > 0, state_norms_m, mx.ones_like(state_norms_m))
        states_n_m = states_m / state_norms_m
        z_m = mx.tanh(enc_m @ mx.array(W, dtype=mx.float32))
        z_aug_m = mx.concatenate(
            [mx.ones((z_m.shape[0], 1), dtype=mx.float32), z_m],
            axis=1,
        )
        reg_m = 1e-3 * mx.eye(z_aug_m.shape[1])
        coef_m = mx.linalg.solve(
            z_aug_m.T @ z_aug_m + reg_m,
            z_aug_m.T @ states_n_m,
            # MLX's current Metal backend does not implement dense solve;
            # keep the surrounding high-dimensional matmuls device-resident.
            stream=mx.cpu,
        )
        recon_m = z_aug_m @ coef_m
        squared_error_m = mx.sum((states_n_m - recon_m) ** 2)
        mse_m = squared_error_m / mx.array(float(states_n_m.size))
        centered_states_m = states_n_m - mx.mean(states_n_m, axis=0, keepdims=True)
        ss_tot_m = mx.sum(centered_states_m * centered_states_m)
        state_r2_m = mx.array(1.0) - squared_error_m / mx.maximum(
            ss_tot_m,
            mx.array(1e-12),
        )
        mx.eval(z_m, coef_m, mse_m, state_r2_m)
        z = np.asarray(z_m, dtype=float)
        coef = np.asarray(coef_m, dtype=float)
        mse = float(np.asarray(mse_m))
        state_r2 = float(np.asarray(state_r2_m))
    else:
        states_n = _normalize_rows(states)
        z = np.tanh(enc_input @ W)
        z_aug = np.hstack([np.ones((z.shape[0], 1)), z])
        reg = 1e-3 * np.eye(z_aug.shape[1])
        coef = np.linalg.solve(z_aug.T @ z_aug + reg, z_aug.T @ states_n)
        recon = z_aug @ coef
    latent_columns = [f"latent.repr_{i}" for i in range(k)]

    # Decoder: z → Ψ_r (normalized state)
    if backend == "numpy":
        mse = float(np.mean((states_n - recon) ** 2))
        ss_tot = float(np.sum((states_n - states_n.mean(axis=0)) ** 2))
        state_r2 = float(1.0 - np.sum((states_n - recon) ** 2) / max(ss_tot, 1e-12))

    # Latent dynamics: z_r → z_{r+1} on consecutive pairs within instance
    update_r2 = float("nan")
    order = np.lexsort(
        (batch.r_indices, np.asarray(batch.instance_ids, dtype=str))
    )
    sorted_ids = np.asarray(batch.instance_ids, dtype=str)[order]
    sorted_r = batch.r_indices[order]
    consecutive = (
        (sorted_ids[:-1] == sorted_ids[1:])
        & (sorted_r[1:] == sorted_r[:-1] + 1)
    )
    pair_positions = np.flatnonzero(consecutive)
    if pair_positions.size >= k + 2:
        Zp = z[order[pair_positions]]
        Zn = z[order[pair_positions + 1]]
        _, r2_columns = ridge_predict_multi(Zp, Zn)
        if np.any(np.isfinite(r2_columns)):
            update_r2 = float(np.nanmean(r2_columns))

    target_corr = float("nan")
    if rows is not None and batch.instance_ids:
        lookup_r: dict[tuple[str, int], float] = {}
        for row in rows:
            iid = str(row.get("instance_id", ""))
            r = row.get("family_index")
            val = row.get(target)
            if iid and r is not None and val is not None and np.isfinite(float(val)):
                lookup_r[(iid, int(r))] = float(val)
        target_values = np.fromiter(
            (
                lookup_r.get((str(iid), int(r)), float("nan"))
                for iid, r in zip(batch.instance_ids, batch.r_indices)
            ),
            dtype=float,
            count=batch.n_samples,
        )
        valid_targets = np.isfinite(target_values)
        if int(valid_targets.sum()) >= k + 2:
            z_target = z[valid_targets]
            target_values = target_values[valid_targets]
            coef, _ = ridge_predict_columns(z_target, target_values)
            if np.all(np.isfinite(coef)):
                pred = np.hstack([np.ones((z_target.shape[0], 1)), z_target]) @ coef
                if np.std(pred) > 0.0 and np.std(target_values) > 0.0:
                    target_corr = safe_pearson_r(pred, target_values)

    descriptor_hits: list[dict[str, Any]] = []
    if rows is not None:
        expanded = np.full((len(rows), k), np.nan)
        row_lookup = {
            (batch.instance_ids[i], int(batch.r_indices[i])): i
            for i in range(batch.n_samples)
        }
        row_indices = np.fromiter(
            (
                row_lookup.get(
                    (str(row.get("instance_id", "")), int(row.get("family_index", -1))),
                    -1,
                )
                for row in rows
            ),
            dtype=int,
            count=len(rows),
        )
        valid_rows = row_indices >= 0
        expanded[valid_rows] = z[row_indices[valid_rows]]
        descriptor_hits = correlate_latent_descriptors(
            expanded,
            latent_columns,
            rows,
            target=target,
            min_abs_r=0.5,
        )

    poly_proxy = k <= min(32, max(4, int(np.sqrt(batch.state_dim)) + 2))
    level_4 = bool(np.isfinite(state_r2) and state_r2 >= 0.65 and poly_proxy and k >= 1)

    if checkpoint:
        save_latent_checkpoint(
            store_root,
            run_id,
            "representation_ae",
            arrays={"encoder_W": W, "decoder_coef": coef, "latent_codes": z},
            metadata={
                "method": "random_features_qr",
                "latent_dim": k,
                "state_reconstruction_r2": state_r2,
                "g4_met": level_4,
            },
        )

    return RepresentationResult(
        method="random_features_qr",
        n_samples=batch.n_samples,
        q_dim=int(q.shape[1]),
        state_dim=int(batch.state_dim),
        latent_dim=k,
        state_reconstruction_r2=state_r2,
        state_reconstruction_error=mse,
        latent_update_r2=update_r2,
        target_correlation=target_corr,
        g4_met=level_4,
        latent_columns=latent_columns,
        latent_descriptor_hits=descriptor_hits[:10],
    )
