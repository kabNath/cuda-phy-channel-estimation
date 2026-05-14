"""
GPU-accelerated MMSE channel estimation using CuPy.

The GPU implementation is API-compatible with estimators_cpu.py, but operates
on cupy.ndarray. It is intended for high-throughput receivers (e.g. NVIDIA
Aerial-style 6G PHY) where many users share the same correlation prior R_HH
and the per-symbol work is the linear solve W @ H_ls.

Two optimisations vs the textbook version:
    1. Precompute the MMSE weight matrix W = R_HH (R_HH + sigma^2 I)^-1 ONCE
       per (SNR, channel statistics) configuration. The per-symbol cost then
       drops to a single dense GEMV / GEMM.
    2. Process the batch in one GEMM call rather than per-user solves.

When R_HH or noise_variance changes, recompute W.
"""

from __future__ import annotations

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:  # graceful fallback so tests run on CPU-only CI
    cp = None
    HAS_CUPY = False


def _require_cupy() -> None:
    if not HAS_CUPY:
        raise ImportError(
            "CuPy is required for GPU estimators. Install with "
            "`pip install cupy-cuda12x` (or matching your CUDA toolkit)."
        )


def precompute_mmse_weights(R_HH, noise_variance: float):
    """
    Compute W = R_HH (R_HH + sigma^2 I)^-1 on the GPU.

    Args:
        R_HH: (N, N) Hermitian correlation matrix as cupy.ndarray or numpy.ndarray.
        noise_variance: sigma^2 at the receiver.

    Returns:
        W as a cupy.ndarray of shape (N, N), complex64 by default for speed.
    """
    _require_cupy()
    R = cp.asarray(R_HH, dtype=cp.complex64)
    N = R.shape[0]
    A = R + cp.asarray(noise_variance, dtype=cp.complex64) * cp.eye(N, dtype=cp.complex64)
    # W = R A^-1  <=>  W A = R  <=>  A^T W^T = R^T
    # Use solve to avoid explicit inverse (more numerically stable).
    W_T = cp.linalg.solve(A.T, R.T)
    W = W_T.T
    return W


def mmse_estimator_gpu(H_ls, W):
    """
    Apply precomputed MMSE weights to a batch of LS estimates on the GPU.

    Args:
        H_ls: (B, N) cupy.ndarray of LS estimates.
        W: (N, N) cupy.ndarray of MMSE weights, from precompute_mmse_weights().

    Returns:
        (B, N) cupy.ndarray of MMSE estimates.

    Notes:
        For very large batches this is one GEMM, which maps efficiently to
        Tensor Cores when complex64 is used.
    """
    _require_cupy()
    H_ls_gpu = cp.asarray(H_ls, dtype=cp.complex64)
    # (B, N) @ (N, N).T = (B, N) — equivalent to W @ H_ls.T then transpose,
    # but in this layout the batch axis is contiguous which is friendlier to GEMM.
    return H_ls_gpu @ W.T


def ls_estimator_gpu(Y, X):
    """LS estimation on GPU. Same definition as CPU version."""
    _require_cupy()
    Y_g = cp.asarray(Y, dtype=cp.complex64)
    X_g = cp.asarray(X, dtype=cp.complex64)
    return Y_g / X_g


def compute_mse_gpu(H_true, H_est) -> float:
    """MSE on GPU; result returned to host."""
    _require_cupy()
    H_t = cp.asarray(H_true)
    H_e = cp.asarray(H_est)
    return float(cp.mean(cp.abs(H_t - H_e) ** 2).get())
