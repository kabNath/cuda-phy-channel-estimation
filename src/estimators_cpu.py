"""
Reference CPU implementations of OFDM channel estimators.

Implements:
    - Least Squares (LS) — the no-prior baseline.
    - Linear MMSE — uses the analytical correlation matrix from channel.py.

The MMSE estimator we implement is the *linear* MMSE for a Gaussian prior on H,
which is the standard form used in 4G/5G receivers. The optimal weight matrix is

    W = R_HH (R_HH + sigma^2 I)^-1

and the estimate is W @ H_LS. See:
    Edfors et al., "OFDM Channel Estimation by Singular Value Decomposition",
    IEEE Trans. Commun., 1998.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def ls_estimator(
    Y: NDArray[np.complex128],
    X: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Least Squares channel estimation.

    H_LS = Y / X.

    Args:
        Y: Received frequency-domain samples, shape (batch_size, num_subcarriers).
        X: Transmitted pilot symbols, same shape (must be non-zero on pilot bins).

    Returns:
        LS estimate, same shape.

    Notes:
        Assumes pilots transmitted on EVERY subcarrier (block-type pilots).
        For comb-type pilot patterns, interpolation between pilot bins is needed
        before applying MMSE. We keep this implementation minimal on purpose.
    """
    return Y / X


def mmse_estimator(
    H_ls: NDArray[np.complex128],
    R_HH: NDArray[np.complex128],
    noise_variance: float,
) -> NDArray[np.complex128]:
    """
    Linear MMSE channel estimation.

    W = R_HH (R_HH + sigma^2 I)^-1
    H_MMSE = W @ H_LS

    Args:
        H_ls: LS estimates, shape (batch_size, num_subcarriers).
        R_HH: Frequency-domain channel correlation matrix, shape (N, N).
        noise_variance: sigma^2 at the receiver, in linear scale.

    Returns:
        MMSE estimates, shape (batch_size, num_subcarriers).
    """
    N = R_HH.shape[0]
    # (R + sigma^2 I) is positive-definite Hermitian — use solve, not explicit inverse.
    # W H_ls = R_HH (R_HH + sigma^2 I)^-1 H_ls
    # Equivalent: solve (R_HH + sigma^2 I) X = H_ls, then H_mmse = R_HH @ X.
    A = R_HH + noise_variance * np.eye(N, dtype=R_HH.dtype)
    # H_ls is (B, N); we need to solve per-batch. np.linalg.solve broadcasts on the
    # last two dims, so reshape H_ls to (B, N, 1).
    X = np.linalg.solve(A, H_ls.T).T  # (B, N)
    # NOTE: solve(A, b) where A is (N, N) and b is (N, B) returns (N, B); transpose.
    H_mmse = X @ R_HH.T  # (B, N) since R_HH is Hermitian: R_HH.T = R_HH.conj()
    # Equivalent more readable form:
    # H_mmse = (R_HH @ np.linalg.solve(A, H_ls.T)).T
    return H_mmse


def compute_mse(
    H_true: NDArray[np.complex128],
    H_est: NDArray[np.complex128],
) -> float:
    """
    Mean squared error per subcarrier, averaged over the batch.

    MSE = E[|H_true - H_est|^2]
    """
    return float(np.mean(np.abs(H_true - H_est) ** 2))


def add_awgn(
    signal: NDArray[np.complex128],
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex128], float]:
    """
    Add complex AWGN at the specified SNR (in dB).

    Args:
        signal: Noise-free signal.
        snr_db: Target SNR in dB, defined relative to E[|signal|^2].
        rng: Random generator.

    Returns:
        (noisy_signal, noise_variance_per_dimension).
    """
    if rng is None:
        rng = np.random.default_rng()
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = signal_power / snr_linear
    noise = (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)) * np.sqrt(noise_var / 2.0)
    return signal + noise, float(noise_var)
