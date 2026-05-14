"""
OFDM multipath channel generation.

Reference:
    Goldsmith, "Wireless Communications", Cambridge, 2005, Ch. 12.
    Heath & Lozano, "Foundations of MIMO Communication", Cambridge, 2018.

This module generates realizations of a frequency-selective Rayleigh fading
channel in both time and frequency domain, suitable for OFDM simulation.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def generate_multipath_pdp(num_paths: int, rms_delay_spread: float = 3.0) -> NDArray[np.float64]:
    """
    Generate an exponential power delay profile (PDP).

    Args:
        num_paths: Number of channel taps L.
        rms_delay_spread: RMS delay spread in sample units.

    Returns:
        Power vector of shape (num_paths,), normalized so that sum equals 1.

    Notes:
        Exponential PDP: P(l) ∝ exp(-l / tau_rms), l = 0, ..., L-1.
        Typical urban / dense urban channels have rms_delay_spread of 2-5 samples
        for sub-6 GHz OFDM with cyclic prefix designed around it.
    """
    taps = np.arange(num_paths)
    pdp = np.exp(-taps / rms_delay_spread)
    return pdp / pdp.sum()


def generate_time_domain_channel(
    batch_size: int,
    num_paths: int,
    rms_delay_spread: float = 3.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """
    Generate complex-Gaussian (Rayleigh) channel taps in time domain.

    Args:
        batch_size: Number of independent channel realizations.
        num_paths: Number of channel taps L.
        rms_delay_spread: RMS delay spread for the exponential PDP.
        rng: NumPy random generator (defaults to default_rng()).

    Returns:
        Tensor of shape (batch_size, num_paths), complex128.
        Each tap h[b, l] ~ CN(0, P(l)) where P is the PDP.
    """
    if rng is None:
        rng = np.random.default_rng()
    pdp = generate_multipath_pdp(num_paths, rms_delay_spread)
    # Complex Gaussian with variance P(l) per tap
    real = rng.standard_normal((batch_size, num_paths))
    imag = rng.standard_normal((batch_size, num_paths))
    h = (real + 1j * imag) * np.sqrt(pdp / 2.0)
    return h


def time_to_frequency(
    h_time: NDArray[np.complex128],
    num_subcarriers: int,
) -> NDArray[np.complex128]:
    """
    Compute the frequency-domain channel via N-point FFT of zero-padded taps.

    Args:
        h_time: Time-domain taps of shape (batch_size, num_paths).
        num_subcarriers: FFT size N (must be >= num_paths).

    Returns:
        Frequency-domain channel of shape (batch_size, num_subcarriers).
    """
    batch_size, num_paths = h_time.shape
    if num_paths > num_subcarriers:
        raise ValueError(
            f"num_paths={num_paths} cannot exceed num_subcarriers={num_subcarriers}"
        )
    padded = np.zeros((batch_size, num_subcarriers), dtype=np.complex128)
    padded[:, :num_paths] = h_time
    return np.fft.fft(padded, axis=-1)


def frequency_correlation_matrix(
    num_subcarriers: int,
    num_paths: int,
    rms_delay_spread: float = 3.0,
) -> NDArray[np.complex128]:
    """
    Closed-form frequency-domain correlation matrix R_HH for the PDP above.

    R_HH[k1, k2] = sum_l P(l) * exp(-j 2 pi (k1 - k2) l / N)

    Used by the MMSE estimator. Computing this analytically (rather than from
    Monte Carlo) gives the estimator a fair, deterministic prior.

    Returns:
        Hermitian matrix of shape (num_subcarriers, num_subcarriers).
    """
    pdp = generate_multipath_pdp(num_paths, rms_delay_spread)
    k = np.arange(num_subcarriers)
    delta = k[:, None] - k[None, :]  # (N, N)
    taps = np.arange(num_paths)
    # R[k1, k2] = sum_l p_l exp(-j 2pi delta l / N)
    phase = -2j * np.pi * delta[:, :, None] * taps[None, None, :] / num_subcarriers
    R = np.einsum("l,ijl->ij", pdp.astype(np.complex128), np.exp(phase))
    return R
