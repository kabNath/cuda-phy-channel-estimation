"""
Unit tests for estimators.

Run: pytest tests/
"""

import numpy as np
import pytest

from src.channel import (
    generate_multipath_pdp,
    generate_time_domain_channel,
    time_to_frequency,
    frequency_correlation_matrix,
)
from src.estimators_cpu import (
    ls_estimator,
    mmse_estimator,
    compute_mse,
    add_awgn,
)


@pytest.fixture
def small_system():
    rng = np.random.default_rng(42)
    N = 64
    L = 8
    batch = 256
    snr_db = 20.0
    h_time = generate_time_domain_channel(batch, L, rng=rng)
    H_true = time_to_frequency(h_time, N)
    X = np.ones((batch, N), dtype=np.complex128)  # all-ones pilots
    Y_clean = H_true * X
    Y, sigma2 = add_awgn(Y_clean, snr_db, rng=rng)
    R_HH = frequency_correlation_matrix(N, L)
    return {"H_true": H_true, "Y": Y, "X": X, "R_HH": R_HH, "sigma2": sigma2}


def test_pdp_normalized():
    pdp = generate_multipath_pdp(8)
    assert np.isclose(pdp.sum(), 1.0)


def test_correlation_matrix_hermitian():
    R = frequency_correlation_matrix(64, 8)
    assert np.allclose(R, R.conj().T)


def test_correlation_diagonal_unit():
    # diag(R_HH) should be 1 (total channel power on each subcarrier = sum(pdp) = 1)
    R = frequency_correlation_matrix(32, 4)
    assert np.allclose(np.diag(R).real, 1.0)
    assert np.allclose(np.diag(R).imag, 0.0, atol=1e-12)


def test_ls_unbiased_at_high_snr(small_system):
    H_ls = ls_estimator(small_system["Y"], small_system["X"])
    mse = compute_mse(small_system["H_true"], H_ls)
    # At 20 dB SNR, LS MSE ~ sigma^2 = 0.01.
    assert mse < 0.02


def test_mmse_beats_ls(small_system):
    H_ls = ls_estimator(small_system["Y"], small_system["X"])
    H_mmse = mmse_estimator(H_ls, small_system["R_HH"], small_system["sigma2"])
    mse_ls = compute_mse(small_system["H_true"], H_ls)
    mse_mmse = compute_mse(small_system["H_true"], H_mmse)
    # MMSE must be strictly better than LS in any reasonable SNR regime.
    assert mse_mmse < mse_ls


def test_mmse_approaches_truth_at_low_noise(small_system):
    # At very low noise, MMSE projects LS onto the L-dimensional signal subspace
    # (R_HH has rank L < N), which is exactly where H_true lives. So H_mmse -> H_true.
    H_ls = ls_estimator(small_system["Y"], small_system["X"])
    H_mmse = mmse_estimator(H_ls, small_system["R_HH"], noise_variance=1e-6)
    mse = compute_mse(small_system["H_true"], H_mmse)
    assert mse < 5e-3, f"Expected MSE near zero, got {mse}"


def test_mmse_weights_idempotent_on_signal_subspace(small_system):
    # The MMSE weight matrix W in the zero-noise limit is the projector
    # onto the column space of R_HH. Projectors are idempotent: W @ W = W.
    R = small_system["R_HH"]
    N = R.shape[0]
    sigma2 = 1e-10
    W = R @ np.linalg.solve(R + sigma2 * np.eye(N), np.eye(N))
    assert np.allclose(W @ W, W, atol=1e-4)
