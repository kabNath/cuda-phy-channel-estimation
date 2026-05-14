"""
Benchmark: MSE of LS and MMSE estimators vs SNR.

Generates the canonical "MSE floor" plot showing the MMSE gain over LS.

Run from project root:
    python -m benchmarks.mse_vs_snr

Outputs:
    docs/mse_vs_snr.png
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.channel import (
    generate_time_domain_channel,
    time_to_frequency,
    frequency_correlation_matrix,
)
from src.estimators_cpu import ls_estimator, mmse_estimator, compute_mse, add_awgn


def run_benchmark(
    num_subcarriers: int = 64,
    num_paths: int = 8,
    batch_size: int = 2000,
    snr_range_db: tuple[float, float, int] = (0.0, 30.0, 16),
    seed: int = 0,
) -> dict:
    """
    Sweep SNR and measure MSE for LS and MMSE.

    Returns a dict with keys: snr_db, mse_ls, mse_mmse, theoretical_ls_floor.
    """
    rng = np.random.default_rng(seed)
    snrs = np.linspace(*snr_range_db)
    mse_ls = np.zeros_like(snrs)
    mse_mmse = np.zeros_like(snrs)

    # Channel statistics (computed once).
    R_HH = frequency_correlation_matrix(num_subcarriers, num_paths)

    for i, snr_db in enumerate(snrs):
        h_time = generate_time_domain_channel(batch_size, num_paths, rng=rng)
        H_true = time_to_frequency(h_time, num_subcarriers)
        X = np.ones_like(H_true)
        Y_clean = H_true * X
        Y, sigma2 = add_awgn(Y_clean, snr_db, rng=rng)

        H_ls = ls_estimator(Y, X)
        H_mmse = mmse_estimator(H_ls, R_HH, sigma2)

        mse_ls[i] = compute_mse(H_true, H_ls)
        mse_mmse[i] = compute_mse(H_true, H_mmse)

    # Theoretical LS MSE = sigma^2 (since channel power is normalized to 1).
    signal_power = 1.0
    snr_linear = 10 ** (snrs / 10.0)
    theoretical_ls = signal_power / snr_linear

    return {
        "snr_db": snrs,
        "mse_ls": mse_ls,
        "mse_mmse": mse_mmse,
        "theoretical_ls": theoretical_ls,
        "num_subcarriers": num_subcarriers,
        "num_paths": num_paths,
        "batch_size": batch_size,
    }


def plot_results(results: dict, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(results["snr_db"], results["mse_ls"], "o-", label="LS (empirical)", linewidth=2)
    ax.semilogy(results["snr_db"], results["theoretical_ls"], "k--", label=r"LS theory: $\sigma^2$", alpha=0.6)
    ax.semilogy(results["snr_db"], results["mse_mmse"], "s-", label="MMSE", linewidth=2, color="C2")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE")
    ax.set_title(
        f"OFDM channel estimation: LS vs MMSE\n"
        f"N={results['num_subcarriers']} subcarriers, "
        f"L={results['num_paths']} paths, "
        f"batch={results['batch_size']}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to {output_path}")


def main():
    results = run_benchmark()
    plot_results(results, "docs/mse_vs_snr.png")
    print("\nResults at key SNR points:")
    print(f"{'SNR (dB)':>10} {'MSE LS':>12} {'MSE MMSE':>12} {'Gain (dB)':>12}")
    for i in [0, 5, 10, 15]:
        snr = results["snr_db"][i]
        ls = results["mse_ls"][i]
        mmse = results["mse_mmse"][i]
        gain_db = 10 * np.log10(ls / mmse)
        print(f"{snr:>10.1f} {ls:>12.4e} {mmse:>12.4e} {gain_db:>12.2f}")


if __name__ == "__main__":
    main()
