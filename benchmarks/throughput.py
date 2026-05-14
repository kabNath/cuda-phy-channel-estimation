"""
Benchmark: per-symbol throughput of MMSE channel estimation, CPU vs GPU.

This script measures how many OFDM symbols per second can be estimated by:
    - NumPy on CPU (baseline)
    - CuPy on GPU (if available)

The MMSE weight matrix W is precomputed once per (R_HH, sigma^2) configuration.
Steady-state per-symbol work is therefore one GEMM: (B, N) @ (N, N).

Run from project root:
    python -m benchmarks.throughput

Outputs:
    Throughput table to stdout.
    docs/throughput.png comparing CPU and GPU across batch sizes (if both available).
"""

from __future__ import annotations
import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.channel import frequency_correlation_matrix


def _build_W_cpu(num_subcarriers: int, num_paths: int, sigma2: float) -> np.ndarray:
    R = frequency_correlation_matrix(num_subcarriers, num_paths).astype(np.complex64)
    A = R + sigma2 * np.eye(num_subcarriers, dtype=np.complex64)
    return R @ np.linalg.inv(A)


def bench_cpu(W: np.ndarray, batch_size: int, num_iters: int = 50) -> float:
    """Return symbols/second."""
    N = W.shape[0]
    H_ls = (np.random.randn(batch_size, N) + 1j * np.random.randn(batch_size, N)).astype(np.complex64)
    # Warmup
    for _ in range(5):
        _ = H_ls @ W.T
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = H_ls @ W.T
    t1 = time.perf_counter()
    total_symbols = batch_size * num_iters
    return total_symbols / (t1 - t0)


def bench_gpu(W_cpu: np.ndarray, batch_size: int, num_iters: int = 50) -> float | None:
    """Return symbols/second, or None if CuPy is unavailable."""
    try:
        import cupy as cp
    except ImportError:
        return None
    N = W_cpu.shape[0]
    W = cp.asarray(W_cpu)
    H_ls = (cp.random.randn(batch_size, N) + 1j * cp.random.randn(batch_size, N)).astype(cp.complex64)
    # Warmup
    for _ in range(5):
        _ = H_ls @ W.T
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = H_ls @ W.T
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    total_symbols = batch_size * num_iters
    return total_symbols / (t1 - t0)


def main():
    num_subcarriers = 256
    num_paths = 16
    sigma2 = 0.01
    batch_sizes = [16, 64, 256, 1024, 4096, 16384]

    W = _build_W_cpu(num_subcarriers, num_paths, sigma2)

    print(f"\nMMSE channel estimation throughput")
    print(f"N={num_subcarriers}, L={num_paths}, complex64")
    print(f"{'Batch':>8} {'CPU sym/s':>15} {'GPU sym/s':>15} {'Speedup':>10}")
    print("-" * 55)

    cpu_results, gpu_results = [], []
    for B in batch_sizes:
        cpu = bench_cpu(W, B)
        gpu = bench_gpu(W, B)
        cpu_results.append(cpu)
        gpu_results.append(gpu)
        if gpu is not None:
            print(f"{B:>8d} {cpu:>15.2e} {gpu:>15.2e} {gpu/cpu:>9.1f}x")
        else:
            print(f"{B:>8d} {cpu:>15.2e} {'(no CuPy)':>15} {'-':>10}")

    if all(g is not None for g in gpu_results):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(batch_sizes, cpu_results, "o-", label="CPU (NumPy)", linewidth=2)
        ax.loglog(batch_sizes, gpu_results, "s-", label="GPU (CuPy)", linewidth=2)
        ax.set_xlabel("Batch size (OFDM symbols)")
        ax.set_ylabel("Throughput (symbols / second)")
        ax.set_title(f"MMSE throughput, N={num_subcarriers}, complex64")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        os.makedirs("docs", exist_ok=True)
        fig.savefig("docs/throughput.png", dpi=150)
        print(f"\nFigure saved to docs/throughput.png")
    else:
        print("\nGPU benchmark skipped (CuPy not installed). Install with:")
        print("    pip install cupy-cuda12x  # or matching your CUDA version")


if __name__ == "__main__":
    main()
