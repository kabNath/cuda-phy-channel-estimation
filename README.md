# cuda-phy-channel-estimation

**A CUDA kernel-optimization case study on the inner loop of MMSE channel estimation.**

[![cuda](https://github.com/kabNath/cuda-phy-channel-estimation/actions/workflows/cuda.yml/badge.svg)](https://github.com/kabNath/cuda-phy-channel-estimation/actions/workflows/cuda.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Channel estimation is the most-frequently-executed signal-processing step in a
5G/6G receiver — every coherence interval, for every user and antenna. Once the
Wiener matrix `W = R_HH (R_HH + σ²I)⁻¹` is precomputed, the hot path is simply
applying it to a fresh batch of LS estimates:

```
H_mmse = W · H_ls,   W ∈ C^{N×N},   H_ls ∈ C^{N×B}
```

a **batched complex GEMM**. This repo takes that one operation and walks the
canonical GPU optimization path — **naive → shared-memory tiled → cuBLAS** —
on an RTX 4090, then explains the result with the memory hierarchy. The goal is
to show kernel-engineering and profiling judgment, not to claim a novel kernel.

> Part of a three-repo set. The end-to-end AI-RAN PHY system (real CUDA MMSE +
> Sionna 5G BLER link + PPO/OLLA link adaptation) lives in
> **[gpu-accelerated-ai-ran-phy-lab](https://github.com/kabNath/gpu-accelerated-ai-ran-phy-lab)**.
> This repo is the *GPU-kernel deep-dive* that sits underneath it.

---

## The optimization journey (RTX 4090, sm_89, CUDA 12.8)

`./mmse_apply --bench --N 256 --B 4096 --iters 50`, complex64:

| variant | ms/call | GFLOP/s | speedup vs naive |
|---|---:|---:|---:|
| v0 naive (1 thread/output) | 0.122 | 17 623 | 1.00× |
| v1 tiled (shared memory)   | 0.107 | 20 168 | 1.14× |
| v2 cuBLAS cgemm            | 0.049 | 44 132 | **2.50×** |

Two things stand out — and both *are* the point of the study:

**1. The naive kernel already hits ~17.6 TFLOP/s (~21 % of the 4090's FP32 peak),
and shared-memory tiling only improves it 1.14×.** That is not a failed
optimization — it is a property of the hardware. At N=256 the Wiener matrix `W`
is 256²·8 B = **512 KB** and the `H_ls`/`H_mmse` batches are 8 MB each, so the
whole working set fits inside the 4090's **72 MB L2 cache**. The naive kernel's
repeated reads of `W` and `H_ls` are therefore served from L2, not DRAM — so it
is *not* the bandwidth-bound kernel the textbook assumes, and tiling (whose job
is to cut DRAM traffic) has little to cut. On large-cache Ada/Hopper GPUs the
classic naive→tiled win shrinks for cache-resident problems.

**2. cuBLAS still wins 2.5× — but through compute scheduling, not bandwidth:**
vectorized complex MMA, register/shared-memory blocking, and arch-tuned tile
shapes that keep the SMs busier and hide latency better than a straightforward
tiled kernel.

The takeaway is engineering judgment: check working-set vs cache size *before*
reaching for a textbook optimization, and know when the right move is to call
the vendor library.

## Quick start

```bash
make ARCH=sm_89            # sm_75 for a T4, sm_80 A100, sm_90 H100
./mmse_apply --selftest    # correctness: all variants < 1e-3 vs CPU reference
./mmse_apply --bench --N 256 --B 4096 --iters 50
```

`--selftest` checks all three variants against a double-precision CPU reference;
on the 4090 they agree to ~1.7e-7.

## Profiling (reproducible)

The analysis above is a claim about the memory hierarchy, so here is how to
verify it with **Nsight Compute** rather than take it on faith:

```bash
ncu --launch-count 1 --kernel-name apply_naive \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct \
    ./csrc/mmse_apply --bench --N 256 --B 4096
# repeat with --kernel-name apply_tiled ; cuBLAS via --kernel-name regex:".*gemm.*"
```

The prediction to check: the naive kernel shows a **high L2 hit rate** with only
moderate DRAM throughput — the quantitative confirmation that it is L2-served,
not DRAM-bound. `profiling/collect.sh` runs this for the two custom kernels and
`profiling/nsight.md` documents each metric and the roofline view.

> On WSL2, GPU performance counters are gated by the Windows driver. Enable them
> via NVIDIA Control Panel → *Developer* → *Manage GPU Performance Counters* →
> "Allow access to all users", then `wsl --shutdown` and relaunch.

## Why the apply step is the right thing to profile

For a receiver tracking ~100 users at SNRs binned in 1-dB intervals, the Wiener
matrices are precomputed and cached — so the *recurring* cost is the apply, not
the solve. The matrix build (`R_HH`) and the Cholesky solve live in the Python
reference and in the flagship repo (see **Scope**).

## Estimator correctness (the statistics)

The CUDA case study is validated numerically (`--selftest`). That the MMSE
estimator achieves the theoretical `10·log₁₀(N/L)` gain over LS is shown by the
NumPy / CuPy reference in `src/`:

| SNR (dB) | MSE LS | MSE MMSE | gain |
|---:|---:|---:|---:|
| 0  | 9.92e-1 | 1.02e-1 | 9.88 dB |
| 10 | 1.00e-1 | 1.22e-2 | 9.14 dB |
| 20 | 9.96e-3 | 1.24e-3 | 9.04 dB |
| 30 | 1.01e-3 | 1.27e-4 | 9.00 dB |

For N=64, L=8 the bound is `10·log₁₀(8) = 9.03 dB` — matched across SNR.

```bash
pip install -r requirements.txt
python -m benchmarks.mse_vs_snr
```

## Mathematical model

One OFDM symbol, `N` subcarriers, frequency-selective Rayleigh channel, `L` taps.
Frequency domain: `Y[k] = H[k]·X[k] + N[k]`, `N[k] ~ CN(0, σ²)`.

- **LS:** `H_LS = Y / X` — unbiased, `MSE = σ²`.
- **MMSE:** `W = R_HH (R_HH + σ²I)⁻¹`, `H_MMSE = W · H_LS`, with
  `R_HH[k₁,k₂] = Σ_l p_l · exp(−j2π(k₁−k₂)l/N)`. Because `R_HH` has rank `L < N`,
  `W` is a soft projection onto the `L`-dimensional signal subspace — which is
  why MMSE rejects noise that LS cannot.

## Repository layout

```
csrc/
  mmse_apply.cu        # v0 naive, v1 tiled, v2 cuBLAS — the case study
  Makefile             # build + `compile-check` (no-GPU, used by CI)
profiling/
  nsight.md            # exact ncu commands + which metrics to capture
  collect.sh           # one-shot metric table for v0/v1
src/
  channel.py           # multipath channel, R_HH
  estimators_cpu.py    # LS, MMSE (NumPy reference)
  estimators_gpu.py    # LS, MMSE (CuPy reference)
benchmarks/            # MSE-vs-SNR, CuPy throughput
tests/                 # estimator unit tests
.github/workflows/cuda.yml   # CI: compile-check the CUDA on every push
```

## Scope (honest limits)

This profiles the **apply** kernel of a single-symbol, all-pilot, single-antenna
estimator. Not covered here (tracked in the flagship / roadmap): the `R_HH`
build and a batched Cholesky solve as CUDA kernels, comb-type pilots with 2-D
Wiener interpolation, MIMO, and doubly-selective (Kalman) tracking.

## References

- O. Edfors et al., "OFDM Channel Estimation by Singular Value Decomposition,"
  *IEEE Trans. Commun.*, 46(7), 1998.
- NVIDIA, *Aerial cuPHY* documentation.
- J. Hoydis et al., "Sionna," arXiv:2203.11854, 2022.

## License

MIT.

