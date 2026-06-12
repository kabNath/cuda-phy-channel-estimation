# Profiling the MMSE-apply kernels with Nsight Compute

The point of this repo is not that a GEMM runs on the GPU — it is to *show the
optimization journey and read it off the profiler*. This guide gives the exact
commands and the handful of metrics that tell the story.

## 0. Build with line info (already in the Makefile)

```bash
make ARCH=sm_89          # -lineinfo is on, so ncu maps stalls back to source
./mmse_apply --selftest  # confirm all three variants are correct first
```

## 1. The four metrics that matter

| Metric (ncu name)                                              | Reads as            |
|---------------------------------------------------------------|---------------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed`            | compute utilization |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed`          | memory utilization  |
| `sm__warps_active.avg.pct_of_peak_sustained_active`           | achieved occupancy  |
| `lts__t_sector_hit_rate.pct`                                  | L2 hit rate         |

A kernel is **memory-bound** when DRAM% is high and SM% is low, **compute-bound**
when the reverse holds. The naive kernel should land memory/latency-bound; tiling
should push SM% up and DRAM% down (data now served from shared memory); cuBLAS
should sit closest to the roofline.

## 2. Per-kernel collection (custom kernels have clean names)

```bash
# v0 naive — one profiled launch is enough
ncu --launch-count 1 --kernel-name apply_naive \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct \
    ./mmse_apply --bench --N 256 --B 4096

# v1 tiled — same metrics, different kernel
ncu --launch-count 1 --kernel-name apply_tiled \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct \
    ./mmse_apply --bench --N 256 --B 4096
```

cuBLAS picks an internal kernel whose name varies by arch (e.g.
`ampere_cgemm_*`). Profile it by regex instead of an exact name:

```bash
ncu --launch-count 1 --kernel-name regex:".*gemm.*" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./mmse_apply --bench --N 256 --B 4096
```

## 3. Full report + roofline (for the screenshots in the README)

```bash
ncu --set full --kernel-name apply_tiled --launch-count 1 \
    -o tiled_report ./mmse_apply --bench --N 256 --B 4096
ncu-ui tiled_report.ncu-rep          # open the GUI, see the Roofline section
```

`collect.sh` runs the per-kernel metric pass for v0 and v1 and prints a compact
table you can paste straight into the README.

## 4. What to write up

For each variant record: ms/call and GFLOP/s (from `--bench`), plus achieved
occupancy, SM% and DRAM%. Then explain the *why* in one line each:
- why naive is latency/memory-bound (W has no reuse, served from global/L2);
- why tiling helps (TILE-fold reuse from shared memory, coalesced loads);
- why cuBLAS still wins (vectorized complex MMA, register/shared-mem blocking,
  arch-tuned tile shapes) — the "know when to call the vendor library" point.
