#!/usr/bin/env bash
# Collect the four key Nsight Compute metrics for the two custom kernels and
# print a compact table. cuBLAS is profiled separately (see profiling/nsight.md).
#   usage: ./profiling/collect.sh [N] [B]
set -euo pipefail
N="${1:-256}"
B="${2:-4096}"
BIN="./csrc/mmse_apply"
M="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu (Nsight Compute) not found in PATH. It ships with the CUDA toolkit;"
  echo "on WSL2 it is usually /usr/local/cuda/bin/ncu — add that to PATH."
  exit 1
fi
[ -x "$BIN" ] || { echo "build first: make ARCH=sm_89"; exit 1; }

for K in apply_naive apply_tiled; do
  echo "===== $K (N=$N B=$B) ====="
  ncu --launch-count 1 --kernel-name "$K" --metrics "$M" \
      "$BIN" --bench --N "$N" --B "$B" 2>/dev/null \
    | grep -E "sm__throughput|dram__throughput|sm__warps_active|lts__t_sector_hit_rate" || true
  echo
done
echo "For cuBLAS:  ncu --launch-count 1 --kernel-name regex:'.*gemm.*' --metrics $M $BIN --bench --N $N --B $B"
