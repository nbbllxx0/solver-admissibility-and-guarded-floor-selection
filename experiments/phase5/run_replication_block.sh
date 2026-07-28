#!/usr/bin/env bash
# Second-platform replication and precision ablation.
#
# Runs the six-state mechanism matrix (cantilever and bridge, seed 43,
# q = 0.10/0.20/0.35) under
#   (i)  the reported FP64 fine path, and
#   (ii) an FP32 fine smoother,
# so that the effect of fine-level precision on the keep/raise decision and on
# the acceptance guard can be read off directly. Running block (i) on a second
# GPU/CuPy/CUDA combination also replicates the reported mechanism matrix.
#
# Usage:
#   PYTHON=/path/to/python bash experiments/phase5/run_replication_block.sh [OUT_ROOT]
#
# Defaults: PYTHON=python, OUT_ROOT=experiments/phase5/results/replication
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"
OUT_ROOT="${1:-experiments/phase5/results/replication}"

# CuPy 14 on CUDA 13: the CUB-backed reductions fail to compile against the
# bundled CCCL headers. Harmless to set on other versions.
export CUPY_ACCELERATORS="${CUPY_ACCELERATORS-}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$ROOT/.cupy_cache}"

cd "$ROOT" || exit 1

run_case () {
  local variant="$1" preset="$2" short="$3" extra="$4"
  local out="${OUT_ROOT}/${variant}/${short}_s43"
  mkdir -p "$out"
  echo "=== $(date +%H:%M:%S) $variant / $short"
  # shellcheck disable=SC2086
  "$PY" experiments/phase5/run_gmg_floor_detector_prospective.py \
    --preset "$preset" \
    --seeds 43 \
    --probabilities 0.10,0.20,0.35 \
    --baseline-rho-min 1e-12 \
    --penal 4.5 \
    --stack-variant "$variant" \
    --raised-rho-mins 1e-3,1e-2 \
    --high-residual-threshold 1e-2 \
    --plateau-residual-threshold 1e-4 \
    --out-dir "$out" \
    $extra 2>&1 | tail -8
}

for spec in "canonical|" "fp32_fine|--fine-smoother fp32"; do
  variant="${spec%%|*}"
  extra="${spec#*|}"
  run_case "$variant" cantilever_gpu_medium cantilever "$extra"
  run_case "$variant" bridge_gpu_medium bridge "$extra"
done

echo "=== $(date +%H:%M:%S) replication block done -> $OUT_ROOT"
