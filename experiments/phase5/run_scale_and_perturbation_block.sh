#!/usr/bin/env bash
# Optimized-design sensitivity perturbation and million-element scale check.
#
# Step 1 measures what a fixed raised floor does to compliance and to the
# compliance gradient on optimized designs whose original floor is admissible.
# Step 2 runs the guarded policy on three million-element optimized states.
#
# Both steps need the stored optimized-density bundle under
# experiments/paper2/runs/<case>/rho_final.npy (see README section 1).
#
# Usage:
#   PYTHON=/path/to/python bash experiments/phase5/run_scale_and_perturbation_block.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"

export CUPY_ACCELERATORS="${CUPY_ACCELERATORS-}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$ROOT/.cupy_cache}"

cd "$ROOT" || exit 1

echo "=== $(date +%H:%M:%S) step 1: optimized-design sensitivity perturbation"
"$PY" experiments/phase5/analyze_optimized_density_sensitivity_perturbation.py \
  --floors 1e-12,1e-3,1e-2 \
  --out-dir experiments/phase5/results/optimized_density_sensitivity_perturbation 2>&1 | tail -30

echo "=== $(date +%H:%M:%S) step 2: million-element guarded policy"
run_scale () {
  local artifact="$1" preset="$2"
  echo "--- $(date +%H:%M:%S) $artifact ($preset)"
  "$PY" experiments/phase5/run_gmg_floor_detector_density_field.py \
    --preset "$preset" \
    --density-paths "experiments/paper2/runs/${artifact}/rho_final.npy" \
    --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
    --out-dir "experiments/phase5/results/scale_${artifact}" 2>&1 | tail -6
}

run_scale C1M_MF cantilever_gpu_xxlarge
run_scale B1M_MF bridge_gpu_1M
run_scale Brk1M_MF bracket_gpu_1M

echo "=== $(date +%H:%M:%S) scale and perturbation block done"
