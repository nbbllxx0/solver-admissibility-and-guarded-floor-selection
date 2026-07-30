# Solver admissibility and guarded floor selection

Code release for:

> Yang, S., Wang, J., and Wang, Y. (2026).
> *When a positive SIMP density floor is not enough: solver admissibility and guarded
> floor selection in matrix-free 3D topology optimization.*

**The finding in one paragraph.** In ersatz-material SIMP, void elements are kept weakly
stiff by a positive density floor. A positive floor is usually treated as sufficient to
make the constrained stiffness operator solvable. It is not: for a given matrix-free
geometric-multigrid FGMRES hierarchy, tolerance, and iteration budget, a mathematically
positive floor can be *solver-inadmissible*. Worse, the failure is not always visible — on
4 of 102 held-out states the outer Krylov iteration set its convergence flag while the
recomputed true residual was up to 49.5× the requested tolerance, and inside an
optimization loop the same failure produced a compliance history oscillating between 0.48
and 4.50 with no error raised. This repository implements the solver stack, the guarded
floor-selection policy that fixes it (probe → preserve/escalate rule → floor ladder →
**recomputed-residual acceptance guard**), and every experiment reported in the paper.

---

## 1. What this repository is, and is not

**Is:** the complete source for the matrix-free finite-element stack, the multigrid
hierarchy, the floor policy, all experiment drivers, all analyzers, the summary builder,
and the figure generators. Every number in the paper can be regenerated from this code
plus a GPU.

**Is not:** a results archive. Result CSVs, logs, figures, and the large stored density
arrays are not tracked here — fresh runs recreate them under
`experiments/phase5/results/`. Two things therefore cannot be reproduced from this
repository alone:

| Needs | Why | Where to get it |
| --- | --- | --- |
| `experiments/paper2/runs/<case>/rho_final.npy` | The optimized-density transfer study and its figure use eight stored final SIMP density fields | Result/artifact bundle deposited with the paper |
| Reported wall-time tables | Timings are hardware-specific single observed runs | Rerun locally; expect different absolute values |

Everything else — direct atlas, retrospective labels, the 102-state held-out suite, fixed
floor controls, sensitivity perturbation, threshold and policy sweeps, mechanism
ablations, and the optimization trajectories — is fully reproducible here.

---

## 2. Quick start

```bash
conda env create -f environment.yml
conda activate paper5-solver-admissibility
python ci/smoke_check.py                     # source-level check, no GPU needed
```

First GPU check (~1 minute, 64k elements, no stored inputs required):

```bash
python experiments/phase5/run_gmg_floor_detector_prospective.py \
  --preset cantilever_gpu_medium --seeds 43 --probabilities 0.35 \
  --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
  --high-residual-threshold 1e-2 --plateau-residual-threshold 1e-4 \
  --out-dir experiments/phase5/results/quickcheck
```

Expected: one row with `trigger=keep`, `recommended_rho_min=1e-12`,
`solve_converged=1`, and 30 FGMRES iterations. The probe feature `r50` should be about
`5.3e-07`. If instead you see `trigger=high_r50`, the environment is fine but the
hierarchy is behaving differently from the reported stack — check §3.

---

## 3. Verified platforms

The stack has been run end to end on two platforms. Convergence decisions, selected
floors, and iteration counts agree between them; wall times do not, and only the first
platform's timings appear in the paper.

| | Platform A (paper) | Platform B (replication) |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24 GiB (SM 8.9) | NVIDIA GeForce RTX 5090, 32 GiB (SM 12.0) |
| CuPy | 13.6.0 (`cupy-cuda12x`) | 14.0.1 (`cupy-cuda13x`) |
| CUDA runtime | 12.x | 13.0 |
| Python | 3.10.18 | 3.11 |
| Covers | all reported results | mechanism matrix, precision ablation, optimized-density perturbation, 1M-element case |

**Platform B notes.** Two adjustments are needed on CuPy 14 / CUDA 13; the second is
already handled in the code:

1. CuPy 14's CUB-backed reductions fail to compile against the CUDA 13 CCCL headers
   (`ambiguous "?" operation … __nv_bfloat16`). Disable the accelerator before running:
   ```bash
   export CUPY_ACCELERATORS=""      # PowerShell: $env:CUPY_ACCELERATORS = ""
   ```
2. The fused matvec kernels compile with `-std=c++17` on CuPy ≥ 14 and `-std=c++14` on
   CuPy 13; `src/gpu_fem/cuda_fused_matvec.py` selects this from the installed version.

A `cupy-cuda12x` wheel will not run on Blackwell (SM 12.0) — it raises
`CUDA_ERROR_NO_BINARY_FOR_GPU`. Install `cupy-cuda13x` for those GPUs.

**Memory.** Flexible GMRES stores two Krylov bases (V and Z). With the reported
`--restart 300` in FP64 that is \(2\times300\times n_{\mathrm{free}}\times8\) bytes, which
dominates the footprint at large sizes:

| Elements | free DOFs | basis at restart 300 | basis at restart 100 |
| --- | --- | --- | --- |
| 512,000 | 1.60 M | 7.7 GiB | 2.6 GiB |
| 514,500 | 1.62 M | 7.8 GiB | 2.6 GiB |
| 1,000,000 | 3.11 M | 14.9 GiB | 5.0 GiB |

Peak post-solve device memory observed: ≈3.8 GiB for the 40.5k–64k held-out policy matrix
and ≈18–23 GiB for the ~500k optimized-density cases. A 24 GiB card therefore covers every
case reported in the paper; at 10⁶ elements the FP64 configuration needs a restarted method
(`experiments/phase5/run_scale_1m_restarted.sh`, restart 100) or the reduced-precision
hierarchy of the companion solver work. Since admissibility is defined relative to the
iteration budget, note that shrinking the budget to fit memory can turn a keep into a raise.

---

## 4. Repository layout

```
src/gpu_fem/                   solver implementation
  presets.py                   problem specifications (geometry, BCs, loads, sizes)
  bc_generator.py              boundary-condition and load assembly
  cuda_fused_matvec.py         fused gather-GEMM-scatter fine-level operator (FP32/BF16)
  multigrid_v4.py              Galerkin GMG hierarchy, smoothers, coarse correction,
                               and `_cupy_fgmres` (the outer Krylov iteration)
  pub_simp_solver.py           element stiffness, edof tables, sparse index helpers
  solver_v4.py, simp_gpu.py    SIMP optimization loop used by the trajectory runs
  local_agents.py, workflow.py optimization routing used by run_simp_floor_trajectory.py

experiments/paper4/
  run_experiments_e1_e10.py    `_build_components()` — builds spec, BCs, operator, and
                               hierarchy for a preset. Every phase-5 GPU driver imports it.

experiments/phase5/            the study itself (drivers, analyzers, figures)

data/analysis_tables/          the analysis tables every figure and table in the
                               paper is computed from, plus MANIFEST.csv (size,
                               SHA-256, and the paper element each one supports)

data/optimized_density_inputs/ the nine stored density fields the transfer study,
                               SI gallery, and 1M scale check take as inputs,
                               with their own checksummed MANIFEST.csv

ci/smoke_check.py              syntax check over all released Python files
environment.yml                pinned environment (Platform A)
environment-blackwell.yml      pinned environment (Platform B: CuPy 14 / CUDA 13)
```

### Where the method lives

| Component of the policy | Implementation |
| --- | --- |
| Probe: 100 FGMRES iterations at `rho_0`, records `r50`, `r100` | `run_gmg_floor_detector_prospective.py`, `run_gmg_floor_detector_density_field.py` |
| Preserve/escalate rule (`keep`/`raise` in the CSV schema): `r50 ≥ 1e-2`, or `r100 ≥ 1e-4` and `r100/r50 ≥ 0.6` | same drivers: `--high-residual-threshold`, `--plateau-residual-threshold`, `--plateau-ratio-threshold` |
| Floor ladder `1e-3 → 1e-2` | `--raised-rho-mins` |
| **Acceptance guard**: recomputed `‖f − Ku‖ / ‖f‖ ≤ tol` | `solve_at_floor()` in both drivers — `final = ‖F − A(x)‖ / ‖F‖` is formed *after* the solve returns and is the only criterion that sets `solve_converged` |
| In-loop version of the same policy | `run_simp_floor_trajectory.py --policy guarded_adaptive` |

The distinction the paper turns on is visible in the output columns:
`solver_reported_converged` is the solver's internal flag, `solve_converged` is the
recomputed-residual verdict. Rows where they disagree are the false acceptances (`false keeps` in the frozen CSV schema) that the paper turns on.

---

## 5. Reproduction guide

Run all commands from the repository root; add `--help` to any driver for its full flag
list. Runtimes are Platform A.

### 5.1 Reduced direct floor atlas — *Fig. S1*

```bash
python experiments/phase5/run_direct_floor_atlas.py \
  --seeds 7,13,19 --probabilities 0.10,0.12,0.15,0.18,0.20,0.35 \
  --out-dir experiments/phase5/results/direct_floor_atlas_seeded

python experiments/phase5/validate_admissibility_detector.py \
  --atlas experiments/phase5/results/direct_floor_atlas_seeded/direct_floor_atlas.csv \
  --critical experiments/phase5/results/direct_floor_atlas_seeded/direct_floor_critical.csv \
  --safety-factors 1,10,100 \
  --out-dir experiments/phase5/results/admissibility_detector_validation
```

Reduced 24×12×6 assembled/direct solves; establishes that a floor boundary exists before
the full hierarchy is involved. Minutes, low memory.

### 5.2 Retrospective GMG labels — *thresholds are fixed here*

```bash
python experiments/phase5/validate_gmg_solver_floor_detector.py \
  --high-residual-threshold 1e-2 --plateau-residual-threshold 1e-4 \
  --plateau-ratio-threshold 0.6 \
  --out-dir experiments/phase5/results/gmg_solver_floor_detector
```

The 16 development states. These are the *only* states used to choose the rule
thresholds; everything below is held out.

### 5.3 Held-out suite — *Fig. 5* (the main evaluation)

72 cantilever states (8 seeds × 9 solid probabilities) and 30 bridge states
(5 seeds × 6 probabilities) under the guarded policy:

```bash
for SEED in 41 43 47 53 59 61 67 71; do
  python experiments/phase5/run_gmg_floor_detector_prospective.py \
    --preset cantilever_gpu_medium --seeds $SEED \
    --probabilities 0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35 \
    --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
    --high-residual-threshold 1e-2 --plateau-residual-threshold 1e-4 \
    --out-dir experiments/phase5/results/heldout_gmg_detector_cantilever_s41_71_guarded_true_residual
done

for SEED in 41 43 47 53 59; do
  python experiments/phase5/run_gmg_floor_detector_prospective.py \
    --preset bridge_gpu_medium --seeds $SEED \
    --probabilities 0.10,0.15,0.20,0.25,0.30,0.35 \
    --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
    --high-residual-threshold 1e-2 --plateau-residual-threshold 1e-4 \
    --out-dir experiments/phase5/results/heldout_gmg_detector_bridge_s41_59_guarded_true_residual
done
```

≈40 s/state, ≈70 minutes total. Expect 102/102 `solve_converged=1`, 24 states preserving
`1e-12`, 78 escalated to `1e-3`, and 4 cantilever rows with
`solver_reported_converged=1, solve_converged=0` at the original floor — the false
acceptances.

**Reference classifications** come from a separate forced full-budget audit (thresholds disabled, so the
original floor is always attempted to exhaustion):

```bash
python experiments/phase5/run_gmg_floor_detector_prospective.py \
  --preset cantilever_gpu_medium --seeds 41 \
  --probabilities 0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35 \
  --high-residual-threshold 1e99 --plateau-residual-threshold 1e99 \
  --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
  --out-dir experiments/phase5/results/heldout_full_true_labels_cantilever_s41
```

Repeat per seed and geometry; directory names must match the list in
`summarize_review_experiments.py`. ≈126 s/state, ≈3.5 hours total. Fixed-floor and
severity-jump baselines on the same matrix are queued by
`run_observed_queue.py --task fixed_floor_baselines` and `--task severity_jump_baselines`.

### 5.4 Bridge ladder and fine sweep — *Fig. 8a–c*

```bash
python experiments/phase5/run_gmg_floor_detector_prospective.py \
  --preset bridge_gpu_medium --seeds 23 --probabilities 0.10,0.20,0.35 \
  --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
  --out-dir experiments/phase5/results/gmg_solver_floor_detector_transfer_bridge_seed23_ladder

python experiments/phase5/run_gmg_floor_detector_prospective.py \
  --preset bridge_gpu_medium --seeds 23,41 --probabilities 0.10,0.20,0.35 \
  --baseline-rho-min 1e-12 --raised-rho-mins 1e-10,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2 \
  --out-dir experiments/phase5/results/fine_ladder_bridge_seed23_41_strict_true_residual
```

The fine sweep shows the reported ladder is conservative but not arbitrary: 4 of 6
states need `1e-2`, one needs `1e-3`, one is admissible at `1e-6`.

### 5.5 Optimized-density transfer — *Fig. S3* (needs the density bundle)

```bash
python experiments/phase5/run_gmg_floor_detector_density_field.py \
  --preset cantilever_gpu_medium \
  --density-paths experiments/paper2/runs/C64_MF/rho_final.npy \
  --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
  --out-dir experiments/phase5/results/optimized_density_C64_MF_strict_true_residual
```

Preset per artifact: `C64_MF → cantilever_gpu_medium`, `C216_MF → cantilever_gpu_large`,
`C512_MF → cantilever_gpu_xlarge`, `B500_MF → bridge_gpu_500k`,
`Brk500_MF → bracket_gpu_500k`, `M500_MF → mbb_gpu_xlarge`, `T500_MF → torsion_gpu_500k`,
`Col500_MF → column_gpu_500k`, `C1M_MF → cantilever_gpu_xxlarge`. Seven of the eight
reported fields preserve `1e-12`; the large bridge field is connected under the `rho ≥ 0.5`
support diagnostic and still requires `1e-2`.

### 5.6 What a fixed raised floor changes — *Fig. 7*

Compliance and timing controls, then the two perturbation studies:

```bash
python experiments/phase5/run_gmg_fixed_floor_controls.py \
  --manifest experiments/phase5/fixed_floor_control_manifest.csv \
  --out-dir experiments/phase5/results/gmg_fixed_floor_controls_strict_true_residual

# severe random true-keep states (needs the held-out audit from 5.3)
python experiments/phase5/analyze_gmg_sensitivity_perturbation.py \
  --floors 1e-12,1e-3,1e-2 \
  --out-dir experiments/phase5/results/heldout_true_keep_sensitivity_perturbation

# optimized designs (needs the density bundle)
python experiments/phase5/analyze_optimized_density_sensitivity_perturbation.py \
  --floors 1e-12,1e-3,1e-2 \
  --out-dir experiments/phase5/results/optimized_density_sensitivity_perturbation
```

The last two ask the same question of different state families, and the difference
between them is the point: severe random states move a great deal, optimized designs move
much less. `bash experiments/phase5/run_scale_and_perturbation_block.sh` runs the
optimized-design study followed by the million-element scale check.

### 5.7 Robustness — *Fig. 8d, Figs. S2 and S4*

```bash
python experiments/phase5/analyze_gmg_threshold_sensitivity.py \
  --out-dir experiments/phase5/results/gmg_solver_floor_detector_sensitivity

python experiments/phase5/run_observed_queue.py --task review_extension_sweeps \
  --hard-timeout-min 75 --idle-timeout-min 25
```

`review_extension_sweeps` covers the SIMP-exponent sweep, the original-floor sweep, and
the seven-variant stack ablation (3-level hierarchy, Jacobi smoother, W-cycle, tolerance
`1e-5`/`1e-7`, and removal of the fine-level adaptive correction — the
`no_root_correction` variant in the CSV schema). The precision variant is run directly:

```bash
python experiments/phase5/run_gmg_floor_detector_prospective.py \
  --preset bridge_gpu_medium --seeds 43 --probabilities 0.10,0.20,0.35 \
  --stack-variant fp32_fine --fine-smoother fp32 \
  --high-residual-threshold 1e-2 --plateau-residual-threshold 1e-4 \
  --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
  --out-dir experiments/phase5/results/replication/fp32_fine/bridge_s43
```

`bash experiments/phase5/run_replication_block.sh` runs the FP64 and FP32 blocks for both
geometries in sequence. Running the FP64 block on a second GPU/CuPy/CUDA combination is
also how the platform replication in the paper was produced.

### 5.8 Optimization trajectories — *Fig. 3*

```bash
# fixed-floor controls, no acceptance test (the unguarded baseline)
python experiments/phase5/run_observed_queue.py --task simp_floor_trajectories \
  --hard-timeout-min 180 --idle-timeout-min 45

# the guarded policy applied at every outer iteration
python experiments/phase5/run_simp_floor_trajectory.py \
  --presets cantilever_gpu_medium,bridge_gpu_medium --iters 40 \
  --policy guarded_adaptive --baseline-rho-min 1e-12 --ladder-rho-mins 1e-3,1e-2 \
  --out-dir experiments/phase5/results/guarded_adaptive_trajectories
```

`--policy fixed_floor` (the default) sweeps `--rho-mins` with no acceptance test;
`--policy guarded_adaptive` runs Algorithm 1 at every outer iteration and records
`solver_true_rel_residual`, `selected_rho_min`, and `policy_trigger` per iteration.

The fixed-`1e-12` cantilever control is the one to look at: from outer iteration 19 its
state solve returns at the 300-iteration cap every time and its compliance oscillates
between 0.484 and 4.502 — while the run reports `failures: 0`, because no acceptance test
is applied. The guarded run on the same problem accepts all 40 solves (largest accepted
true residual `9.84e-07`) and switches to `1e-3` at iteration 17, the first iteration after
the continuation step.

### 5.9 Summaries and figures

```bash
python experiments/phase5/summarize_review_experiments.py   # no arguments; fixed paths
python experiments/phase5/make_paper5_v3_figures.py         # manuscript figure set
```

`summarize_review_experiments.py` consolidates completed result directories into
`experiments/phase5/results/review_experiment_summary/`, which is what the figure scripts
read; it skips directories that do not exist, so partial reruns produce partial summaries.
`make_paper5_v2_figures.py` and `make_paper5_journal_figures.py` are retained and produce
the earlier figure sets.

### 5.10 Conditioning of the reduced problem (CPU only)

```bash
python experiments/phase5/run_direct_floor_conditioning.py  # ~2 minutes, no GPU
```

Computes the extreme eigenvalues of the constrained operator for the 18 reduced
`24x12x6` atlas fields at eight floors and writes
`experiments/phase5/results/direct_floor_conditioning/direct_floor_conditioning.csv`.
This is the source of the main-text conditioning figure (Fig. 4), including the
hierarchy-independent screening rule `rho_min >~ c(rho) * eps / tau` of its panel (c);
it is a diagnostic of the operator, and no result of it is used to accept a solve.

### 5.11 Fixed-preconditioner control — the attribution check

Disabling both adaptive preconditioner components leaves a fixed linear V-cycle (the
flexible outer method then coincides with right-preconditioned GMRES). Rerunning the four
false-acceptance states, their neighbours, one admissible state, and two severe bridge
states under the full guarded protocol:

```bash
python experiments/phase5/run_gmg_floor_detector_prospective.py   --preset cantilever_gpu_medium --seeds 47,59,71 --probabilities 0.30,0.35   --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2   --high-residual-threshold 10 --plateau-residual-threshold 10   --coarse-correction-policy none --root-local-correction-mode none   --stack-variant fixed_linear_preconditioner   --out-dir experiments/phase5/results/fixed_precond_control_cantilever
# repeat with --seeds 43 --probabilities 0.35, and with
# --preset bridge_gpu_medium --seeds 43 --probabilities 0.10,0.20
```

The thresholds of `10` disable the escalation rule, so the original floor is always
attempted to exhaustion. Expected: **no false acceptance in any of the 18 floor
attempts** — every rejected attempt returns with its projected estimate 3–6 orders of
magnitude above tolerance, so every failure is visible. The four false-acceptance states
fail visibly at `1e-12` and are accepted at `1e-3` in 15–42 iterations; the admissible
state needs 130 iterations instead of the full stack's 30; the two severe bridge states
fail at every tested floor although the full stack accepts them at `1e-2`. The
stopping-estimate drift is therefore tied to the iterate-dependent preconditioner, and
the components that make severe states solvable are the ones that make the flag
untrustworthy.

---

## 6. Extending the study

- **New geometry:** add a `ProblemSpec` to `src/gpu_fem/presets.py` and pass `--preset`.
  Nothing in the policy is geometry-specific.
- **New stack variant:** the drivers expose `--n-levels`, `--smoother-type`,
  `--cycle-type`, `--fine-smoother`, `--coarse-correction-policy`,
  `--root-local-correction-mode`, `--inner-krylov-steps`, and `--tol`. Pass
  `--stack-variant <name>` so the variant is recorded in the output CSV.
- **New rule:** thresholds are CLI flags; `analyze_gmg_threshold_sensitivity.py` maps the
  safe region around the reported triplet.
- **A different acceptance criterion:** the guard is one line in `solve_at_floor()`. A
  checkpointed variant that recomputes the true residual at iterations 50 and 100 is the
  most useful untested extension — it is what would make the probe itself trustworthy.

---

## 7. Interpretation boundary

- The policy is an empirical solver-control layer for the tested stack, not a new SIMP
  material model and not a multigrid convergence theorem. Raising the floor provably
  regularizes the operator (monotone coercivity), but that does not prove convergence for
  any particular hierarchy.
- The acceptance guard, unlike the thresholds, is not tuned to anything. It is the part
  worth adopting regardless of stack.
- Fixed raised floors are faster and converge everywhere in our held-out matrix. They are
  not neutral: they change the operator, and on severe states they change compliance and
  gradients substantially. Read §5.6 before choosing one.
- The two in-loop trajectories demonstrate executable semantics, not optimization-path
  preservation.
- All experiments use the FP64 fine path unless `--fine-smoother fp32` is passed.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `CUDA_ERROR_NO_BINARY_FOR_GPU` | CuPy wheel has no kernels for your architecture; install `cupy-cuda13x` on Blackwell |
| `NVRTC_ERROR_COMPILATION` mentioning `__nv_bfloat16` in CCCL headers | CuPy 14 CUB reductions on CUDA 13; `export CUPY_ACCELERATORS=""` |
| `ModuleNotFoundError: gpu_fem` | Run from the repository root; drivers insert `src/` on `sys.path` themselves |
| `FileNotFoundError: run_experiments_e1_e10.py` | `experiments/paper4/` is required by every GPU driver; do not prune it |
| Driver cannot find `rho_final.npy` | Optimized-density inputs ship separately; see §1 |
| Out of memory on a large preset | Start with `cantilever_gpu_medium`; the 500k presets need ≈8–19 GiB |
| Plot script cannot find a CSV | Run `summarize_review_experiments.py` first, or the experiment it summarizes |
| Very slow first run | NVRTC compiles kernels on first use; set `CUPY_CACHE_DIR` to a persistent path |

---

## 9. Citation

See `CITATION.cff` for this repository. The accompanying paper is:

> Yang, S., Wang, J., and Wang, Y. (2026). *When a positive SIMP density floor is not
> enough: solver admissibility and guarded floor selection in matrix-free 3D topology
> optimization.* arXiv preprint (identifier to be added on announcement).

Companion preprints for the operator and hierarchy this study wraps:

- *Matrix-Free 3D SIMP Topology Optimization with Fused Gather-GEMM-Scatter Kernels*,
  <https://arxiv.org/abs/2604.18020>
- *A Matrix-Free Galerkin Multigrid Solver and Failure-Mode Screen for Single-GPU 3D SIMP
  Linear Systems*, <https://arxiv.org/abs/2604.26441>

Those papers contribute the kernels and the hierarchy. This one contributes the
admissibility question, the guarded policy, and the evidence that a solver flag is not an
acceptance criterion.

## 10. License

BSD 3-Clause. See `LICENSE`.
