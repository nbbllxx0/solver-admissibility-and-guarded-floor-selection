# Solver-Admissibility Code Release

Code-only release for the experiment suite accompanying:

> Yang, S., Wang, J., and Wang, Y. (2026).  
> *Solver-Admissibility Testing of SIMP Density Floors in Matrix-Free GMG-FGMRES.*

This repository is intentionally a code release. It contains the solver source,
experiment drivers, analysis scripts, plotting scripts, environment file, and
configuration manifests needed to rerun the experiments. It does not contain the
manuscript source, PDFs, generated figures, result CSVs, logs, caches, or large
stored density arrays. Fresh runs recreate those artifacts under local
`experiments/phase5/results/` and figure-output directories.

## Scope

The paper studies a solver-facing failure mode in density-based topology
optimization: a positive SIMP density floor can still be inadmissible for a
specific matrix-free geometric-multigrid FGMRES hierarchy and residual
tolerance. The code here implements the tested matrix-free finite-element stack,
the residual-probe floor policy, direct and GMG validation experiments, fixed
floor controls, policy-cost accounting, mechanism ablations, sensitivity
sweeps, and trajectory controls.

This release is not a frozen reviewer bundle. It is the public code base for
rerunning or extending the experiments. Exact reported numeric tables require
rerunning the scripts on compatible hardware or combining this code release with
the separate artifact/result bundle.

## Code-Only Boundary

Included:

- `src/gpu_fem/`: matrix-free finite-element, SIMP, boundary-condition, solver,
  and GMG support code.
- `experiments/phase5/`: experiment drivers, analyzers, plotting scripts,
  summary builder, figure generator, and small experiment manifest CSVs.
- `environment.yml`: pinned software environment used for the paper package.
- `ci/smoke_check.py`: lightweight source compilation check.
- `LICENSE`, `CITATION.cff`, and this README.

Excluded by design:

- manuscript `.tex`, `.pdf`, `.bbl`, and journal-submission files;
- generated figures and rendered pages;
- result CSV/JSON/NPY outputs under `experiments/phase5/results/`;
- stored optimized-density input arrays under `experiments/paper2/runs/`;
- runtime scratch folders, CuPy caches, logs, and local temporary outputs.

For exact reproduction of the optimized-density transfer rows and the 3D
gallery, provide the separate fixed-density artifact bundle at
`experiments/paper2/runs/<case>/rho_best.npy` and
`experiments/paper2/runs/<case>/rho_final.npy`, together with the associated
`meta.json` and `iters.csv` files. Without those fixed input states, the random
field, direct, GMG, policy, ablation, and trajectory experiments remain
runnable, but optimized-density transfer and Figure 7 cannot be reproduced
exactly.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/gpu_fem/` | Solver implementation and reusable GPU-FEM utilities. |
| `experiments/phase5/run_direct_floor_atlas.py` | Reduced assembled/direct floor sweeps. |
| `experiments/phase5/validate_admissibility_detector.py` | Leave-one-seed direct detector validation. |
| `experiments/phase5/validate_gmg_solver_floor_detector.py` | Retrospective GMG residual-probe rule check. |
| `experiments/phase5/run_gmg_floor_detector_prospective.py` | Prospective random cases, held-out guarded cases, bridge transfer, and baseline/fallback policy variants. |
| `experiments/phase5/run_gmg_floor_detector_density_field.py` | Optimized-density transfer on stored density fields. |
| `experiments/phase5/run_gmg_fixed_floor_controls.py` | Fixed-floor compliance/timing controls. |
| `experiments/phase5/analyze_gmg_sensitivity_perturbation.py` | True-keep sensitivity-vector perturbation under fixed raised floors. |
| `experiments/phase5/analyze_gmg_threshold_sensitivity.py` | Residual-rule threshold sensitivity grid. |
| `experiments/phase5/analyze_gmg_policy_overhead.py` | Probe, reuse, selected-solve, and failed-ladder iteration accounting. |
| `experiments/phase5/run_observed_queue.py` | Serial runner for long observed experiment queues. |
| `experiments/phase5/run_simp_floor_trajectory.py` | Fixed-floor and guarded adaptive in-loop trajectory jobs. |
| `experiments/phase5/summarize_review_experiments.py` | Consolidates completed result directories into manuscript-ready summary CSVs. |
| `experiments/phase5/make_paper5_journal_figures.py` | Regenerates paper-native figures from result summaries and density inputs. |
| `experiments/phase5/fixed_floor_control_manifest.csv` | Small input manifest for fixed-floor controls. |
| `experiments/phase5/review_required_experiments.csv` | Experiment queue/status plan used by the observed runner. |
| `ci/smoke_check.py` | Syntax-level release check that does not require a GPU. |

## Hardware Requirements

The full experiment suite was run on a single NVIDIA GeForce RTX 4090. Smaller
reduced/direct experiments are less demanding, but the GMG and trajectory
experiments use CUDA/CuPy paths and should be run on an NVIDIA GPU with a recent
driver and enough memory for the target problem size.

Practical guidance:

- Use a 24 GiB GPU for the largest reported cases.
- Start with reduced/direct and small prospective GMG cases before launching
  held-out or trajectory queues.
- Keep runtime scratch, CuPy cache, and result output on a local disk with
  enough free space for generated CSV/JSON/NPY outputs.

## Software Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate paper5-solver-admissibility
```

If you are using an existing environment, the key package versions are pinned in
`environment.yml`: Python 3.10, CuPy 13.6, NumPy 2.2, SciPy 1.15, Matplotlib
3.10, pandas 2.3, scikit-image 0.25, PyVista 0.46, and pyamg 5.3.

Run the source-level smoke check:

```bash
python ci/smoke_check.py
```

This only verifies that release Python files compile. It intentionally does not
import CUDA modules or run GPU solves.

## Output Convention

Scripts write generated outputs under:

```text
experiments/phase5/results/
```

This directory is ignored in the code-only release. Result directories are
created by the experiment scripts as needed. If you want a clean rerun, remove or
move the relevant subdirectory before launching the script again.

The common pattern is:

```bash
python experiments/phase5/<script>.py --out-dir experiments/phase5/results/<run_name>
```

Some legacy plotting scripts have default input paths tied to the paper
workspace. Prefer passing explicit `--input`, `--out-dir`, or equivalent
arguments when rerunning in a fresh clone.

## Recommended Reproduction Order

The following order rebuilds the evidence chain from cheaper diagnostics to
expensive GPU queues.

### 1. Reduced Direct Floor Atlas

Run reduced assembled/direct floor sweeps:

```bash
python experiments/phase5/run_direct_floor_atlas.py \
  --out-dir experiments/phase5/results/direct_floor_atlas_seeded
```

Validate the direct detector:

```bash
python experiments/phase5/validate_admissibility_detector.py \
  --atlas experiments/phase5/results/direct_floor_atlas_seeded/direct_floor_atlas.csv \
  --critical experiments/phase5/results/direct_floor_atlas_seeded/direct_floor_critical.csv \
  --out-dir experiments/phase5/results/admissibility_detector_validation
```

Scientific purpose: establish that sparse low-density frozen fields have a
measurable floor transition before testing the full matrix-free GMG hierarchy.

### 2. Retrospective GMG Detector Check

```bash
python experiments/phase5/validate_gmg_solver_floor_detector.py \
  --out-dir experiments/phase5/results/gmg_solver_floor_detector
```

Scientific purpose: evaluate the residual-probe rule on labeled full-solver
cases and record true raises, true keeps, and rescue outcomes.

### 3. Prospective Random and Held-Out GMG Runs

Use `run_gmg_floor_detector_prospective.py` for prospective random transfer,
bridge transfer, held-out guarded validation, full original-floor audits, and
policy baselines. The exact flags depend on the row you are reproducing; inspect
the script help first:

```bash
python experiments/phase5/run_gmg_floor_detector_prospective.py --help
```

Typical output roots:

```text
experiments/phase5/results/gmg_solver_floor_detector_prospective/
experiments/phase5/results/heldout_gmg_detector_cantilever_s41_71_guarded_true_residual/
experiments/phase5/results/heldout_gmg_detector_bridge_s41_59_guarded_true_residual/
experiments/phase5/results/policy_fixed_floor_baselines/
experiments/phase5/results/policy_severity_jump_baselines/
```

Scientific purpose: test whether the guarded policy converges held-out frozen
GMG cases, quantify detector-only false keeps, and compare policy time against
full original-floor-then-fallback and fixed-floor baselines.

### 4. Optimized-Density Transfer

This step requires the separate fixed-density artifact bundle:

```text
experiments/paper2/runs/C64_MF/
experiments/paper2/runs/C216_MF/
experiments/paper2/runs/C512_MF/
experiments/paper2/runs/B500_MF/
experiments/paper2/runs/Brk500_MF/
experiments/paper2/runs/M500_MF/
experiments/paper2/runs/T500_MF/
experiments/paper2/runs/Col500_MF/
```

Each directory should include `meta.json`, `iters.csv`, `rho_best.npy`, and
`rho_final.npy`.

Run one case:

```bash
python experiments/phase5/run_gmg_floor_detector_density_field.py \
  --case C64_MF \
  --density-kind final \
  --out-dir experiments/phase5/results/optimized_density_C64_MF_strict_true_residual
```

Check the script help for the exact available case/state flags:

```bash
python experiments/phase5/run_gmg_floor_detector_density_field.py --help
```

Scientific purpose: test the floor policy on fixed optimized-density states
rather than only random stress fields.

### 5. Threshold Sensitivity

```bash
python experiments/phase5/analyze_gmg_threshold_sensitivity.py \
  --predictions experiments/phase5/results/gmg_solver_floor_detector/gmg_solver_floor_detector_predictions.csv \
  --out-dir experiments/phase5/results/gmg_solver_floor_detector_sensitivity
```

Scientific purpose: show how detector false keeps and raise/keep decisions vary
around the reported high-residual and plateau thresholds.

### 6. Policy-Overhead Accounting

```bash
python experiments/phase5/analyze_gmg_policy_overhead.py \
  --out-dir experiments/phase5/results/gmg_policy_overhead
```

Scientific purpose: report selected-solve iterations separately from probe,
reuse, and failed-ladder iteration cost.

### 7. Fixed-Floor Controls

```bash
python experiments/phase5/run_gmg_fixed_floor_controls.py \
  --manifest experiments/phase5/fixed_floor_control_manifest.csv \
  --out-dir experiments/phase5/results/gmg_fixed_floor_controls_strict_true_residual
```

Analyze true-keep sensitivity perturbation:

```bash
python experiments/phase5/analyze_gmg_sensitivity_perturbation.py \
  --out-dir experiments/phase5/results/heldout_true_keep_sensitivity_perturbation
```

Scientific purpose: show that globally raised floors can be faster but change
the operator and sensitivity vectors on benign true-keep cases.

### 8. Mechanism and Policy-Sensitivity Queues

The observed queue runner serializes longer experiment groups and skips chunks
whose expected output already exists:

```bash
python experiments/phase5/run_observed_queue.py \
  --task review_extension_sweeps \
  --hard-timeout-min 75 \
  --idle-timeout-min 25
```

This queue covers representative SIMP-exponent sensitivity, original-floor
sensitivity, and stack-mechanism ablation chunks.

Scientific purpose: test whether the selected-floor pattern is tied to one SIMP
exponent, one original floor, or one specific stack component.

### 9. Fixed-Floor and Guarded Adaptive Trajectories

Run fixed-floor trajectory controls:

```bash
python experiments/phase5/run_observed_queue.py \
  --task simp_floor_trajectories \
  --hard-timeout-min 180 \
  --idle-timeout-min 45
```

Run guarded adaptive in-loop trajectory jobs:

```bash
python experiments/phase5/run_observed_queue.py \
  --task guarded_adaptive_trajectories \
  --hard-timeout-min 180 \
  --idle-timeout-min 45
```

Scientific purpose: distinguish frozen-state solver admissibility from repeated
floor choices during optimization. The paper treats the guarded adaptive
trajectory evidence as a two-case addendum, not a broad trajectory-invariance
claim.

### 10. Summary Tables

After completing the desired result directories, rebuild consolidated summaries:

```bash
python experiments/phase5/summarize_review_experiments.py \
  --results-root experiments/phase5/results \
  --out-dir experiments/phase5/results/review_experiment_summary
```

Scientific purpose: produce the curated CSV files used by tables and
figure-generation scripts, while keeping raw stage outputs separate.

### 11. Figure Generation

Figure generation reads summary CSVs and, for the optimized-density gallery, the
separate fixed-density input states.

```bash
python experiments/phase5/make_paper5_journal_figures.py
```

By default, paper-native figure PDFs are written to the manuscript-oriented
output path used in the author workspace. For a code-only clone, either create
that output path locally or edit the script/output root before rendering.

If PyVista off-screen rendering is unavailable, the script contains a
Matplotlib/scikit-image marching-cubes fallback for the 3D topology gallery.

## Experiment Matrix

| Evidence block | Main scripts | Required inputs | Main outputs |
| --- | --- | --- | --- |
| Reduced direct floor boundary | `run_direct_floor_atlas.py`, `validate_admissibility_detector.py` | Synthetic random fields generated by script | Direct floor atlas and detector validation CSVs. |
| Retrospective GMG detector | `validate_gmg_solver_floor_detector.py` | Existing labeled GMG stage outputs or regenerated probes | Residual-rule predictions and labels. |
| Prospective/held-out GMG | `run_gmg_floor_detector_prospective.py` | Synthetic random fields generated by script | Guarded selected-floor rows, histories, timings, false-keep audits. |
| Optimized-density transfer | `run_gmg_floor_detector_density_field.py` | Separate fixed-density artifact bundle | Best/final transfer outcomes and histories. |
| Threshold sensitivity | `analyze_gmg_threshold_sensitivity.py` | Detector prediction CSVs | Threshold grid and summary CSVs. |
| Policy overhead | `analyze_gmg_policy_overhead.py` | Transfer and held-out policy outputs | Policy iteration accounting CSVs. |
| Fixed-floor controls | `run_gmg_fixed_floor_controls.py` | `fixed_floor_control_manifest.csv` | Compliance/timing controls. |
| Sensitivity perturbation | `analyze_gmg_sensitivity_perturbation.py` | Held-out true-keep outputs | Relative sensitivity-vector perturbation summaries. |
| Mechanism/sensitivity sweeps | `run_observed_queue.py` | Queue definitions in script and result roots | Mechanism ablation and policy-sensitivity summaries. |
| Trajectory controls | `run_simp_floor_trajectory.py`, `run_observed_queue.py` | Generated initial conditions and solver stack | Fixed-floor and guarded adaptive trajectory histories. |
| Summary and figures | `summarize_review_experiments.py`, `make_paper5_journal_figures.py` | Completed result directories and optional fixed-density inputs | Curated summary CSVs and figure PDFs. |

## Interpretation Notes

- The residual-probe policy is an empirical solver-control layer for the tested
  matrix-free GMG-FGMRES stack. It is not a new SIMP material model and not a
  universal multigrid convergence theorem.
- The selected floor is accepted only after a recomputed true-residual guard.
  Detector-only results should be treated as a negative control, not as the
  operational policy.
- Fixed raised floors can be faster but intentionally change the operator.
  Compare fixed-floor controls against true-keep sensitivity and compliance
  perturbation before interpreting speed alone.
- Optimized-density transfer uses stored fixed input states. In this code-only
  release those arrays are absent by design, so exact transfer/gallery
  reproduction requires the separate artifact bundle.
- Broad guarded adaptive optimization-trajectory invariance is outside the
  code release's default claim boundary. The provided trajectory scripts support
  extension studies.

## Troubleshooting

If `ci/smoke_check.py` fails, fix syntax or missing-file damage before running
GPU jobs.

If CuPy cannot see the GPU, verify the NVIDIA driver, CUDA runtime, and
`cupy-cuda12x` installation inside the active conda environment.

If a large GMG run runs out of memory, start with reduced/direct experiments or
smaller prospective cases, then scale up.

If plotting scripts cannot find input CSVs, either run the corresponding
experiment/analyzer first or pass explicit input paths. This code-only release
does not ship generated result tables.

If optimized-density scripts cannot find `rho_best.npy` or `rho_final.npy`, add
the separate fixed-density artifact bundle under `experiments/paper2/runs/`.

## License

BSD 3-Clause. See `LICENSE`.
