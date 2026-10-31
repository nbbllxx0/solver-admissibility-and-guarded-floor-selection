# Release status

Public repository: <https://github.com/nbbllxx0/solver-admissibility-and-guarded-floor-selection>

Note on tags: the first Zenodo-archived GitHub release was tagged `v1.0.0`
(DOI 10.5281/zenodo.21697876) although this file already recorded internal
versions 2.0.0 and 2.1.0; the tag numbering rejoins this file's numbering at
v2.2.0.

## Version 2.3.0 — 2026-10-31

Aligned with the revised manuscript *Iterative solvers in GPU multigrid topology
optimization can report false convergence, and a recomputed residual catches it*.

- **Convergence-flag logging in the optimization loop.** `solver_v4.py` records the
  solver's convergence flag, `simp_gpu.py` writes it per state solve, and
  `run_simp_floor_trajectory.py` reports `solves_flagged_converged`,
  `false_acceptances` and `solves_at_cap` per run. No solver behaviour changed.
- **New figure generator `make_paper5_srep_figures.py`** for main-text Figs. 2–6, with
  its style module `pubstyle.py`; Fig. 1 is a schematic and is not generated here.
  `make_paper5_v3_figures.py` updated for the supplementary figures.
- **Schematic code removed from the earlier figure scripts:** `make_paper5_v2_figures.py`,
  `make_paper5_v3_figures.py` and `make_paper5_journal_figures.py` no longer draw the
  policy schematic. The guard-evidence panel of the earlier policy figure is now written
  on its own as `fig1b_guard_evidence` (v2) and `fig2b_guard_evidence` (v3).
- **Final Supplementary Fig. S6 layout:** `make_paper5_srep_conditioning.py` renders
  the conditioning data at 165 mm width with separate panel labels and a shared
  legend. It can use the deposited CSVs without a GPU and checks the strict
  residual counts (114 passes and 30 failures) and all 18 floor screens.
  Its default output is now `figures/generated/figS5_direct_conditioning.pdf`.
- **Figures are written without metadata.** `pubstyle.save` and every `savefig` call
  clear the tool and date fields of PDF and PNG files, and `pubstyle.save` removes the
  comment that matplotlib adds to SVG files.
- **40 analysis tables added (90 in total):** the standard-floor optimization runs
  (Fig. 3, Supplementary Table S8), the strict held-out replay history (Figs. 1 and
  2a), the optimized-bridge solve history (Fig. 6c), the replication and FP32 runs
  (Supplementary Table S15), and a per-seed rerun of the guarded held-out suite.
- `MANIFEST.csv` `supports` column renumbered to the revised manuscript; five tables
  used only by the earlier preprint version are marked as such.
- README: reproduction commands for the standard-floor runs and the per-seed
  held-out rerun; the per-seed loop now writes each seed to its own folder;
  figure and table references renumbered; a third hardware setup listed.
- Platform software corrected from the run records: Platform A ran Python 3.9.23
  (not 3.10.18) and Platform B ran Python 3.10.18 (not 3.11); the environment files
  now say they pin newer Python versions. The `cupy-cuda12x` note for Blackwell GPUs
  is softened: version 13.6.0 with CUDA runtime 12.9 ran on an RTX 5090.

## Version 2.2.1 — 2026-07-30

- **Optimized-density input arrays deposited (`data/optimized_density_inputs/`).**
  The nine stored fields used by the transfer study, the SI gallery, and the
  million-element scale check (33.1 MB, float64), each checksummed in a
  MANIFEST.csv with shape, dtype, minimum, and mean. With these, the
  optimized-density transfer study is reproducible from this repository alone.
- Deposit metadata corrected: RELEASE_STATUS table count (50, was misstated as
  40), manifest `supports` cross-references updated to the journal manuscript's
  table numbering, CITATION.cff terminology aligned (preserve/escalate).

## Version 2.2.0 — 2026-07-30

- **Research-data deposit completed (`data/analysis_tables/`).** 50 analysis
  tables covering every figure and table in the paper, including the unguarded
  fixed-floor trajectories, both guarded held-out audits with rejected-attempt
  histories, the bridge fine ladder, the million-element scale check, the
  threshold-sensitivity grid, and the optimized-density source metadata. Each
  file is listed in `MANIFEST.csv` with size, SHA-256, and the manuscript
  element it supports.
- `.gitattributes` pins the deposited CSVs to LF so the manifest hashes verify
  on Windows checkouts.
- `CITATION.cff` carries the Zenodo concept and version DOIs and matches this
  file's version numbering.

## Version 2.1.0 — 2026-07-28

- **Fixed-preconditioner control documented (README §5.11).** No new code: existing
  driver flags reproduce the nine-state attribution check reported in the paper
  (no false acceptance with the adaptive preconditioner components disabled).
- **Figure generator `make_paper5_v3_figures.py` updated** to the submitted figure set:
  preserve/escalate vocabulary throughout, three-panel conditioning figure with the
  screening-rule panel, repaired label collisions, and semantic-colour fixes.
- **README revised**: preserve/escalate terminology (with the frozen `keep`/`raise`
  CSV-schema mapping noted), figure references aligned with the submitted manuscript,
  Platform-B environment file (`environment-blackwell.yml`) listed in the layout,
  and a citation block for the accompanying paper.
- `CITATION.cff` completed with abstract and repository URL.

## Version 2.0.0 — 2026-07-27

Aligned with the restructured manuscript *When a positive SIMP density floor is not
enough: solver admissibility and guarded floor selection in matrix-free 3D topology
optimization*.

Changes since 1.0.0:

- **Added `experiments/paper4/run_experiments_e1_e10.py`.** Every phase-5 GPU driver
  imports `_build_components()` from this file. Version 1.0.0 omitted it, so no GPU
  experiment could actually be run from a clean clone. `ci/smoke_check.py` now fails if it
  or any other load-bearing file is missing.
- **Added `experiments/phase5/analyze_optimized_density_sensitivity_perturbation.py`.**
  Measures compliance and compliance-gradient perturbation under fixed raised floors on
  optimized designs, the counterpart to the existing random-state study.
- **Added `experiments/phase5/make_paper5_v2_figures.py`**, which builds the manuscript
  figure set (6 main + 4 supplementary). The earlier ten-figure generator is retained.
- **Added `run_replication_block.sh`, `run_scale_and_perturbation_block.sh`, and
  `run_scale_1m_restarted.sh`** for the second-platform replication, the FP32 precision
  ablation, the optimized-design perturbation study, and the million-element scale check
  (the last needs a restarted Krylov method to fit the basis in memory; see README §3).
- **`run_gmg_floor_detector_density_field.py` gained `--fine-smoother`** and records the
  value in its output, so optimized-density transfers can be run in reduced precision.
- **Blackwell support.** `src/gpu_fem/cuda_fused_matvec.py` now selects the NVRTC C++
  standard from the installed CuPy version (`-std=c++17` on CuPy ≥ 14, `-std=c++14`
  otherwise), which is what CuPy 14's bundled CCCL headers require. Added
  `environment-blackwell.yml`; README section 3 documents the `CUPY_ACCELERATORS=""`
  requirement on CUDA 13.
- **Rewritten README** with the corrected command set. Several version 1.0.0 commands did
  not match the actual CLIs (`run_gmg_floor_detector_density_field.py` takes `--preset`
  and `--density-paths`, not `--case`/`--density-kind`; `summarize_review_experiments.py`
  takes no arguments; there is no `guarded_adaptive_trajectories` queue task).

## Verification for this release

- `python ci/smoke_check.py` passes: 54 files parse, all required files present.
- The quick-start GPU command in README section 2 reproduces the reported row
  (`trigger=keep`, 30 FGMRES iterations, `r50 ≈ 5.3e-07`) on both verified platforms.

## What is not in this repository

Raw logs and generated figures are not stored here. The reduced result tables the
paper's figures and tables are computed from *are* here, under
`data/analysis_tables/` (since v2.2.0; see that entry above). The stored
optimized-density input arrays are deposited under `data/optimized_density_inputs/`
(since v2.2.1). Everything except the hardware-specific timing tables can be
regenerated from this repository alone.
