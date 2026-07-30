# Optimized-density input arrays

The nine stored density fields used as solver inputs by the article's
optimized-density transfer study, SI gallery, and million-element scale check.
Each is the final density field of a prior 120-iteration matrix-free SIMP
optimization run, saved as a flat float64 NumPy array (`np.load` and reshape to
the mesh in the article's geometry table).

These are **inputs carried over from prior work**, not results of this study:
the article evaluates the guarded floor-selection policy *on* them. Their
source-run diagnostics (optimizer, iterations, recorded compliance, grayness,
capped-solve percentages) are deposited in
`../analysis_tables/review_experiment_summary__optimized_density_case_details.csv`
and reported in the article's optimized-source-metadata table. For two fields
(MBB and bridge) every source solve terminated at the iteration cap, so they
are treated as optimizer-generated solver stress states, not as converged
designs — see the article.

| File | Elements | Family |
| --- | --- | --- |
| `C64_MF__rho_final.npy` | 64,000 | cantilever |
| `C216_MF__rho_final.npy` | 216,000 | cantilever |
| `C512_MF__rho_final.npy` | 512,000 | cantilever |
| `B500_MF__rho_final.npy` | 514,500 | bridge |
| `Brk500_MF__rho_final.npy` | 512,000 | bracket |
| `M500_MF__rho_final.npy` | 514,500 | MBB beam |
| `T500_MF__rho_final.npy` | 499,125 | torsion |
| `Col500_MF__rho_final.npy` | 500,000 | column |
| `C1M_MF__rho_final.npy` | 1,000,000 | cantilever (scale check) |

`MANIFEST.csv` records size, SHA-256, shape, dtype, minimum, and mean for each
array. The mean equals the run's volume fraction; the minimum is the source
run's effective density floor.

Licence: same terms as the rest of this repository (BSD 3-Clause). If you use
these fields, please cite the article.
