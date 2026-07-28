"""Compliance and sensitivity perturbation of fixed raised floors on optimized designs.

The companion script ``analyze_gmg_sensitivity_perturbation.py`` measures what a
fixed raised floor does to compliance and to the compliance gradient on the
held-out Bernoulli random states. Those states are deliberately severe. This
script asks the same question on the stored optimized SIMP density fields whose
original floor is solver-admissible, which is the case that matters for a design
workflow.

Differences from the random-field script:

* the density field is the stored optimized field, not a floor-substituted
  binary mask, so the sensitivity convention is the ordinary SIMP derivative
  ``dC/drho = -p (1 - rho_min) rho^(p-1) u_e^T K_e u_e``;
* "solid" elements are those with ``rho >= solid_threshold`` under the same
  support diagnostic used elsewhere in the paper.

Every solve is accepted only on a recomputed true residual, exactly as in the
guarded policy.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Reader-facing label, preset, stored density field. These are the seven
# optimized families whose original floor is admissible in the transfer study;
# the large bridge field is excluded because it requires escalation, so a
# "floor that was not needed" comparison is not defined for it.
DEFAULT_CASES = [
    ("Medium cantilever", "cantilever_gpu_medium", "C64_MF"),
    ("216k cantilever", "cantilever_gpu_large", "C216_MF"),
    ("Large cantilever", "cantilever_gpu_xlarge", "C512_MF"),
    ("Large bracket", "bracket_gpu_500k", "Brk500_MF"),
    ("Large MBB beam", "mbb_gpu_xlarge", "M500_MF"),
    ("Large torsion", "torsion_gpu_500k", "T500_MF"),
    ("Large column", "column_gpu_500k", "Col500_MF"),
]


def _load_paper4_runner():
    path = ROOT / "experiments" / "paper4" / "run_experiments_e1_e10.py"
    spec = importlib.util.spec_from_file_location("paper4_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _safe_norm_ratio(num: np.ndarray, den: np.ndarray) -> float:
    return float(np.linalg.norm(num) / max(float(np.linalg.norm(den)), 1e-300))


def _safe_linf_ratio(num: np.ndarray, den: np.ndarray) -> float:
    return float(np.max(np.abs(num)) / max(float(np.max(np.abs(den))), 1e-300))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floors", default="1e-12,1e-3,1e-2")
    parser.add_argument("--reference-rho-min", type=float, default=1e-12)
    parser.add_argument("--penal", type=float, default=4.5)
    parser.add_argument("--solid-threshold", type=float, default=0.5)
    parser.add_argument("--restart", type=int, default=300)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--density-root", default=str(ROOT / "experiments" / "paper2" / "runs"))
    parser.add_argument("--only", default="", help="comma-separated artifact ids, empty means all")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results"
                    / "optimized_density_sensitivity_perturbation"),
    )
    args = parser.parse_args()

    import cupy as cp
    from gpu_fem.multigrid_v4 import _cupy_fgmres

    floors = _parse_float_list(args.floors)
    if args.reference_rho_min not in floors:
        floors = [args.reference_rho_min, *floors]

    only = {item.strip() for item in args.only.split(",") if item.strip()}
    cases = [c for c in DEFAULT_CASES if not only or c[2] in only]
    paper4 = _load_paper4_runner()
    density_root = Path(args.density_root)

    rows: list[dict] = []
    print("label,rho_min,converged,iters,true_residual,rel_dc_l2_solid,rel_compliance_change",
          flush=True)

    for label, preset, artifact in cases:
        density_path = density_root / artifact / "rho_final.npy"
        spec, bc, free, free_gpu, edof_gpu, F_free_gpu, mf_op, gmg = paper4._build_components(
            preset,
            n_levels=4,
            fine_smoother="fp64",
            smoother_type="chebyshev",
            cycle_type="v",
            coarse_correction_policy="residual_line_search",
            root_local_correction_mode="node_block_inner_fgmres",
            root_local_correction_inner_steps=20,
            root_local_correction_inner_restart=20,
            inner_krylov_steps=10,
            inner_krylov_restart=10,
        )
        n_elem = spec.nelx * spec.nely * spec.nelz
        b_norm = float(cp.linalg.norm(F_free_gpu))

        density = np.clip(np.asarray(np.load(density_path), dtype=np.float64).reshape(-1), 0.0, 1.0)
        if density.size != n_elem:
            raise ValueError(f"{density_path} has {density.size} entries, expected {n_elem}")
        solid = density >= args.solid_threshold
        rho_gpu = cp.asarray(density, dtype=cp.float64)

        reference = None
        for floor in floors:
            E_e = floor + (1.0 - floor) * rho_gpu**args.penal
            setup_start = time.perf_counter()
            gmg.setup(E_e)
            cp.cuda.Stream.null.synchronize()
            setup_time = time.perf_counter() - setup_start

            def A_op(x, E_e=E_e):
                return mf_op.matvec(x, E_e)

            history: list[float] = []
            solve_start = time.perf_counter()
            x, iters, reported = _cupy_fgmres(
                A_op, F_free_gpu, gmg.apply,
                tol=args.tol, maxiter=args.maxiter, restart=args.restart, history=history,
            )
            cp.cuda.Stream.null.synchronize()
            solve_time = time.perf_counter() - solve_start
            true_residual = float(cp.linalg.norm(F_free_gpu - A_op(x)) / b_norm)

            U_gpu = cp.zeros(bc.ndof, dtype=cp.float64)
            U_gpu[free_gpu] = x
            Ue = U_gpu[edof_gpu]
            ce = ((Ue @ mf_op._KE_unit) * Ue).sum(axis=1)
            dc = cp.asnumpy(
                -args.penal * (1.0 - floor) * rho_gpu ** (args.penal - 1.0) * ce
            )
            compliance = float(cp.dot(F_free_gpu, x).get())

            result = dict(dc=dc, compliance=compliance)
            if floor == args.reference_rho_min:
                reference = result
            assert reference is not None, "reference floor must be solved first"

            dc_diff = dc - reference["dc"]
            row = {
                "label": label,
                "artifact_id": artifact,
                "preset": preset,
                "n_elem": int(n_elem),
                "rho_min": floor,
                "reference_rho_min": args.reference_rho_min,
                "penal": args.penal,
                "solid_threshold": args.solid_threshold,
                "solid_count": int(solid.sum()),
                "solid_fraction": float(solid.mean()),
                "sensitivity_convention": "stored_density_derivative",
                "setup_time_s": setup_time,
                "solve_time_s": solve_time,
                "solve_converged": int(true_residual <= args.tol),
                "solver_reported_converged": int(reported),
                "solve_iters": int(iters),
                "solve_final_rel_residual": true_residual,
                "compliance": compliance,
                "reference_compliance": reference["compliance"],
                "rel_compliance_change": float(
                    (compliance - reference["compliance"])
                    / max(abs(reference["compliance"]), 1e-300)
                ),
                "rel_dc_l2_all": _safe_norm_ratio(dc_diff, reference["dc"]),
                "rel_dc_l2_solid": _safe_norm_ratio(dc_diff[solid], reference["dc"][solid]),
                "rel_dc_linf_solid": _safe_linf_ratio(dc_diff[solid], reference["dc"][solid]),
                "pearson_dc_solid": _pearson(dc[solid], reference["dc"][solid]),
            }
            rows.append(row)
            print(f"{label},{floor:.0e},{row['solve_converged']},{iters},"
                  f"{true_residual:.3e},{row['rel_dc_l2_solid']:.6g},"
                  f"{row['rel_compliance_change']:.6g}", flush=True)

        del rho_gpu, gmg, mf_op, F_free_gpu, free_gpu, edof_gpu
        cp.get_default_memory_pool().free_all_blocks()

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "optimized_density_sensitivity_perturbation.csv", rows, list(rows[0]))
    print(f"Wrote {out_dir / 'optimized_density_sensitivity_perturbation.csv'}", flush=True)


if __name__ == "__main__":
    main()
