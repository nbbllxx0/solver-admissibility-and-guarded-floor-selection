from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SUMMARY_DIR = ROOT / "experiments" / "phase5" / "results" / "review_experiment_summary"


def _load_paper4_runner():
    path = ROOT / "experiments" / "paper4" / "run_experiments_e1_e10.py"
    spec = importlib.util.spec_from_file_location("paper4_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _random_solid_mask(spec, seed: int, probability: float) -> np.ndarray:
    n_elem = spec.nelx * spec.nely * spec.nelz
    random_field = np.random.default_rng(seed).random(n_elem)
    return random_field < probability


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
    parser.add_argument(
        "--joined-csv",
        default=str(SUMMARY_DIR / "heldout_full_true_labels_joined.csv"),
        help="Joined held-out label audit used to select true-keep cases.",
    )
    parser.add_argument("--case-filter-seed", type=int, default=-1)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all selected cases")
    parser.add_argument("--floors", default="1e-12,1e-3,1e-2")
    parser.add_argument("--reference-rho-min", type=float, default=1e-12)
    parser.add_argument("--penal", type=float, default=4.5)
    parser.add_argument("--restart", type=int, default=300)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--out-dir",
        default=str(
            ROOT
            / "experiments"
            / "phase5"
            / "results"
            / "heldout_true_keep_sensitivity_perturbation"
        ),
    )
    args = parser.parse_args()

    import cupy as cp
    from gpu_fem.multigrid_v4 import _cupy_fgmres

    floors = _parse_float_list(args.floors)
    if args.reference_rho_min not in floors:
        floors = [args.reference_rho_min, *floors]

    joined = pd.read_csv(args.joined_csv)
    true_keep = _bool_series(joined["true_keep_original_floor"])
    cases = joined[true_keep].copy()
    if args.case_filter_seed >= 0:
        cases = cases[cases["seed"].astype(int) == args.case_filter_seed]
    if args.max_cases > 0:
        cases = cases.head(args.max_cases)
    if cases.empty:
        raise ValueError("No true-keep cases selected for sensitivity perturbation.")

    paper4 = _load_paper4_runner()
    rows: list[dict] = []
    history_rows: list[dict] = []

    print(
        "case_id,rho_min,converged,iters,final_rel_residual,rel_dc_l2_solid,"
        "rel_dc_linf_solid,rel_compliance_change",
        flush=True,
    )

    for preset, preset_cases in cases.groupby("preset", sort=False):
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
        b_norm = float(cp.linalg.norm(F_free_gpu))

        def solve_case(solid_mask: np.ndarray, floor: float) -> dict:
            rho_np = np.where(solid_mask, 1.0, floor).astype(np.float64)
            rho_gpu = cp.asarray(rho_np, dtype=cp.float64)
            E_e = floor + (1.0 - floor) * rho_gpu**args.penal
            setup_start = time.perf_counter()
            gmg.setup(E_e)
            cp.cuda.Stream.null.synchronize()
            setup_time = time.perf_counter() - setup_start

            def A_op(x):
                return mf_op.matvec(x, E_e)

            history: list[float] = []
            solve_start = time.perf_counter()
            x, solve_iters, solver_reported_converged = _cupy_fgmres(
                A_op,
                F_free_gpu,
                gmg.apply,
                tol=args.tol,
                maxiter=args.maxiter,
                restart=args.restart,
                history=history,
            )
            cp.cuda.Stream.null.synchronize()
            solve_time = time.perf_counter() - solve_start
            final_rel_residual = float(cp.linalg.norm(F_free_gpu - A_op(x)) / b_norm)
            U_gpu = cp.zeros(bc.ndof, dtype=cp.float64)
            U_gpu[free_gpu] = x
            Ue = U_gpu[edof_gpu]
            KUe = Ue @ mf_op._KE_unit
            ce = (KUe * Ue).sum(axis=1)
            dc_gpu = -args.penal * (1.0 - floor) * rho_gpu ** (args.penal - 1.0) * ce
            compliance = float(cp.dot(F_free_gpu, x).get())
            return {
                "setup_time_s": setup_time,
                "solve_time_s": solve_time,
                "solve_converged": int(final_rel_residual <= args.tol),
                "solver_reported_converged": int(solver_reported_converged),
                "solve_iters": int(solve_iters),
                "solve_final_rel_residual": final_rel_residual,
                "compliance": compliance,
                "dc": cp.asnumpy(dc_gpu),
                "history": history,
            }

        for case in preset_cases.itertuples(index=False):
            seed = int(case.seed)
            probability = float(case.solid_probability)
            case_id = f"{preset}_s{seed}_q{probability:g}"
            solid_mask = _random_solid_mask(spec, seed, probability)
            reference = None
            solid = solid_mask
            void = ~solid_mask
            for floor in floors:
                result = solve_case(solid_mask, floor)
                if floor == args.reference_rho_min:
                    reference = result
                if reference is None:
                    raise RuntimeError("Reference floor must be solved before raised floors.")

                dc = result["dc"]
                dc_ref = reference["dc"]
                dc_diff = dc - dc_ref
                rel_dc_l2_all = _safe_norm_ratio(dc_diff, dc_ref)
                rel_dc_l2_solid = _safe_norm_ratio(dc_diff[solid], dc_ref[solid])
                rel_dc_l2_void = _safe_norm_ratio(dc_diff[void], dc_ref[void]) if void.any() else float("nan")
                rel_dc_linf_solid = _safe_linf_ratio(dc_diff[solid], dc_ref[solid])
                rel_dc_linf_all = _safe_linf_ratio(dc_diff, dc_ref)
                pearson_solid = _pearson(dc[solid], dc_ref[solid])
                rel_compliance_change = float(
                    (result["compliance"] - reference["compliance"])
                    / max(abs(reference["compliance"]), 1e-300)
                )

                row = {
                    "case_id": case_id,
                    "preset": preset,
                    "seed": seed,
                    "solid_probability": probability,
                    "rho_min": floor,
                    "reference_rho_min": args.reference_rho_min,
                    "penal": args.penal,
                    "n_elem": int(solid_mask.size),
                    "solid_count": int(solid.sum()),
                    "solid_fraction": float(solid.mean()),
                    "sensitivity_convention": "floor_substituted_density_derivative",
                    "setup_time_s": result["setup_time_s"],
                    "solve_time_s": result["solve_time_s"],
                    "solve_converged": result["solve_converged"],
                    "solver_reported_converged": result["solver_reported_converged"],
                    "solve_iters": result["solve_iters"],
                    "solve_final_rel_residual": result["solve_final_rel_residual"],
                    "compliance": result["compliance"],
                    "reference_compliance": reference["compliance"],
                    "rel_compliance_change": rel_compliance_change,
                    "rel_dc_l2_all": rel_dc_l2_all,
                    "rel_dc_l2_solid": rel_dc_l2_solid,
                    "rel_dc_l2_void": rel_dc_l2_void,
                    "rel_dc_linf_all": rel_dc_linf_all,
                    "rel_dc_linf_solid": rel_dc_linf_solid,
                    "pearson_dc_solid": pearson_solid,
                    "history_len": len(result["history"]),
                }
                rows.append(row)
                for i, value in enumerate(result["history"]):
                    history_rows.append(
                        {
                            "case_id": case_id,
                            "preset": preset,
                            "rho_min": floor,
                            "iter": i,
                            "rel_residual": float(value),
                        }
                    )
                print(
                    f"{case_id},{floor:.0e},{result['solve_converged']},"
                    f"{result['solve_iters']},{result['solve_final_rel_residual']:.3e},"
                    f"{rel_dc_l2_solid:.6g},{rel_dc_linf_solid:.6g},"
                    f"{rel_compliance_change:.6g}",
                    flush=True,
                )

    out_dir = Path(args.out_dir)
    summary_fields = [
        "case_id",
        "preset",
        "seed",
        "solid_probability",
        "rho_min",
        "reference_rho_min",
        "penal",
        "n_elem",
        "solid_count",
        "solid_fraction",
        "sensitivity_convention",
        "setup_time_s",
        "solve_time_s",
        "solve_converged",
        "solver_reported_converged",
        "solve_iters",
        "solve_final_rel_residual",
        "compliance",
        "reference_compliance",
        "rel_compliance_change",
        "rel_dc_l2_all",
        "rel_dc_l2_solid",
        "rel_dc_l2_void",
        "rel_dc_linf_all",
        "rel_dc_linf_solid",
        "pearson_dc_solid",
        "history_len",
    ]
    _write_csv(out_dir / "sensitivity_perturbation_summary.csv", rows, summary_fields)
    _write_csv(
        out_dir / "sensitivity_perturbation_history.csv",
        history_rows,
        ["case_id", "preset", "rho_min", "iter", "rel_residual"],
    )
    print(f"Wrote {out_dir / 'sensitivity_perturbation_summary.csv'}", flush=True)
    print(f"Wrote {out_dir / 'sensitivity_perturbation_history.csv'}", flush=True)


if __name__ == "__main__":
    main()
