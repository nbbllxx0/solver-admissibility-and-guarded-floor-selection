from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load_paper4_runner():
    path = ROOT / "experiments" / "paper4" / "run_experiments_e1_e10.py"
    spec = importlib.util.spec_from_file_location("paper4_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_floors(text: str) -> list[float]:
    return [float(item.strip()) for item in text.replace(",", ";").split(";") if item.strip()]


def _density_for_case(row: dict, spec) -> np.ndarray:
    n_elem = spec.nelx * spec.nely * spec.nelz
    if row["case_type"] == "density":
        density = np.asarray(np.load(ROOT / row["density_path"]), dtype=np.float64).reshape(-1)
        if density.size != n_elem:
            raise ValueError(f"{row['case_id']} has {density.size} entries, expected {n_elem}")
        return np.clip(density, 0.0, 1.0)
    if row["case_type"] == "random":
        seed = int(row["seed"])
        probability = float(row["solid_probability"])
        field = np.random.default_rng(seed).random(n_elem)
        return (field < probability).astype(np.float64)
    raise ValueError(f"unknown case_type {row['case_type']!r}")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "experiments" / "phase5" / "fixed_floor_control_manifest.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "experiments" / "phase5" / "results" / "gmg_fixed_floor_controls"))
    parser.add_argument("--restart", type=int, default=300)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    import cupy as cp
    from gpu_fem.multigrid_v4 import _cupy_fgmres

    paper4 = _load_paper4_runner()
    rows = _read_manifest(Path(args.manifest))
    by_preset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_preset[row["preset"]].append(row)

    summary_rows = []
    history_rows = []
    out_dir = Path(args.out_dir)

    for preset, preset_rows in by_preset.items():
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

        for row in preset_rows:
            density = _density_for_case(row, spec)
            density_gpu = cp.asarray(density, dtype=cp.float64)
            penal = float(row["penal"])
            floors = _parse_floors(row["floors"])
            for floor in floors:
                E_e = floor + (1.0 - floor) * density_gpu**penal
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
                residual = F_free_gpu - A_op(x)
                final_rel_residual = float(cp.linalg.norm(residual) / b_norm)
                solve_converged = final_rel_residual <= args.tol
                compliance = float(cp.dot(F_free_gpu, x).get())
                density_mean = float(np.mean(density))
                density_grayness = float(np.mean(4.0 * density * (1.0 - density)))
                summary = {
                    "case_id": row["case_id"],
                    "case_type": row["case_type"],
                    "preset": preset,
                    "seed": row["seed"],
                    "solid_probability": row["solid_probability"],
                    "density_path": row["density_path"],
                    "n_elem": density.size,
                    "rho_min": floor,
                    "penal": penal,
                    "density_mean": density_mean,
                    "density_grayness": density_grayness,
                    "restart": args.restart,
                    "maxiter": args.maxiter,
                    "tol": args.tol,
                    "setup_time_s": setup_time,
                    "solve_time_s": solve_time,
                    "solve_converged": int(solve_converged),
                    "solver_reported_converged": int(solver_reported_converged),
                    "solve_iters": solve_iters,
                    "solve_final_rel_residual": final_rel_residual,
                    "compliance": compliance,
                    "history_len": len(history),
                }
                summary_rows.append(summary)
                for i, value in enumerate(history):
                    history_rows.append(
                        {
                            "case_id": row["case_id"],
                            "preset": preset,
                            "rho_min": floor,
                            "iter": i,
                            "rel_residual": float(value),
                        }
                    )
                print(
                    f"{row['case_id']},{preset},rho={floor:.0e},conv={int(solve_converged)},"
                    f"iters={solve_iters},res={final_rel_residual:.3e},solve_s={solve_time:.2f}",
                    flush=True,
                )

    summary_fields = [
        "case_id",
        "case_type",
        "preset",
        "seed",
        "solid_probability",
        "density_path",
        "n_elem",
        "rho_min",
        "penal",
        "density_mean",
        "density_grayness",
        "restart",
        "maxiter",
        "tol",
        "setup_time_s",
        "solve_time_s",
        "solve_converged",
        "solver_reported_converged",
        "solve_iters",
        "solve_final_rel_residual",
        "compliance",
        "history_len",
    ]
    _write_csv(out_dir / "fixed_floor_control_summary.csv", summary_rows, summary_fields)
    _write_csv(
        out_dir / "fixed_floor_control_history.csv",
        history_rows,
        ["case_id", "preset", "rho_min", "iter", "rel_residual"],
    )
    print(f"Wrote {out_dir / 'fixed_floor_control_summary.csv'}")
    print(f"Wrote {out_dir / 'fixed_floor_control_history.csv'}")


if __name__ == "__main__":
    main()
