from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import deque
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


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _connectivity(mask: np.ndarray) -> dict:
    nelx, nely, nelz = mask.shape
    comp_id = -np.ones_like(mask, dtype=np.int32)
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    n_components = 0
    largest = 0
    support_touching = 0
    right_touching = 0
    load_touching = 0
    support_to_right = 0
    support_to_load = 0
    for start in zip(*np.nonzero(mask)):
        if comp_id[start] >= 0:
            continue
        q = deque([start])
        comp_id[start] = n_components
        size = 0
        touches_support = False
        touches_right = False
        touches_load = False
        while q:
            x, y, z = q.popleft()
            size += 1
            touches_support = touches_support or x == 0
            touches_right = touches_right or x == nelx - 1
            touches_load = touches_load or (
                x == nelx - 1 and abs(y - nely // 2) <= 1 and abs(z - nelz // 2) <= 1
            )
            for dx, dy, dz in neighbors:
                xn, yn, zn = x + dx, y + dy, z + dz
                if 0 <= xn < nelx and 0 <= yn < nely and 0 <= zn < nelz:
                    if mask[xn, yn, zn] and comp_id[xn, yn, zn] < 0:
                        comp_id[xn, yn, zn] = n_components
                        q.append((xn, yn, zn))
        largest = max(largest, size)
        support_touching += int(touches_support)
        right_touching += int(touches_right)
        load_touching += int(touches_load)
        support_to_right += int(touches_support and touches_right)
        support_to_load += int(touches_support and touches_load)
        n_components += 1
    solid_count = int(mask.sum())
    return {
        "solid_count": solid_count,
        "solid_fraction": solid_count / mask.size,
        "n_components": n_components,
        "largest_component": largest,
        "largest_fraction_of_solid": largest / max(solid_count, 1),
        "support_touching_components": support_touching,
        "right_face_touching_components": right_touching,
        "load_patch_touching_components": load_touching,
        "support_to_right_components": support_to_right,
        "support_to_load_patch_components": support_to_load,
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="cantilever_gpu_medium")
    parser.add_argument("--probabilities", default="0.10,0.12,0.15,0.18,0.20,0.35,0.50")
    parser.add_argument("--seeds", default="19")
    parser.add_argument("--rho-min", default=1e-12, type=float)
    parser.add_argument("--penal", default=4.5, type=float)
    parser.add_argument("--restart", default=200, type=int)
    parser.add_argument("--maxiter", default=500, type=int)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "experiments" / "phase5" / "results" / "percolation_atlas"))
    args = parser.parse_args()

    import cupy as cp
    from gpu_fem.multigrid_v4 import _cupy_fgmres

    paper4 = _load_paper4_runner()
    probabilities = _parse_float_list(args.probabilities)
    seeds = _parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)

    spec, bc, free, free_gpu, edof_gpu, F_free_gpu, mf_op, gmg = paper4._build_components(
        args.preset,
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
    summary_rows = []
    history_rows = []
    print(
        "seed,solid_probability,solid_fraction,n_components,largest_fraction_of_solid,"
        "support_to_load_patch_components,solve_converged,solve_iters,solve_final_rel_residual",
        flush=True,
    )
    for seed in seeds:
        random_field = np.random.default_rng(seed).random(spec.nelx * spec.nely * spec.nelz)
        random_field = random_field.reshape(spec.nelx, spec.nely, spec.nelz)
        for probability in probabilities:
            solid = random_field < probability
            conn = _connectivity(solid)
            rho = np.where(solid.reshape(-1), 1.0, args.rho_min)
            E_e = args.rho_min + (1.0 - args.rho_min) * cp.asarray(rho, dtype=cp.float64) ** args.penal
            gmg.setup(E_e)

            solve_converged = None
            solve_iters = None
            solve_final = float("nan")
            history = []
            if not args.skip_solve:
                def A_op(x):
                    return mf_op.matvec(x, E_e)

                x, solve_iters, solve_converged = _cupy_fgmres(
                    A_op,
                    F_free_gpu,
                    gmg.apply,
                    tol=args.tol,
                    maxiter=args.maxiter,
                    restart=args.restart,
                    history=history,
                )
                solve_final = float(cp.linalg.norm(F_free_gpu - A_op(x)) / b_norm)

            row = {
                "preset": args.preset,
                "seed": seed,
                "solid_probability": probability,
                "rho_min": args.rho_min,
                "penal": args.penal,
                "restart": args.restart,
                "maxiter": args.maxiter,
                "tol": args.tol,
                **conn,
                "solve_converged": int(solve_converged) if solve_converged is not None else "",
                "solve_iters": solve_iters if solve_iters is not None else "",
                "solve_final_rel_residual": solve_final,
                "solve_history_len": len(history),
            }
            summary_rows.append(row)
            for i, value in enumerate(history):
                history_rows.append(
                    {
                        "seed": seed,
                        "solid_probability": probability,
                        "iter": i,
                        "rel_residual": float(value),
                    }
                )
            print(
                f"{seed},{probability:.4g},{conn['solid_fraction']:.12g},{conn['n_components']},"
                f"{conn['largest_fraction_of_solid']:.12g},{conn['support_to_load_patch_components']},"
                f"{row['solve_converged']},{row['solve_iters']},{solve_final:.12g}",
                flush=True,
            )

    summary_fields = [
        "preset",
        "seed",
        "solid_probability",
        "rho_min",
        "penal",
        "restart",
        "maxiter",
        "tol",
        "solid_count",
        "solid_fraction",
        "n_components",
        "largest_component",
        "largest_fraction_of_solid",
        "support_touching_components",
        "right_face_touching_components",
        "load_patch_touching_components",
        "support_to_right_components",
        "support_to_load_patch_components",
        "solve_converged",
        "solve_iters",
        "solve_final_rel_residual",
        "solve_history_len",
    ]
    _write_csv(out_dir / "atlas_summary.csv", summary_rows, summary_fields)
    _write_csv(out_dir / "atlas_history.csv", history_rows, ["seed", "solid_probability", "iter", "rel_residual"])
    print(f"Wrote {out_dir / 'atlas_summary.csv'}", flush=True)
    print(f"Wrote {out_dir / 'atlas_history.csv'}", flush=True)


if __name__ == "__main__":
    main()
