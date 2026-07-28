from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
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


def _parse_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _connectivity(mask: np.ndarray) -> dict:
    nelx, nely, nelz = mask.shape
    comp_id = -np.ones_like(mask, dtype=np.int32)
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    n_components = 0
    largest = 0
    for start in zip(*np.nonzero(mask)):
        if comp_id[start] >= 0:
            continue
        q = deque([start])
        comp_id[start] = n_components
        size = 0
        while q:
            x, y, z = q.popleft()
            size += 1
            for dx, dy, dz in neighbors:
                xn, yn, zn = x + dx, y + dy, z + dz
                if 0 <= xn < nelx and 0 <= yn < nely and 0 <= zn < nelz:
                    if mask[xn, yn, zn] and comp_id[xn, yn, zn] < 0:
                        comp_id[xn, yn, zn] = n_components
                        q.append((xn, yn, zn))
        largest = max(largest, size)
        n_components += 1
    solid_count = int(mask.sum())
    return {
        "solid_count": solid_count,
        "solid_fraction": solid_count / mask.size,
        "n_components": n_components,
        "largest_component": largest,
        "largest_fraction_of_solid": largest / max(solid_count, 1),
    }


def _classify(
    r50: float,
    r100: float,
    *,
    high_residual_threshold: float,
    plateau_residual_threshold: float,
    plateau_ratio_threshold: float,
) -> tuple[bool, str]:
    if r50 >= high_residual_threshold:
        return True, "high_r50"
    ratio = r100 / max(r50, 1e-300)
    if r100 >= plateau_residual_threshold and ratio >= plateau_ratio_threshold:
        return True, "plateau"
    return False, "keep"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _memory_snapshot(cp) -> dict:
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    pinned = cp.get_default_pinned_memory_pool()
    return {
        "gpu_total_mb": total_b / 1e6,
        "gpu_free_mb": free_b / 1e6,
        "gpu_used_mb": (total_b - free_b) / 1e6,
        "cupy_pool_used_mb": pool.used_bytes() / 1e6,
        "cupy_pool_total_mb": pool.total_bytes() / 1e6,
        "cupy_pinned_free_blocks": pinned.n_free_blocks(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="cantilever_gpu_medium")
    parser.add_argument(
        "--density-paths",
        required=True,
        help="Comma-separated .npy density fields with one value per element.",
    )
    parser.add_argument("--baseline-rho-min", type=float, default=1e-12)
    parser.add_argument("--raised-rho-mins", default="1e-3,1e-2")
    parser.add_argument("--penal", default=4.5, type=float)
    parser.add_argument("--solid-threshold", default=0.5, type=float)
    parser.add_argument(
        "--fine-smoother",
        default="fp64",
        choices=["fp64", "fp32", "bf16"],
        help="Precision of the finest-level smoother inside the multigrid preconditioner. "
             "Coarse levels, the outer Krylov operator, and the acceptance residual stay FP64.",
    )
    parser.add_argument("--probe-iters", default=100, type=int)
    parser.add_argument("--probe-r50", default=50, type=int)
    parser.add_argument("--restart", default=300, type=int)
    parser.add_argument("--maxiter", default=300, type=int)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument("--high-residual-threshold", type=float, default=1e-2)
    parser.add_argument("--plateau-residual-threshold", type=float, default=1e-4)
    parser.add_argument("--plateau-ratio-threshold", type=float, default=0.6)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results" / "gmg_solver_floor_detector_density"),
    )
    args = parser.parse_args()

    import cupy as cp
    from gpu_fem.multigrid_v4 import _cupy_fgmres

    paper4 = _load_paper4_runner()
    density_paths = _parse_paths(args.density_paths)
    raised_rho_mins = _parse_float_list(args.raised_rho_mins)
    out_dir = Path(args.out_dir)

    spec, bc, free, free_gpu, edof_gpu, F_free_gpu, mf_op, gmg = paper4._build_components(
        args.preset,
        n_levels=4,
        fine_smoother=args.fine_smoother,
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
    summary_rows = []
    history_rows = []
    print(
        "density,r50,r100,trigger,recommended_rho_min,solve_converged,solver_reported_converged,"
        "solve_iters,solve_final_rel_residual",
        flush=True,
    )

    for density_path in density_paths:
        density = np.asarray(np.load(density_path), dtype=np.float64).reshape(-1)
        if density.size != n_elem:
            raise ValueError(
                f"{density_path} has {density.size} entries, expected {n_elem} for preset {args.preset}"
            )
        density = np.clip(density, 0.0, 1.0)
        mask = density.reshape(spec.nelx, spec.nely, spec.nelz) >= args.solid_threshold
        conn = _connectivity(mask)

        def solve_at_floor(rho_min: float, *, maxiter: int, history: list[float]):
            E_e = rho_min + (1.0 - rho_min) * cp.asarray(density, dtype=cp.float64) ** args.penal
            cp.cuda.Stream.null.synchronize()
            mem_before = _memory_snapshot(cp)
            setup_start = time.perf_counter()
            gmg.setup(E_e)
            cp.cuda.Stream.null.synchronize()
            setup_time = time.perf_counter() - setup_start
            mem_after_setup = _memory_snapshot(cp)

            def A_op(x):
                return mf_op.matvec(x, E_e)

            solve_start = time.perf_counter()
            x, iters, solver_reported_converged = _cupy_fgmres(
                A_op,
                F_free_gpu,
                gmg.apply,
                tol=args.tol,
                maxiter=maxiter,
                restart=args.restart,
                history=history,
            )
            cp.cuda.Stream.null.synchronize()
            solve_time = time.perf_counter() - solve_start
            final = float(cp.linalg.norm(F_free_gpu - A_op(x)) / b_norm)
            true_converged = final <= args.tol
            cp.cuda.Stream.null.synchronize()
            mem_after_solve = _memory_snapshot(cp)
            metrics = {
                "setup_time_s": setup_time,
                "solve_time_s": solve_time,
                "gpu_used_mb_before": mem_before["gpu_used_mb"],
                "gpu_used_mb_after_setup": mem_after_setup["gpu_used_mb"],
                "gpu_used_mb_after_solve": mem_after_solve["gpu_used_mb"],
                "cupy_pool_used_mb_before": mem_before["cupy_pool_used_mb"],
                "cupy_pool_used_mb_after_setup": mem_after_setup["cupy_pool_used_mb"],
                "cupy_pool_used_mb_after_solve": mem_after_solve["cupy_pool_used_mb"],
                "cupy_pool_total_mb_after_solve": mem_after_solve["cupy_pool_total_mb"],
            }
            return iters, solver_reported_converged, true_converged, final, metrics

        probe_history: list[float] = []
        _, _, _, _, probe_metrics = solve_at_floor(
            args.baseline_rho_min,
            maxiter=args.probe_iters,
            history=probe_history,
        )
        r50 = probe_history[min(args.probe_r50, len(probe_history) - 1)]
        r100 = probe_history[min(args.probe_iters, len(probe_history) - 1)]
        raise_floor, trigger = _classify(
            r50,
            r100,
            high_residual_threshold=args.high_residual_threshold,
            plateau_residual_threshold=args.plateau_residual_threshold,
            plateau_ratio_threshold=args.plateau_ratio_threshold,
        )

        floor_ladder = raised_rho_mins if raise_floor else [args.baseline_rho_min]
        attempted = []
        recommended_rho_min = floor_ladder[-1]
        solve_iters = args.maxiter
        solve_converged = False
        solver_reported_converged = False
        solve_final = float("nan")
        post_solve_fallback_triggered = False
        selected_metrics = {
            "setup_time_s": float("nan"),
            "solve_time_s": float("nan"),
            "gpu_used_mb_before": float("nan"),
            "gpu_used_mb_after_setup": float("nan"),
            "gpu_used_mb_after_solve": float("nan"),
            "cupy_pool_used_mb_before": float("nan"),
            "cupy_pool_used_mb_after_setup": float("nan"),
            "cupy_pool_used_mb_after_solve": float("nan"),
            "cupy_pool_total_mb_after_solve": float("nan"),
        }
        failed_ladder_setup_time_s = 0.0
        failed_ladder_solve_time_s = 0.0
        solve_histories: list[tuple[float, list[float]]] = []

        def run_floor_sequence(sequence: list[float]) -> None:
            nonlocal recommended_rho_min
            nonlocal solve_iters
            nonlocal solve_converged
            nonlocal solver_reported_converged
            nonlocal solve_final
            nonlocal selected_metrics
            nonlocal failed_ladder_setup_time_s
            nonlocal failed_ladder_solve_time_s
            for rho_min in sequence:
                if rho_min in attempted:
                    continue
                attempted.append(rho_min)
                solve_history: list[float] = []
                (
                    solve_iters,
                    candidate_solver_reported_converged,
                    candidate_true_converged,
                    solve_final,
                    candidate_metrics,
                ) = solve_at_floor(
                    rho_min,
                    maxiter=args.maxiter,
                    history=solve_history,
                )
                solve_histories.append((rho_min, solve_history))
                recommended_rho_min = rho_min
                solve_converged = candidate_true_converged
                solver_reported_converged = candidate_solver_reported_converged
                selected_metrics = candidate_metrics
                if candidate_true_converged:
                    break
                failed_ladder_setup_time_s += candidate_metrics["setup_time_s"]
                failed_ladder_solve_time_s += candidate_metrics["solve_time_s"]

        run_floor_sequence(floor_ladder)
        if (not solve_converged) and (not raise_floor):
            post_solve_fallback_triggered = True
            run_floor_sequence(raised_rho_mins)

        row = {
            "preset": args.preset,
            "density_path": str(density_path),
            "density_name": density_path.parent.name,
            "n_elem": n_elem,
            "density_min": float(np.min(density)),
            "density_max": float(np.max(density)),
            "density_mean": float(np.mean(density)),
            "density_grayness": float(np.mean(4.0 * density * (1.0 - density))),
            "baseline_rho_min": args.baseline_rho_min,
            "raised_rho_mins": ";".join(f"{value:.0e}" for value in raised_rho_mins),
            "recommended_rho_min": recommended_rho_min,
            "attempted_rho_mins": ";".join(f"{value:.0e}" for value in attempted),
            "n_floor_attempts": len(attempted),
            "penal": args.penal,
            "fine_smoother": args.fine_smoother,
            "probe_iters": args.probe_iters,
            "restart": args.restart,
            "maxiter": args.maxiter,
            "tol": args.tol,
            **conn,
            "probe_r50": r50,
            "probe_r100": r100,
            "probe_r100_over_r50": r100 / max(r50, 1e-300),
            "probe_setup_time_s": probe_metrics["setup_time_s"],
            "probe_solve_time_s": probe_metrics["solve_time_s"],
            "probe_gpu_used_mb_after_solve": probe_metrics["gpu_used_mb_after_solve"],
            "probe_cupy_pool_used_mb_after_solve": probe_metrics["cupy_pool_used_mb_after_solve"],
            "trigger": trigger,
            "predicted_raise_floor": int(raise_floor),
            "post_solve_fallback_triggered": int(post_solve_fallback_triggered),
            "detector_false_keep": int((not raise_floor) and post_solve_fallback_triggered),
            "solve_converged": int(solve_converged),
            "solver_reported_converged": int(solver_reported_converged),
            "solve_iters": solve_iters,
            "solve_final_rel_residual": solve_final,
            "selected_setup_time_s": selected_metrics["setup_time_s"],
            "selected_solve_time_s": selected_metrics["solve_time_s"],
            "selected_gpu_used_mb_after_solve": selected_metrics["gpu_used_mb_after_solve"],
            "selected_cupy_pool_used_mb_after_solve": selected_metrics["cupy_pool_used_mb_after_solve"],
            "selected_cupy_pool_total_mb_after_solve": selected_metrics["cupy_pool_total_mb_after_solve"],
            "failed_ladder_setup_time_s": failed_ladder_setup_time_s,
            "failed_ladder_solve_time_s": failed_ladder_solve_time_s,
            "recorded_policy_time_s": (
                probe_metrics["setup_time_s"]
                + probe_metrics["solve_time_s"]
                + failed_ladder_setup_time_s
                + failed_ladder_solve_time_s
                + selected_metrics["setup_time_s"]
                + selected_metrics["solve_time_s"]
            ),
        }
        summary_rows.append(row)

        for i, value in enumerate(probe_history):
            history_rows.append(
                {
                    "density_name": row["density_name"],
                    "density_path": str(density_path),
                    "phase": "probe",
                    "rho_min": args.baseline_rho_min,
                    "iter": i,
                    "rel_residual": float(value),
                }
            )
        for rho_min, solve_history in solve_histories:
            for i, value in enumerate(solve_history):
                history_rows.append(
                    {
                        "density_name": row["density_name"],
                        "density_path": str(density_path),
                        "phase": "recommended_solve",
                        "rho_min": rho_min,
                        "iter": i,
                        "rel_residual": float(value),
                    }
                )
        print(
            f"{density_path},{r50:.12g},{r100:.12g},{trigger},{recommended_rho_min:.0e},"
            f"{int(solve_converged)},{int(solver_reported_converged)},{solve_iters},{solve_final:.12g}",
            flush=True,
        )

    fieldnames = [
        "preset",
        "density_path",
        "density_name",
        "n_elem",
        "density_min",
        "density_max",
        "density_mean",
        "density_grayness",
        "baseline_rho_min",
        "raised_rho_mins",
        "recommended_rho_min",
        "attempted_rho_mins",
        "n_floor_attempts",
        "penal",
        "fine_smoother",
        "probe_iters",
        "restart",
        "maxiter",
        "tol",
        "solid_count",
        "solid_fraction",
        "n_components",
        "largest_component",
        "largest_fraction_of_solid",
        "probe_r50",
        "probe_r100",
        "probe_r100_over_r50",
        "probe_setup_time_s",
        "probe_solve_time_s",
        "probe_gpu_used_mb_after_solve",
        "probe_cupy_pool_used_mb_after_solve",
        "trigger",
        "predicted_raise_floor",
        "post_solve_fallback_triggered",
        "detector_false_keep",
        "solve_converged",
        "solver_reported_converged",
        "solve_iters",
        "solve_final_rel_residual",
        "selected_setup_time_s",
        "selected_solve_time_s",
        "selected_gpu_used_mb_after_solve",
        "selected_cupy_pool_used_mb_after_solve",
        "selected_cupy_pool_total_mb_after_solve",
        "failed_ladder_setup_time_s",
        "failed_ladder_solve_time_s",
        "recorded_policy_time_s",
    ]
    _write_csv(out_dir / "density_detector_summary.csv", summary_rows, fieldnames)
    _write_csv(
        out_dir / "density_detector_history.csv",
        history_rows,
        ["density_name", "density_path", "phase", "rho_min", "iter", "rel_residual"],
    )
    print(f"Wrote {out_dir / 'density_detector_summary.csv'}", flush=True)
    print(f"Wrote {out_dir / 'density_detector_history.csv'}", flush=True)


if __name__ == "__main__":
    main()
