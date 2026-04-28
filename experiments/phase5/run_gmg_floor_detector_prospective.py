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


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _connectivity(mask: np.ndarray) -> dict:
    nelx, nely, nelz = mask.shape
    comp_id = -np.ones_like(mask, dtype=np.int32)
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    n_components = 0
    largest = 0
    support_to_load = 0
    for start in zip(*np.nonzero(mask)):
        if comp_id[start] >= 0:
            continue
        q = deque([start])
        comp_id[start] = n_components
        size = 0
        touches_support = False
        touches_load = False
        while q:
            x, y, z = q.popleft()
            size += 1
            touches_support = touches_support or x == 0
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
        support_to_load += int(touches_support and touches_load)
        n_components += 1
    solid_count = int(mask.sum())
    return {
        "solid_count": solid_count,
        "solid_fraction": solid_count / mask.size,
        "n_components": n_components,
        "largest_component": largest,
        "largest_fraction_of_solid": largest / max(solid_count, 1),
        "support_to_load_patch_components": support_to_load,
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
    parser.add_argument("--probabilities", default="0.10,0.15,0.18,0.20")
    parser.add_argument("--seeds", default="23")
    parser.add_argument("--baseline-rho-min", type=float, default=1e-12)
    parser.add_argument("--raised-rho-min", type=float, default=1e-3)
    parser.add_argument(
        "--raised-rho-mins",
        default="",
        help="Optional comma-separated floor ladder for predicted-raise cases. "
        "Defaults to --raised-rho-min.",
    )
    parser.add_argument("--penal", default=4.5, type=float)
    parser.add_argument("--probe-iters", default=100, type=int)
    parser.add_argument("--probe-r50", default=50, type=int)
    parser.add_argument("--restart", default=300, type=int)
    parser.add_argument("--maxiter", default=300, type=int)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument("--stack-variant", default="canonical")
    parser.add_argument("--n-levels", default=4, type=int)
    parser.add_argument("--fine-smoother", default="fp64")
    parser.add_argument("--smoother-type", default="chebyshev", choices=["chebyshev", "jacobi"])
    parser.add_argument("--cycle-type", default="v", choices=["v", "w"])
    parser.add_argument(
        "--coarse-correction-policy",
        default="residual_line_search",
        choices=["none", "residual_line_search"],
    )
    parser.add_argument(
        "--root-local-correction-mode",
        default="node_block_inner_fgmres",
        choices=["none", "node_block_line_search", "node_block_inner_fgmres"],
    )
    parser.add_argument("--root-local-correction-inner-steps", default=20, type=int)
    parser.add_argument("--root-local-correction-inner-restart", default=20, type=int)
    parser.add_argument("--inner-krylov-steps", default=10, type=int)
    parser.add_argument("--inner-krylov-restart", default=10, type=int)
    parser.add_argument("--high-residual-threshold", type=float, default=1e-2)
    parser.add_argument("--plateau-residual-threshold", type=float, default=1e-4)
    parser.add_argument("--plateau-ratio-threshold", type=float, default=0.6)
    parser.add_argument(
        "--severity-jump-r50-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional r50 threshold for a severity-aware baseline. When positive, "
            "predicted-raise cases with r50 at or above this value skip rescue "
            "floors below --severity-jump-rho-min."
        ),
    )
    parser.add_argument(
        "--severity-jump-rho-min",
        type=float,
        default=1e-2,
        help="First floor to try when the optional severity jump triggers.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results" / "gmg_solver_floor_detector_prospective"),
    )
    args = parser.parse_args()

    import cupy as cp
    from gpu_fem.multigrid_v4 import _cupy_fgmres

    paper4 = _load_paper4_runner()
    probabilities = _parse_float_list(args.probabilities)
    seeds = _parse_int_list(args.seeds)
    raised_rho_mins = (
        _parse_float_list(args.raised_rho_mins)
        if args.raised_rho_mins.strip()
        else [args.raised_rho_min]
    )
    out_dir = Path(args.out_dir)

    spec, bc, free, free_gpu, edof_gpu, F_free_gpu, mf_op, gmg = paper4._build_components(
        args.preset,
        n_levels=args.n_levels,
        fine_smoother=args.fine_smoother,
        smoother_type=args.smoother_type,
        cycle_type=args.cycle_type,
        coarse_correction_policy=args.coarse_correction_policy,
        root_local_correction_mode=args.root_local_correction_mode,
        root_local_correction_inner_steps=args.root_local_correction_inner_steps,
        root_local_correction_inner_restart=args.root_local_correction_inner_restart,
        inner_krylov_steps=args.inner_krylov_steps,
        inner_krylov_restart=args.inner_krylov_restart,
    )
    b_norm = float(cp.linalg.norm(F_free_gpu))
    summary_rows = []
    history_rows = []
    print(
        "seed,p,r50,r100,trigger,recommended_rho_min,solve_converged,solver_reported_converged,"
        "solve_iters,solve_final_rel_residual,attempted_rho_mins",
        flush=True,
    )

    for seed in seeds:
        random_field = np.random.default_rng(seed).random(spec.nelx * spec.nely * spec.nelz)
        random_field = random_field.reshape(spec.nelx, spec.nely, spec.nelz)
        for probability in probabilities:
            solid = random_field < probability
            conn = _connectivity(solid)

            def solve_at_floor(rho_min: float, *, maxiter: int, history: list[float]):
                rho = np.where(solid.reshape(-1), 1.0, rho_min)
                E_e = rho_min + (1.0 - rho_min) * cp.asarray(rho, dtype=cp.float64) ** args.penal
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
                return E_e, iters, solver_reported_converged, true_converged, final, metrics

            probe_history: list[float] = []
            _, _, _, _, _, probe_metrics = solve_at_floor(
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
            severity_jump_triggered = (
                raise_floor
                and args.severity_jump_r50_threshold > 0.0
                and r50 >= args.severity_jump_r50_threshold
            )
            if raise_floor and severity_jump_triggered:
                floor_ladder = [
                    value for value in raised_rho_mins if value >= args.severity_jump_rho_min
                ]
                if not floor_ladder:
                    floor_ladder = [args.severity_jump_rho_min]
            else:
                floor_ladder = raised_rho_mins if raise_floor else [args.baseline_rho_min]
            attempted_rho_mins = []
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
            solve_history: list[tuple[float, list[float]]] = []

            def run_floor_sequence(sequence: list[float]) -> None:
                nonlocal recommended_rho_min
                nonlocal solve_iters
                nonlocal solve_converged
                nonlocal solver_reported_converged
                nonlocal solve_final
                nonlocal selected_metrics
                nonlocal failed_ladder_setup_time_s
                nonlocal failed_ladder_solve_time_s
                for candidate_rho_min in sequence:
                    if candidate_rho_min in attempted_rho_mins:
                        continue
                    attempted_rho_mins.append(candidate_rho_min)
                    candidate_history: list[float] = []
                    (
                        _,
                        candidate_iters,
                        candidate_solver_reported_converged,
                        candidate_true_converged,
                        candidate_final,
                        candidate_metrics,
                    ) = solve_at_floor(
                        candidate_rho_min,
                        maxiter=args.maxiter,
                        history=candidate_history,
                    )
                    solve_history.append((candidate_rho_min, candidate_history))
                    recommended_rho_min = candidate_rho_min
                    solve_iters = candidate_iters
                    solve_converged = candidate_true_converged
                    solver_reported_converged = candidate_solver_reported_converged
                    solve_final = candidate_final
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
                "seed": seed,
                "solid_probability": probability,
                "baseline_rho_min": args.baseline_rho_min,
                "raised_rho_min": args.raised_rho_min,
                "raised_rho_mins": ";".join(f"{value:.0e}" for value in raised_rho_mins),
                "recommended_rho_min": recommended_rho_min,
                "attempted_rho_mins": ";".join(f"{value:.0e}" for value in attempted_rho_mins),
                "n_floor_attempts": len(attempted_rho_mins),
                "penal": args.penal,
                "probe_iters": args.probe_iters,
                "restart": args.restart,
                "maxiter": args.maxiter,
                "tol": args.tol,
                "stack_variant": args.stack_variant,
                "n_levels": args.n_levels,
                "fine_smoother": args.fine_smoother,
                "smoother_type": args.smoother_type,
                "cycle_type": args.cycle_type,
                "coarse_correction_policy": args.coarse_correction_policy,
                "root_local_correction_mode": args.root_local_correction_mode,
                "root_local_correction_inner_steps": args.root_local_correction_inner_steps,
                "root_local_correction_inner_restart": args.root_local_correction_inner_restart,
                "inner_krylov_steps": args.inner_krylov_steps,
                "inner_krylov_restart": args.inner_krylov_restart,
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
                "severity_jump_triggered": int(severity_jump_triggered),
                "severity_jump_r50_threshold": args.severity_jump_r50_threshold,
                "severity_jump_rho_min": args.severity_jump_rho_min,
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
                        "seed": seed,
                        "solid_probability": probability,
                        "phase": "probe",
                        "rho_min": args.baseline_rho_min,
                        "iter": i,
                        "rel_residual": float(value),
                    }
                )
            for candidate_rho_min, candidate_history in solve_history:
                for i, value in enumerate(candidate_history):
                    history_rows.append(
                        {
                            "seed": seed,
                            "solid_probability": probability,
                            "phase": "recommended_solve",
                            "rho_min": candidate_rho_min,
                            "iter": i,
                            "rel_residual": float(value),
                        }
                    )
            print(
                f"{seed},{probability:.4g},{r50:.12g},{r100:.12g},{trigger},"
                f"{recommended_rho_min:.0e},{int(solve_converged)},{int(solver_reported_converged)},"
                f"{solve_iters},{solve_final:.12g},"
                f"{';'.join(f'{value:.0e}' for value in attempted_rho_mins)}",
                flush=True,
            )

    summary_fields = [
        "preset",
        "seed",
        "solid_probability",
        "baseline_rho_min",
        "raised_rho_min",
        "raised_rho_mins",
        "recommended_rho_min",
        "attempted_rho_mins",
        "n_floor_attempts",
        "penal",
        "probe_iters",
        "restart",
        "maxiter",
        "tol",
        "stack_variant",
        "n_levels",
        "fine_smoother",
        "smoother_type",
        "cycle_type",
        "coarse_correction_policy",
        "root_local_correction_mode",
        "root_local_correction_inner_steps",
        "root_local_correction_inner_restart",
        "inner_krylov_steps",
        "inner_krylov_restart",
        "solid_count",
        "solid_fraction",
        "n_components",
        "largest_component",
        "largest_fraction_of_solid",
        "support_to_load_patch_components",
                "probe_r50",
                "probe_r100",
                "probe_r100_over_r50",
                "probe_setup_time_s",
                "probe_solve_time_s",
                "probe_gpu_used_mb_after_solve",
                "probe_cupy_pool_used_mb_after_solve",
                "trigger",
                "predicted_raise_floor",
                "severity_jump_triggered",
                "severity_jump_r50_threshold",
                "severity_jump_rho_min",
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
    _write_csv(out_dir / "prospective_summary.csv", summary_rows, summary_fields)
    _write_csv(
        out_dir / "prospective_history.csv",
        history_rows,
        ["seed", "solid_probability", "phase", "rho_min", "iter", "rel_residual"],
    )
    print(f"Wrote {out_dir / 'prospective_summary.csv'}", flush=True)
    print(f"Wrote {out_dir / 'prospective_history.csv'}", flush=True)


if __name__ == "__main__":
    main()
