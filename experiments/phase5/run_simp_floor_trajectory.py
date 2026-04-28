from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _configure_runtime_tmp() -> None:
    root = Path(__file__).resolve().parents[2]
    stamp = f"traj_{os.getpid()}_{int(time.time() * 1000)}"
    runtime_tmp = root / "tmp" / "phase5_trajectory_runtime" / stamp
    cupy_cache = root / "tmp" / "phase5_trajectory_cupy_cache" / stamp
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    cupy_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(runtime_tmp)
    os.environ["TEMP"] = str(runtime_tmp)
    os.environ["CUPY_CACHE_DIR"] = str(cupy_cache)
    os.environ["CUPY_TEMPDIR"] = str(runtime_tmp)
    tempfile.tempdir = str(runtime_tmp)


_configure_runtime_tmp()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpu_fem.bc_generator import generate_bc
from gpu_fem.local_agents import PureFEMRouter
from gpu_fem.presets import get_preset
from gpu_fem.pub_baseline_controller import ScheduleOnlyController
from gpu_fem.simp_gpu import TO3DParams, run_simp_surrogate_gpu
from gpu_fem.solver_v4 import SolverV4


def _history_at(history: list[float], iteration: int) -> float:
    if not history:
        return float("inf")
    index = min(max(iteration, 0), len(history) - 1)
    return float(history[index])


def _classify_probe(
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


class GuardedAdaptiveFloorSolverV4(SolverV4):
    """SolverV4 wrapper that applies the Phase 5 guarded floor policy per SIMP iteration."""

    def __init__(
        self,
        *args,
        guarded_base_rho_min: float = 1e-12,
        guarded_ladder: tuple[float, ...] = (1e-3, 1e-2),
        guarded_probe_iters: int = 100,
        guarded_probe_r50: int = 50,
        guarded_high_residual_threshold: float = 1e-2,
        guarded_plateau_residual_threshold: float = 1e-4,
        guarded_plateau_ratio_threshold: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.guarded_base_rho_min = float(guarded_base_rho_min)
        self.guarded_ladder = tuple(float(v) for v in guarded_ladder)
        self.guarded_probe_iters = int(guarded_probe_iters)
        self.guarded_probe_r50 = int(guarded_probe_r50)
        self.guarded_high_residual_threshold = float(guarded_high_residual_threshold)
        self.guarded_plateau_residual_threshold = float(guarded_plateau_residual_threshold)
        self.guarded_plateau_ratio_threshold = float(guarded_plateau_ratio_threshold)
        self.policy_events: list[dict] = []
        self.last_selected_rho_min = self.guarded_base_rho_min
        self.last_policy_trigger = ""
        self.last_policy_fallback_used = 0
        self.last_policy_attempted_rho_mins = ""
        self.last_policy_probe_r50 = float("nan")
        self.last_policy_probe_r100 = float("nan")
        self.last_policy_probe_true_rel_residual = float("nan")

    def _solve_with_floor(
        self,
        rho_phys: np.ndarray,
        penal: float,
        *,
        rho_min: float,
        maxiter: int,
        phase: str,
    ) -> tuple[float, np.ndarray, dict]:
        old_emin = self.Emin
        old_maxiter = self.cg_maxiter
        self.Emin = float(rho_min)
        self.cg_maxiter = int(maxiter)
        try:
            start = time.perf_counter()
            compliance, dc_phys = super().solve(rho_phys, penal)
            elapsed = time.perf_counter() - start
            metrics = {
                "phase": phase,
                "rho_min": float(rho_min),
                "maxiter": int(maxiter),
                "compliance": float(compliance),
                "iters": int(getattr(self, "last_cg_iters", -1)),
                "true_rel_residual": float(getattr(self, "last_true_rel_residual", float("nan"))),
                "history": list(getattr(self, "last_cg_history", [])),
                "wall_time_s": elapsed,
            }
            return compliance, dc_phys, metrics
        finally:
            self.Emin = old_emin
            self.cg_maxiter = old_maxiter

    def solve(self, rho_phys: np.ndarray, penal: float) -> tuple[float, np.ndarray]:
        attempted: list[dict] = []

        compliance, dc_phys, probe = self._solve_with_floor(
            rho_phys,
            penal,
            rho_min=self.guarded_base_rho_min,
            maxiter=self.guarded_probe_iters,
            phase="probe",
        )
        attempted.append(probe)
        r50 = _history_at(probe["history"], self.guarded_probe_r50)
        r100 = _history_at(probe["history"], self.guarded_probe_iters)
        self.last_policy_probe_r50 = r50
        self.last_policy_probe_r100 = r100
        self.last_policy_probe_true_rel_residual = float(probe["true_rel_residual"])

        if probe["true_rel_residual"] <= self.cg_tol:
            selected = probe
            trigger = "probe_converged_keep"
            fallback_used = 0
        else:
            predicted_raise, trigger = _classify_probe(
                r50,
                r100,
                high_residual_threshold=self.guarded_high_residual_threshold,
                plateau_residual_threshold=self.guarded_plateau_residual_threshold,
                plateau_ratio_threshold=self.guarded_plateau_ratio_threshold,
            )
            selected = None
            fallback_used = 0
            if not predicted_raise:
                compliance, dc_phys, full = self._solve_with_floor(
                    rho_phys,
                    penal,
                    rho_min=self.guarded_base_rho_min,
                    maxiter=self.cg_maxiter,
                    phase="guarded_keep_full",
                )
                attempted.append(full)
                if full["true_rel_residual"] <= self.cg_tol:
                    selected = full
                else:
                    fallback_used = 1
                    trigger = f"{trigger}_guard_failed"

            if selected is None:
                for floor in self.guarded_ladder:
                    compliance, dc_phys, ladder = self._solve_with_floor(
                        rho_phys,
                        penal,
                        rho_min=floor,
                        maxiter=self.cg_maxiter,
                        phase="ladder",
                    )
                    attempted.append(ladder)
                    if ladder["true_rel_residual"] <= self.cg_tol:
                        selected = ladder
                        break

            if selected is None:
                self.policy_events.append(
                    {
                        "trigger": trigger,
                        "selected_rho_min": float("nan"),
                        "fallback_used": 1,
                        "attempted_rho_mins": ";".join(f"{row['rho_min']:.0e}" for row in attempted),
                        "probe_r50": r50,
                        "probe_r100": r100,
                        "accepted_true_rel_residual": float("nan"),
                    }
                )
                raise RuntimeError("guarded adaptive floor policy exhausted the ladder")

        self.last_selected_rho_min = float(selected["rho_min"])
        self.last_true_rel_residual = float(selected["true_rel_residual"])
        self.last_cg_iters = int(selected["iters"])
        self.last_cg_history = list(selected["history"])
        self.last_policy_trigger = trigger
        self.last_policy_fallback_used = int(fallback_used)
        self.last_policy_attempted_rho_mins = ";".join(f"{row['rho_min']:.0e}" for row in attempted)
        event = {
            "trigger": trigger,
            "selected_rho_min": self.last_selected_rho_min,
            "fallback_used": self.last_policy_fallback_used,
            "attempted_rho_mins": self.last_policy_attempted_rho_mins,
            "probe_r50": r50,
            "probe_r100": r100,
            "probe_true_rel_residual": self.last_policy_probe_true_rel_residual,
            "accepted_true_rel_residual": self.last_true_rel_residual,
            "accepted_iters": self.last_cg_iters,
            "attempt_count": len(attempted),
            "policy_time_s": sum(float(row["wall_time_s"]) for row in attempted),
        }
        self.policy_events.append(event)
        return compliance, dc_phys


def _parse_csv_floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def _vram_used_gb() -> float:
    try:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        return float((total - free) / 1024**3)
    except Exception:
        return float("nan")


def _solver_kwargs(spec, args: argparse.Namespace) -> dict:
    kwargs = {
        "grid_dims": (spec.nelx, spec.nely, spec.nelz),
        "enable_warm_start": True,
        "enable_matrix_free": True,
        "enable_fused_cuda": False,
        "enable_matfree_gmg": True,
        "matfree_gmg_levels": args.n_levels,
        "gmg_fine_smoother": args.fine_smoother,
        "gmg_fine_degree": args.fine_degree,
        "gmg_outer_solver": "fgmres",
        "gmg_restart": args.restart,
        "gmg_smoother_type": args.smoother_type,
        "gmg_cycle_type": args.cycle_type,
        "cg_tol": args.tol,
        "cg_maxiter": args.maxiter,
        "enable_profiling": True,
    }
    if args.policy == "guarded_adaptive":
        kwargs.update(
            {
                "guarded_base_rho_min": args.baseline_rho_min,
                "guarded_ladder": tuple(_parse_csv_floats(args.ladder_rho_mins)),
                "guarded_probe_iters": args.probe_iters,
                "guarded_probe_r50": args.probe_r50,
                "guarded_high_residual_threshold": args.high_residual_threshold,
                "guarded_plateau_residual_threshold": args.plateau_residual_threshold,
                "guarded_plateau_ratio_threshold": args.plateau_ratio_threshold,
            }
        )
    return kwargs


def _run_case(
    *,
    preset: str,
    rho_min: float,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[dict, list[dict]]:
    spec = get_preset(preset)
    bc = generate_bc(spec)
    active_rho_min = args.baseline_rho_min if args.policy == "guarded_adaptive" else rho_min
    params = TO3DParams(
        nelx=spec.nelx,
        nely=spec.nely,
        nelz=spec.nelz,
        volfrac=spec.volfrac,
        penal=args.penal,
        emin=active_rho_min,
        rmin=spec.rmin if spec.rmin is not None else 1.5,
        max_iter=args.iters,
        tol=args.opt_tol,
        min_iter=min(args.min_iter, args.iters),
    )

    run_id = (
        f"{preset}_guarded_adaptive"
        if args.policy == "guarded_adaptive"
        else f"{preset}_rho{rho_min:.0e}".replace("+", "")
    )
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    progress_rows: list[dict] = []

    def progress_callback(state: dict) -> None:
        row = {
            "run_id": run_id,
            "preset": preset,
            "rho_min": rho_min,
            **state,
            "vram_used_gb": _vram_used_gb(),
        }
        progress_rows.append(row)
        print(
            f"{run_id},iter={state['iteration']},C={state['compliance']:.9g},"
            f"gray={state['grayness']:.6g},floor={state.get('selected_rho_min', active_rho_min):.0e},"
            f"iters={state['cg_iters']},r={state.get('solver_true_rel_residual', float('nan')):.3e}",
            flush=True,
        )

    t0 = time.perf_counter()
    error = ""
    try:
        result = run_simp_surrogate_gpu(
            params=params,
            fixed=bc.fixed_dofs.astype(np.int32),
            free=bc.free_dofs.astype(np.int32),
            F=bc.F,
            ndof=bc.ndof,
            surrogate=None,
            router=PureFEMRouter(),
            device="auto",
            param_controller=ScheduleOnlyController(),
            verbose=False,
            progress_callback=progress_callback,
            solver_class=GuardedAdaptiveFloorSolverV4 if args.policy == "guarded_adaptive" else SolverV4,
            solver_kwargs=_solver_kwargs(spec, args),
        )
    except Exception as exc:
        result = {}
        error = f"{type(exc).__name__}: {exc}"
    wall_s = time.perf_counter() - t0

    if "rho_final" in result:
        np.save(run_dir / "rho_final.npy", result["rho_final"])
    if "rho_best" in result:
        np.save(run_dir / "rho_best.npy", result["rho_best"])
    (run_dir / "params_log.json").write_text(
        json.dumps(result.get("params_log", []), indent=2),
        encoding="utf-8",
    )

    params_log = result.get("params_log", [])
    final_log = params_log[-1] if params_log else {}
    summary = {
        "run_id": run_id,
        "preset": preset,
        "rho_min": rho_min,
        "policy": args.policy,
        "n_elem": spec.nelx * spec.nely * spec.nelz,
        "volfrac": spec.volfrac,
        "rmin": spec.rmin if spec.rmin is not None else 1.5,
        "penal": args.penal,
        "iters_requested": args.iters,
        "iters_completed": int(final_log.get("iter", 0)),
        "wall_time_s": wall_s,
        "final_compliance": float(result.get("final_compliance", float("nan"))),
        "best_compliance": float(result.get("best_compliance", float("nan"))),
        "best_iteration": int(result.get("best_iteration", 0) or 0),
        "final_grayness": float(result.get("final_grayness", float("nan"))),
        "best_grayness": float(result.get("best_grayness", float("nan"))),
        "best_is_valid": bool(result.get("best_is_valid", False)),
        "fem_calls": int(result.get("fem_calls", 0) or 0),
        "surrogate_calls": int(result.get("surrogate_calls", 0) or 0),
        "final_volume": float(final_log.get("volume", float("nan"))),
        "final_change": float(final_log.get("change", float("nan"))),
        "final_solver_iters": int(final_log.get("cg_iters", -1)),
        "vram_used_gb_after": _vram_used_gb(),
        "solver_stack": "SolverV4 matrix-free GMG-FGMRES trajectory",
        "policy_semantics": (
            "true_guarded_adaptive_in_loop_policy"
            if args.policy == "guarded_adaptive"
            else "fixed_floor_trajectory_not_guarded_policy"
        ),
        "selected_floor_counts": _floor_counts(progress_rows),
        "fallback_events": sum(int(row.get("policy_fallback_used", 0) or 0) for row in progress_rows),
        "max_true_rel_residual": _max_float(progress_rows, "solver_true_rel_residual"),
        "mean_policy_probe_r50": _mean_float(progress_rows, "policy_probe_r50"),
        "error": error,
    }
    return summary, progress_rows


def _floor_counts(rows: list[dict]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get("selected_rho_min", "")
        try:
            key = f"{float(value):.0e}"
        except Exception:
            continue
        counts[key] = counts.get(key, 0) + 1
    return "; ".join(f"{key}:{counts[key]}" for key in sorted(counts, key=lambda v: float(v)))


def _mean_float(rows: list[dict], key: str) -> float:
    vals = []
    for row in rows:
        try:
            val = float(row.get(key, float("nan")))
        except Exception:
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def _max_float(rows: list[dict], key: str) -> float:
    vals = []
    for row in rows:
        try:
            val = float(row.get(key, float("nan")))
        except Exception:
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.max(vals)) if vals else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", default="cantilever_gpu_medium,bridge_gpu_medium")
    parser.add_argument("--rho-mins", default="1e-12,1e-3,1e-2")
    parser.add_argument("--policy", choices=["fixed_floor", "guarded_adaptive"], default="fixed_floor")
    parser.add_argument("--baseline-rho-min", type=float, default=1e-12)
    parser.add_argument("--ladder-rho-mins", default="1e-3,1e-2")
    parser.add_argument("--probe-iters", type=int, default=100)
    parser.add_argument("--probe-r50", type=int, default=50)
    parser.add_argument("--high-residual-threshold", type=float, default=1e-2)
    parser.add_argument("--plateau-residual-threshold", type=float, default=1e-4)
    parser.add_argument("--plateau-ratio-threshold", type=float, default=0.6)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--min-iter", type=int, default=20)
    parser.add_argument("--penal", type=float, default=4.5)
    parser.add_argument("--opt-tol", type=float, default=0.01)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--restart", type=int, default=300)
    parser.add_argument("--n-levels", type=int, default=4)
    parser.add_argument("--fine-smoother", default="fp64")
    parser.add_argument("--fine-degree", type=int, default=2)
    parser.add_argument("--smoother-type", default="chebyshev", choices=["chebyshev", "jacobi"])
    parser.add_argument("--cycle-type", default="v", choices=["v", "w"])
    parser.add_argument(
        "--allow-failed-cases",
        action="store_true",
        help="Write error rows and return success even if every trajectory case fails.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results" / "simp_floor_trajectories"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    summaries: list[dict] = []
    all_progress: list[dict] = []
    for preset in _parse_csv_strings(args.presets):
        rho_min_values = [args.baseline_rho_min] if args.policy == "guarded_adaptive" else _parse_csv_floats(args.rho_mins)
        for rho_min in rho_min_values:
            summary, progress = _run_case(preset=preset, rho_min=rho_min, args=args, out_dir=out_dir)
            summaries.append(summary)
            all_progress.extend(progress)
            _write_csv(out_dir / "trajectory_summary.csv", summaries)
            _write_csv(out_dir / "trajectory_iters.csv", all_progress)

    print(f"Wrote {out_dir / 'trajectory_summary.csv'}", flush=True)
    print(f"Wrote {out_dir / 'trajectory_iters.csv'}", flush=True)
    failures = [row for row in summaries if str(row.get("error", "")).strip()]
    if failures and not args.allow_failed_cases:
        raise SystemExit(f"{len(failures)}/{len(summaries)} trajectory cases failed; see trajectory_summary.csv")


if __name__ == "__main__":
    main()
