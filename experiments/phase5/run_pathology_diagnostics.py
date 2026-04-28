"""
Phase 5 targeted pathology diagnostics for the canonical paper4 hard cases.

This script is deliberately narrower than the full paper benchmark suite.  It
traces one multigrid cycle on each selected case, records per-level correction
quality, and optionally runs the matching outer solve.  It is intended to
establish mechanism and verify breakthroughs before any broad sweep.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


def _prefer_pytorch_env() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_tmp = root / "tmp" / "phase5_runtime_tmp"
    cupy_cache = root / "tmp" / "phase5_cupy_cache"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    cupy_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(runtime_tmp)
    os.environ["TEMP"] = str(runtime_tmp)
    os.environ["CUPY_CACHE_DIR"] = str(cupy_cache)
    os.environ["CUPY_TEMPDIR"] = str(runtime_tmp)
    os.environ["CUPY_CACHE_IN_MEMORY"] = "1"
    tempfile.tempdir = str(runtime_tmp)

    def _safe_mkdtemp(suffix=None, prefix=None, dir=None):
        suffix = "" if suffix is None else suffix
        prefix = "tmp" if prefix is None else prefix
        base = Path(dir or tempfile.tempdir or runtime_tmp)
        base.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            candidate = base / f"{prefix}{os.urandom(8).hex()}{suffix}"
            try:
                candidate.mkdir(mode=0o777)
            except FileExistsError:
                continue
            return str(candidate)
        raise FileExistsError(f"could not create a temporary directory under {base}")

    tempfile.mkdtemp = _safe_mkdtemp

    class _SafeTemporaryDirectory:
        def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False):
            self.name = _safe_mkdtemp(suffix=suffix, prefix=prefix, dir=dir)

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            return False

        def cleanup(self):
            return None

    tempfile.TemporaryDirectory = _SafeTemporaryDirectory


_prefer_pytorch_env()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def _load_paper4_runner():
    mod_path = ROOT / "experiments" / "paper4" / "run_experiments_e1_e10.py"
    spec = importlib.util.spec_from_file_location("paper4_runner", mod_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load module from {mod_path}")
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _make_cases(paper4):
    import cupy as cp

    def checkerboard(spec):
        n_elem = spec.nelx * spec.nely * spec.nelz
        rho = cp.asarray((np.arange(n_elem) % 2).astype(np.float64))
        return (1e-9 + (1.0 - 1e-9) * rho**3.0)

    def layered_band(spec):
        n_elem = spec.nelx * spec.nely * spec.nelz
        rho = np.repeat((np.arange(spec.nelx) < spec.nelx // 2).astype(np.float64), spec.nely * spec.nelz)
        return (1e-9 + (1.0 - 1e-9) * cp.asarray(rho)**3.0)

    def mixed_very_low(spec):
        n_elem = spec.nelx * spec.nely * spec.nelz
        rho = np.where(np.random.default_rng(19).random(n_elem) < 0.1, 1.0, 1e-12)
        return (1e-12 + (1.0 - 1e-12) * cp.asarray(rho)**4.5)

    return {
        "uniform-vf0.5": {
            "family": "control",
            "volfrac": 0.5,
            "rho_min": 0.5,
            "penal": 3.0,
            "seed": "deterministic",
            "build": lambda spec: paper4._E_e(spec, rho_scalar=0.5, penal=3.0),
        },
        "layered-band": {
            "family": "control",
            "volfrac": 0.5,
            "rho_min": 1e-9,
            "penal": 3.0,
            "seed": "deterministic",
            "build": layered_band,
        },
        "binary-vf0.2-p1.5": {
            "family": "hardset",
            "volfrac": 0.2,
            "rho_min": 1e-9,
            "penal": 1.5,
            "seed": 7,
            "build": lambda spec: paper4._E_e_heterogeneous(spec, volfrac=0.2, penal=1.5, rho_min=1e-9, seed=7),
        },
        "binary-vf0.8-p4.5": {
            "family": "hardset",
            "volfrac": 0.8,
            "rho_min": 1e-9,
            "penal": 4.5,
            "seed": 13,
            "build": lambda spec: paper4._E_e_heterogeneous(spec, volfrac=0.8, penal=4.5, rho_min=1e-9, seed=13),
        },
        "checkerboard": {
            "family": "hardset",
            "volfrac": 0.5,
            "rho_min": 1e-9,
            "penal": 3.0,
            "seed": "deterministic",
            "build": checkerboard,
        },
        "rho-min-1e-12": {
            "family": "hardset",
            "volfrac": 0.5,
            "rho_min": 1e-12,
            "penal": 3.0,
            "seed": 17,
            "build": lambda spec: paper4._E_e_heterogeneous(spec, volfrac=0.5, penal=3.0, rho_min=1e-12, seed=17),
        },
        "mixed-very-low": {
            "family": "hardset",
            "volfrac": 0.1,
            "rho_min": 1e-12,
            "penal": 4.5,
            "seed": 19,
            "build": mixed_very_low,
        },
    }


def _select_case_names(case_map: dict, selector: str) -> list[str]:
    if selector == "all":
        return list(case_map.keys())
    if selector == "hardset":
        return [name for name, meta in case_map.items() if meta["family"] == "hardset"]
    if selector == "controls":
        return [name for name, meta in case_map.items() if meta["family"] == "control"]
    names = [chunk.strip() for chunk in selector.split(",") if chunk.strip()]
    unknown = [name for name in names if name not in case_map]
    if unknown:
        raise ValueError(f"Unknown case names: {unknown}")
    return names


def _trace_aggregates(trace) -> dict:
    visits = sorted(trace.visits, key=lambda visit: visit.visit_id)
    root_visit = next((visit for visit in visits if visit.level == 0), None)
    max_gap = max((visit.coarse_space_gap_ratio for visit in visits if not visit.is_coarsest), default=0.0)
    worst_first_corr = max((visit.corrected_residual_vs_presmooth for visit in visits if not visit.is_coarsest), default=0.0)
    second_values = [
        visit.second_corrected_residual_vs_corrected
        for visit in visits
        if visit.second_corrected_residual_vs_corrected is not None
    ]
    worst_second_corr = max(second_values, default=0.0)
    alignments = [visit.correction_alignment for visit in visits if not visit.is_coarsest]
    min_alignment = min(alignments, default=0.0)
    applied_scales = [visit.correction_scale_applied for visit in visits if not visit.is_coarsest]
    min_scale = min(applied_scales, default=1.0)
    rejected = sum(1 for visit in visits if not visit.is_coarsest and visit.correction_scale_applied <= 1e-12)
    return {
        "root_presmooth_residual_ratio": root_visit.presmooth_residual_ratio if root_visit else float("nan"),
        "root_corrected_residual_vs_presmooth": root_visit.corrected_residual_vs_presmooth if root_visit else float("nan"),
        "root_postsmooth_residual_ratio": root_visit.postsmooth_residual_ratio if root_visit else float("nan"),
        "root_correction_scale_applied": root_visit.correction_scale_applied if root_visit else float("nan"),
        "max_coarse_space_gap_ratio": max_gap,
        "worst_first_correction_ratio": worst_first_corr,
        "worst_second_correction_ratio": worst_second_corr,
        "min_correction_alignment": min_alignment,
        "min_correction_scale_applied": min_scale,
        "rejected_corrections": rejected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="cantilever_gpu_medium")
    parser.add_argument("--cases", default="hardset",
                        help="hardset, controls, all, or comma-separated explicit case names")
    parser.add_argument("--n-levels", default=4, type=int)
    parser.add_argument("--fine-smoother", default="fp64")
    parser.add_argument("--smoother-type", default="chebyshev")
    parser.add_argument("--cycle-type", default="v", choices=["v", "w"])
    parser.add_argument("--coarse-policy", default="none", choices=["none", "residual_line_search"])
    parser.add_argument("--coarse-acceptance-ratio", default=1.0, type=float)
    parser.add_argument("--coarse-max-scale", default=1.0, type=float)
    parser.add_argument(
        "--root-enrichment-mode",
        default="none",
        choices=[
            "none",
            "weighted_residual_second",
            "weighted_gap_second",
            "void_residual_second",
            "void_gap_second",
        ],
    )
    parser.add_argument("--root-enrichment-weight-floor", default=1e-6, type=float)
    parser.add_argument("--root-enrichment-weight-power", default=0.5, type=float)
    parser.add_argument(
        "--root-local-correction-mode",
        default="none",
        choices=["none", "node_block_line_search", "node_block_inner_fgmres"],
    )
    parser.add_argument("--root-local-correction-max-scale", default=1.0, type=float)
    parser.add_argument("--root-local-correction-inner-steps", default=10, type=int)
    parser.add_argument("--root-local-correction-inner-restart", default=None, type=int)
    parser.add_argument("--inner-fgmres-steps", default=0, type=int)
    parser.add_argument("--inner-fgmres-restart", default=None, type=int)
    parser.add_argument("--restart", default=50, type=int)
    parser.add_argument("--maxiter", default=500, type=int)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "experiments" / "phase5" / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper4 = _load_paper4_runner()
    case_map = _make_cases(paper4)
    case_names = _select_case_names(case_map, args.cases)

    summary_rows = []
    visit_rows = []
    solve_history_rows = []
    trace_json = []

    print(f"Phase 5 pathology diagnostics: preset={args.preset} cases={case_names}")
    for case_name in case_names:
        meta = case_map[case_name]
        spec, bc, free, free_gpu, edof_gpu, F_free_gpu, mf_op, gmg = paper4._build_components(
            args.preset,
            n_levels=args.n_levels,
            fine_smoother=args.fine_smoother,
            smoother_type=args.smoother_type,
            cycle_type=args.cycle_type,
            coarse_correction_policy=args.coarse_policy,
            coarse_correction_acceptance_ratio=args.coarse_acceptance_ratio,
            coarse_correction_max_scale=args.coarse_max_scale,
            root_enrichment_mode=args.root_enrichment_mode,
            root_enrichment_weight_floor=args.root_enrichment_weight_floor,
            root_enrichment_weight_power=args.root_enrichment_weight_power,
            root_local_correction_mode=args.root_local_correction_mode,
            root_local_correction_max_scale=args.root_local_correction_max_scale,
            root_local_correction_inner_steps=args.root_local_correction_inner_steps,
            root_local_correction_inner_restart=args.root_local_correction_inner_restart,
            inner_krylov_steps=args.inner_fgmres_steps,
            inner_krylov_restart=args.inner_fgmres_restart,
        )
        E_e = meta["build"](spec)
        gmg.setup(E_e)

        trace = gmg.cycle_trace(F_free_gpu, cycle_type=args.cycle_type)
        aggregates = _trace_aggregates(trace)
        adapt = gmg.adaptive_params

        solve_conv = None
        solve_iters = None
        solve_error = ""
        solve_history = []
        solve_final_rel_residual = float("nan")
        if not args.skip_solve:
            try:
                def A_op(v):
                    return mf_op.matvec(v, E_e)

                _, solve_iters, solve_conv = paper4._fgmres(
                    A_op,
                    F_free_gpu,
                    gmg.apply,
                    tol=args.tol,
                    maxiter=args.maxiter,
                    restart=args.restart,
                    history=solve_history,
                )
                if solve_history:
                    solve_final_rel_residual = float(solve_history[-1])
            except Exception as exc:
                solve_conv = 0
                solve_iters = -1
                solve_error = str(exc)

        summary_row = {
            "case": case_name,
            "family": meta["family"],
            "preset": args.preset,
            "cycle_type": trace.cycle_type,
            "n_levels": args.n_levels,
            "fine_smoother": args.fine_smoother,
            "smoother_type": args.smoother_type,
            "coarse_correction_policy": trace.coarse_correction_policy,
            "coarse_correction_acceptance_ratio": trace.coarse_correction_acceptance_ratio,
            "coarse_correction_max_scale": trace.coarse_correction_max_scale,
            "root_enrichment_mode": trace.root_enrichment_mode,
            "root_enrichment_weight_floor": trace.root_enrichment_weight_floor,
            "root_enrichment_weight_power": trace.root_enrichment_weight_power,
            "root_local_correction_mode": trace.root_local_correction_mode,
            "root_local_correction_inner_steps": trace.root_local_correction_inner_steps,
            "root_local_correction_inner_restart": trace.root_local_correction_inner_restart,
            "root_local_correction_raw_scale": trace.root_local_correction_raw_scale,
            "root_local_correction_applied_scale": trace.root_local_correction_applied_scale,
            "root_local_correction_residual_ratio": trace.root_local_correction_residual_ratio,
            "inner_krylov_steps": trace.inner_krylov_steps,
            "inner_krylov_restart": trace.inner_krylov_restart,
            "contrast_ratio": trace.contrast_ratio,
            "is_high_contrast": trace.is_high_contrast,
            "fine_degree_adapt": trace.fine_degree_adapt,
            "coarse_iters_adapt": trace.coarse_iters_adapt,
            "rhs_norm": trace.rhs_norm,
            "jacobi_action_norm": trace.jacobi_action_norm,
            "output_norm": trace.output_norm,
            "z_over_jacobi": trace.z_over_jacobi,
            "rz": trace.rz,
            "pd": int(trace.pd),
            "output_residual_norm": trace.output_residual_norm,
            "output_residual_ratio": trace.output_residual_ratio,
            "volfrac": meta["volfrac"],
            "rho_min": meta["rho_min"],
            "penal": meta["penal"],
            "seed": meta["seed"],
            "solve_converged": solve_conv,
            "solve_iters": solve_iters,
            "solve_final_rel_residual": solve_final_rel_residual,
            "solve_history_len": len(solve_history),
            "solve_error": solve_error,
        }
        summary_row.update(aggregates)
        summary_rows.append(summary_row)

        for visit in sorted(trace.visits, key=lambda item: item.visit_id):
            row = {
                "case": case_name,
                "family": meta["family"],
                "preset": args.preset,
                "solve_converged": solve_conv,
                "solve_iters": solve_iters,
            }
            row.update(visit.to_dict())
            visit_rows.append(row)

        for iter_idx, rel_residual in enumerate(solve_history):
            solve_history_rows.append({
                "case": case_name,
                "family": meta["family"],
                "preset": args.preset,
                "cycle_type": trace.cycle_type,
                "coarse_correction_policy": trace.coarse_correction_policy,
                "root_enrichment_mode": trace.root_enrichment_mode,
                "root_local_correction_mode": trace.root_local_correction_mode,
                "inner_krylov_steps": trace.inner_krylov_steps,
                "iter": iter_idx,
                "rel_residual": float(rel_residual),
            })

        trace_json.append({
            "case": case_name,
            "family": meta["family"],
            "summary": summary_row,
            "trace": trace.to_dict(),
        })

        print(
            f"  {case_name:<20} conv={solve_conv} iters={solve_iters} "
            f"z/jac={trace.z_over_jacobi:.3g} root_post={aggregates['root_postsmooth_residual_ratio']:.3g} "
            f"gap_max={aggregates['max_coarse_space_gap_ratio']:.3g} align_min={aggregates['min_correction_alignment']:.3g} "
            f"scale_min={aggregates['min_correction_scale_applied']:.3g}"
        )

    summary_path = out_dir / "pathology_summary.csv"
    visits_path = out_dir / "pathology_visits.csv"
    history_path = out_dir / "pathology_solve_history.csv"
    json_path = out_dir / "pathology_traces.json"
    _write_csv(summary_path, summary_rows)
    _write_csv(visits_path, visit_rows)
    _write_csv(history_path, solve_history_rows)
    _write_json(json_path, trace_json)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {visits_path}")
    if solve_history_rows:
        print(f"Wrote {history_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
