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


def _read_rule_thresholds(path: Path, *, safety_factor: float, mode: str) -> dict[float, float]:
    if mode not in {"median", "conservative"}:
        raise ValueError(f"unknown rule mode {mode!r}")
    field = "median_seed_threshold" if mode == "median" else "conservative_seed_threshold"
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if abs(float(row["safety_factor"]) - safety_factor) < 1e-12:
                out[float(row["solid_probability"])] = float(row[field])
    if not out:
        raise ValueError(f"no thresholds for safety_factor={safety_factor} in {path}")
    return out


def _lookup_threshold(probability: float, thresholds: dict[float, float]) -> float:
    if probability in thresholds:
        return thresholds[probability]
    nearest = min(thresholds, key=lambda p: abs(p - probability))
    return thresholds[nearest]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="cantilever_gpu_medium")
    parser.add_argument("--probabilities", default="0.10,0.12,0.15,0.18,0.20,0.35")
    parser.add_argument("--seeds", default="19")
    parser.add_argument("--baseline-rho-min", default=1e-12, type=float)
    parser.add_argument("--penal", default=4.5, type=float)
    parser.add_argument("--restart", default=300, type=int)
    parser.add_argument("--maxiter", default=300, type=int)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument("--rule-safety-factor", default=10.0, type=float)
    parser.add_argument("--rule-mode", default="median", choices=["median", "conservative"])
    parser.add_argument(
        "--rule-table",
        default=str(
            ROOT
            / "experiments"
            / "phase5"
            / "results"
            / "admissibility_detector_validation"
            / "detector_rule_thresholds.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results" / "gmg_floor_rule_check"),
    )
    args = parser.parse_args()

    import cupy as cp
    from gpu_fem.multigrid_v4 import _cupy_fgmres

    paper4 = _load_paper4_runner()
    probabilities = _parse_float_list(args.probabilities)
    seeds = _parse_int_list(args.seeds)
    thresholds = _read_rule_thresholds(
        Path(args.rule_table),
        safety_factor=args.rule_safety_factor,
        mode=args.rule_mode,
    )
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
        "seed,solid_probability,case_kind,rho_min,solve_converged,solve_iters,solve_final_rel_residual,"
        "n_components,largest_fraction_of_solid",
        flush=True,
    )
    for seed in seeds:
        random_field = np.random.default_rng(seed).random(spec.nelx * spec.nely * spec.nelz)
        random_field = random_field.reshape(spec.nelx, spec.nely, spec.nelz)
        for probability in probabilities:
            solid = random_field < probability
            conn = _connectivity(solid)
            rule_rho_min = _lookup_threshold(probability, thresholds)
            cases = [("baseline", args.baseline_rho_min)]
            if abs(rule_rho_min - args.baseline_rho_min) / max(rule_rho_min, args.baseline_rho_min) > 1e-12:
                cases.append((f"rule_safety_{args.rule_safety_factor:g}_{args.rule_mode}", rule_rho_min))

            for case_kind, rho_min in cases:
                rho = np.where(solid.reshape(-1), 1.0, rho_min)
                E_e = rho_min + (1.0 - rho_min) * cp.asarray(rho, dtype=cp.float64) ** args.penal
                gmg.setup(E_e)

                def A_op(x):
                    return mf_op.matvec(x, E_e)

                history = []
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
                    "case_kind": case_kind,
                    "rho_min": rho_min,
                    "penal": args.penal,
                    "restart": args.restart,
                    "maxiter": args.maxiter,
                    "tol": args.tol,
                    **conn,
                    "solve_converged": int(solve_converged),
                    "solve_iters": solve_iters,
                    "solve_final_rel_residual": solve_final,
                    "solve_history_len": len(history),
                }
                summary_rows.append(row)
                for i, value in enumerate(history):
                    history_rows.append(
                        {
                            "seed": seed,
                            "solid_probability": probability,
                            "case_kind": case_kind,
                            "rho_min": rho_min,
                            "iter": i,
                            "rel_residual": float(value),
                        }
                    )
                print(
                    f"{seed},{probability:.4g},{case_kind},{rho_min:.0e},{int(solve_converged)},"
                    f"{solve_iters},{solve_final:.12g},{conn['n_components']},"
                    f"{conn['largest_fraction_of_solid']:.12g}",
                    flush=True,
                )

    summary_fields = [
        "preset",
        "seed",
        "solid_probability",
        "case_kind",
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
        "support_to_load_patch_components",
        "solve_converged",
        "solve_iters",
        "solve_final_rel_residual",
        "solve_history_len",
    ]
    _write_csv(out_dir / "gmg_floor_rule_summary.csv", summary_rows, summary_fields)
    _write_csv(
        out_dir / "gmg_floor_rule_history.csv",
        history_rows,
        ["seed", "solid_probability", "case_kind", "rho_min", "iter", "rel_residual"],
    )
    print(f"Wrote {out_dir / 'gmg_floor_rule_summary.csv'}", flush=True)
    print(f"Wrote {out_dir / 'gmg_floor_rule_history.csv'}", flush=True)


if __name__ == "__main__":
    main()
