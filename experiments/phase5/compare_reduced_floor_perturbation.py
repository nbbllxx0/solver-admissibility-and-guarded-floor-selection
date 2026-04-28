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


def _element_stiffness_from_mask(mask: np.ndarray, rho_min: float, penal: float) -> np.ndarray:
    rho = np.where(mask, 1.0, rho_min)
    return rho_min + (1.0 - rho_min) * rho**penal


def _assemble_kff_cpu(paper4, spec, free: np.ndarray, E_cpu: np.ndarray):
    import scipy.sparse as sp

    edof = paper4._edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    row_idx, col_idx = paper4._build_sparse_indices(edof)
    n_elem = spec.nelx * spec.nely * spec.nelz
    data = np.tile(paper4.KE_UNIT_3D.ravel(), n_elem) * np.repeat(E_cpu, 576)
    ndof = 3 * (spec.nelx + 1) * (spec.nely + 1) * (spec.nelz + 1)
    K = sp.csr_matrix((data, (row_idx, col_idx)), shape=(ndof, ndof))
    K.sum_duplicates()
    return K[free][:, free].tocsr()


def _safe_energy_norm(K, x: np.ndarray) -> float:
    value = float(x @ (K @ x))
    return float(np.sqrt(max(value, 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="cantilever_3d_large")
    parser.add_argument("--probabilities", default="0.10,0.20,0.35")
    parser.add_argument("--seeds", default="19,41")
    parser.add_argument("--reference-rho-min", type=float, default=1e-12)
    parser.add_argument(
        "--rho-min-values",
        default="1e-12,1e-10,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2",
    )
    parser.add_argument("--penal", default=4.5, type=float)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results" / "reduced_floor_perturbation"),
    )
    args = parser.parse_args()

    from gpu_fem.bc_generator import generate_bc
    from gpu_fem.presets import get_preset
    from scipy.sparse.linalg import spsolve

    paper4 = _load_paper4_runner()
    spec = get_preset(args.preset)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    Ff = bc.F[free].astype(np.float64)
    b_norm = float(np.linalg.norm(Ff))
    n_elem = spec.nelx * spec.nely * spec.nelz
    seeds = _parse_int_list(args.seeds)
    probabilities = _parse_float_list(args.probabilities)
    rho_values = _parse_float_list(args.rho_min_values)
    if args.reference_rho_min not in rho_values:
        rho_values = [args.reference_rho_min, *rho_values]

    rows: list[dict] = []
    print(
        "preset,seed,p,rho_min,rel_u_error,rel_compliance_change,energy_error,original_operator_residual",
        flush=True,
    )
    for seed in seeds:
        rng = np.random.default_rng(seed)
        random_field = rng.random(n_elem)
        for probability in probabilities:
            mask = random_field < probability
            ref_E = _element_stiffness_from_mask(mask, args.reference_rho_min, args.penal)
            t0 = time.perf_counter()
            K_ref = _assemble_kff_cpu(paper4, spec, free, ref_E)
            x_ref = spsolve(K_ref, Ff)
            ref_solve_time = time.perf_counter() - t0
            compliance_ref = float(Ff @ x_ref)
            ref_norm = float(np.linalg.norm(x_ref))
            ref_energy = _safe_energy_norm(K_ref, x_ref)
            ref_residual = float(np.linalg.norm(Ff - K_ref @ x_ref) / max(b_norm, 1e-300))

            for rho_min in rho_values:
                E = _element_stiffness_from_mask(mask, rho_min, args.penal)
                t1 = time.perf_counter()
                K = _assemble_kff_cpu(paper4, spec, free, E)
                x = spsolve(K, Ff)
                solve_time = time.perf_counter() - t1
                dx = x - x_ref
                compliance = float(Ff @ x)
                rel_u_error = float(np.linalg.norm(dx) / max(ref_norm, 1e-300))
                rel_compliance_change = float((compliance - compliance_ref) / max(abs(compliance_ref), 1e-300))
                energy_error = float(_safe_energy_norm(K_ref, dx) / max(ref_energy, 1e-300))
                residual_under_ref = float(np.linalg.norm(Ff - K_ref @ x) / max(b_norm, 1e-300))
                rel_residual = float(np.linalg.norm(Ff - K @ x) / max(b_norm, 1e-300))
                row = {
                    "preset": args.preset,
                    "seed": seed,
                    "solid_probability": probability,
                    "n_elem": n_elem,
                    "n_free": len(free),
                    "rho_min": rho_min,
                    "reference_rho_min": args.reference_rho_min,
                    "penal": args.penal,
                    "solid_count": int(mask.sum()),
                    "solid_fraction": float(mask.mean()),
                    "compliance": compliance,
                    "reference_compliance": compliance_ref,
                    "rel_compliance_change": rel_compliance_change,
                    "rel_u_error": rel_u_error,
                    "energy_norm_error": energy_error,
                    "direct_rel_residual": rel_residual,
                    "reference_direct_rel_residual": ref_residual,
                    "raised_solution_residual_under_reference_operator": residual_under_ref,
                    "reference_solve_time_s": ref_solve_time,
                    "solve_time_s": solve_time,
                }
                rows.append(row)
                print(
                    f"{args.preset},{seed},{probability:.4g},{rho_min:.0e},"
                    f"{rel_u_error:.6g},{rel_compliance_change:.6g},"
                    f"{energy_error:.6g},{residual_under_ref:.6g}",
                    flush=True,
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reduced_floor_perturbation_summary.csv"
    fieldnames = [
        "preset",
        "seed",
        "solid_probability",
        "n_elem",
        "n_free",
        "rho_min",
        "reference_rho_min",
        "penal",
        "solid_count",
        "solid_fraction",
        "compliance",
        "reference_compliance",
        "rel_compliance_change",
        "rel_u_error",
        "energy_norm_error",
        "direct_rel_residual",
        "reference_direct_rel_residual",
        "raised_solution_residual_under_reference_operator",
        "reference_solve_time_s",
        "solve_time_s",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
