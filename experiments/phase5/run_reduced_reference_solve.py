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


def _build_mixed_field(spec, *, probability: float, seed: int, rho_min: float, penal: float):
    import cupy as cp

    n_elem = spec.nelx * spec.nely * spec.nelz
    random_field = np.random.default_rng(seed).random(n_elem)
    rho = np.where(random_field < probability, 1.0, rho_min)
    E_cpu = rho_min + (1.0 - rho_min) * rho**penal
    return E_cpu, cp.asarray(E_cpu, dtype=cp.float64)


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


def _rigid_body_modes(spec, free: np.ndarray) -> np.ndarray:
    nx, ny, nz = spec.nelx + 1, spec.nely + 1, spec.nelz + 1
    ndof = 3 * nx * ny * nz
    coords = np.zeros((ndof // 3, 3), dtype=np.float64)
    idx = 0
    for ix in range(nx):
        x = spec.Lx * ix / max(spec.nelx, 1)
        for iy in range(ny):
            y = spec.Ly * iy / max(spec.nely, 1)
            for iz in range(nz):
                z = spec.Lz * iz / max(spec.nelz, 1)
                coords[idx] = (x, y, z)
                idx += 1
    centroid = coords.mean(axis=0)
    rel = coords - centroid
    modes_full = np.zeros((ndof, 6), dtype=np.float64)
    for node, (x, y, z) in enumerate(rel):
        base = 3 * node
        modes_full[base + 0, 0] = 1.0
        modes_full[base + 1, 1] = 1.0
        modes_full[base + 2, 2] = 1.0
        modes_full[base + 1, 3] = -z
        modes_full[base + 2, 3] = y
        modes_full[base + 0, 4] = z
        modes_full[base + 2, 4] = -x
        modes_full[base + 0, 5] = -y
        modes_full[base + 1, 5] = x
    B = modes_full[free, :]
    keep = np.linalg.norm(B, axis=0) > 1e-14
    return B[:, keep]


def _run_cpu_cg(Kff, Ff, *, M=None, tol: float, maxiter: int):
    from scipy.sparse.linalg import cg

    residuals = []
    b_norm = np.linalg.norm(Ff)

    def callback(xk):
        residuals.append(float(np.linalg.norm(Ff - Kff @ xk) / max(b_norm, 1e-300)))

    t0 = time.perf_counter()
    x, info = cg(Kff, Ff, M=M, rtol=tol, atol=0.0, maxiter=maxiter, callback=callback)
    elapsed = time.perf_counter() - t0
    rel = float(np.linalg.norm(Ff - Kff @ x) / max(b_norm, 1e-300))
    iters = len(residuals)
    converged = info == 0 and rel <= max(tol * 1.05, 1e-12)
    return {
        "converged": int(converged),
        "iters": iters,
        "info": info,
        "rel_residual": rel,
        "wall_s": elapsed,
    }


def _run_cpu_direct(Kff, Ff, *, tol: float):
    from scipy.sparse.linalg import spsolve

    b_norm = np.linalg.norm(Ff)
    t0 = time.perf_counter()
    x = spsolve(Kff, Ff)
    elapsed = time.perf_counter() - t0
    rel = float(np.linalg.norm(Ff - Kff @ x) / max(b_norm, 1e-300))
    return {
        "converged": int(rel <= max(tol * 1.05, 1e-12)),
        "iters": 1,
        "info": 0 if rel <= max(tol * 1.05, 1e-12) else 1,
        "rel_residual": rel,
        "wall_s": elapsed,
        "x_norm": float(np.linalg.norm(x)),
    }


def _run_gpu_gmg(paper4, preset: str, E_gpu, *, tol: float, maxiter: int, restart: int):
    import cupy as cp
    from gpu_fem.multigrid_v4 import _cupy_fgmres

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
    gmg.setup(E_gpu)

    def A_op(x):
        return mf_op.matvec(x, E_gpu)

    history = []
    t0 = time.perf_counter()
    x, iters, converged = _cupy_fgmres(
        A_op,
        F_free_gpu,
        gmg.apply,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        history=history,
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    rel = float(cp.linalg.norm(F_free_gpu - A_op(x)) / cp.linalg.norm(F_free_gpu))
    true_converged = converged and rel <= max(tol * 1.05, 1e-12)
    return {
        "converged": int(true_converged),
        "iters": iters,
        "info": 0 if true_converged else maxiter,
        "rel_residual": rel,
        "wall_s": elapsed,
        "history_len": len(history),
        "solver_reported_converged": int(converged),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="cantilever_3d_large")
    parser.add_argument("--probabilities", default="0.10,0.20")
    parser.add_argument("--seeds", default="19")
    parser.add_argument("--rho-min", default=1e-12, type=float)
    parser.add_argument("--penal", default=4.5, type=float)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument("--maxiter", default=300, type=int)
    parser.add_argument("--restart", default=300, type=int)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results" / "reduced_reference_solve"),
    )
    args = parser.parse_args()

    import pyamg
    import scipy.sparse as sp

    paper4 = _load_paper4_runner()
    probabilities = _parse_float_list(args.probabilities)
    seeds = _parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build once to get the exact paper4 BC/free-DOF layout.
    spec, bc, free, free_gpu, edof_gpu, F_free_gpu, mf_op, gmg = paper4._build_components(
        args.preset,
        n_levels=4,
        fine_smoother="fp64",
        smoother_type="chebyshev",
    )
    Ff = bc.F[free].astype(np.float64)
    B = _rigid_body_modes(spec, free)

    rows = []
    print("preset,seed,p,method,converged,iters,info,rel_residual,wall_s", flush=True)
    for seed in seeds:
        for probability in probabilities:
            E_cpu, E_gpu = _build_mixed_field(
                spec,
                probability=probability,
                seed=seed,
                rho_min=args.rho_min,
                penal=args.penal,
            )
            Kff = _assemble_kff_cpu(paper4, spec, free, E_cpu)

            method_results = []
            method_results.append(("cpu_direct_spsolve", _run_cpu_direct(Kff, Ff, tol=args.tol)))

            diag = Kff.diagonal()
            diag_safe = np.where(np.abs(diag) > 1e-300, diag, 1.0)
            M_jacobi = sp.diags(1.0 / diag_safe, format="csr")
            method_results.append(("cpu_cg_jacobi", _run_cpu_cg(Kff, Ff, M=M_jacobi, tol=args.tol, maxiter=args.maxiter)))

            t0 = time.perf_counter()
            ml_plain = pyamg.smoothed_aggregation_solver(Kff, symmetry="symmetric")
            build_plain = time.perf_counter() - t0
            result_plain = _run_cpu_cg(
                Kff,
                Ff,
                M=ml_plain.aspreconditioner(),
                tol=args.tol,
                maxiter=args.maxiter,
            )
            result_plain["build_s"] = build_plain
            method_results.append(("cpu_pyamg_plain", result_plain))

            t0 = time.perf_counter()
            ml_rbm = pyamg.smoothed_aggregation_solver(Kff, B=B, symmetry="symmetric")
            build_rbm = time.perf_counter() - t0
            result_rbm = _run_cpu_cg(
                Kff,
                Ff,
                M=ml_rbm.aspreconditioner(),
                tol=args.tol,
                maxiter=args.maxiter,
            )
            result_rbm["build_s"] = build_rbm
            method_results.append(("cpu_pyamg_rbm", result_rbm))

            method_results.append(
                (
                    "gpu_gmg_phase5",
                    _run_gpu_gmg(
                        paper4,
                        args.preset,
                        E_gpu,
                        tol=args.tol,
                        maxiter=args.maxiter,
                        restart=args.restart,
                    ),
                )
            )

            for method, result in method_results:
                row = {
                    "preset": args.preset,
                    "seed": seed,
                    "solid_probability": probability,
                    "rho_min": args.rho_min,
                    "penal": args.penal,
                    "method": method,
                    "n_free": len(free),
                    "nnz": int(Kff.nnz),
                    "converged": result["converged"],
                    "iters": result["iters"],
                    "info": result["info"],
                    "rel_residual": result["rel_residual"],
                    "wall_s": result["wall_s"],
                    "build_s": result.get("build_s", 0.0),
                    "history_len": result.get("history_len", ""),
                    "x_norm": result.get("x_norm", ""),
                    "solver_reported_converged": result.get("solver_reported_converged", ""),
                }
                rows.append(row)
                print(
                    f"{args.preset},{seed},{probability:.4g},{method},{row['converged']},"
                    f"{row['iters']},{row['info']},{row['rel_residual']:.12g},{row['wall_s']:.3f}",
                    flush=True,
                )

    out_path = out_dir / "reduced_reference_summary.csv"
    fieldnames = [
        "preset",
        "seed",
        "solid_probability",
        "rho_min",
        "penal",
        "method",
        "n_free",
        "nnz",
        "converged",
        "iters",
        "info",
        "rel_residual",
        "wall_s",
        "build_s",
        "history_len",
        "x_norm",
        "solver_reported_converged",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
