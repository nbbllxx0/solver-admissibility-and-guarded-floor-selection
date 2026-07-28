"""Conditioning of the frozen operator as a function of the SIMP floor.

Companion diagnostic to ``run_direct_floor_atlas.py``. For the same reduced
24x12x6 cantilever fields, this script computes the extreme eigenvalues of the
constrained free-DOF stiffness matrix at a range of floors, so the empirical
critical floor of the atlas can be compared against a floor-independent
quantity: the spectral condition number and the residual level any FP64 solver
can be expected to attain on such a matrix.

Everything here runs on the CPU with SciPy. Nothing in this script touches the
GPU stack, and no result of it is used to accept a solve.

Usage:
    python experiments/phase5/run_direct_floor_conditioning.py
"""

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

# IEEE-754 binary64 unit roundoff; a backward-stable solve cannot be expected to
# return a relative residual far below kappa * EPS.
EPS = float(np.finfo(np.float64).eps)


def _load_paper4_runner():
    path = ROOT / "experiments" / "paper4" / "run_experiments_e1_e10.py"
    spec = importlib.util.spec_from_file_location("paper4_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="cantilever_3d")
    parser.add_argument("--probabilities", default="0.10,0.12,0.15,0.18,0.20,0.35")
    parser.add_argument("--rho-min-values",
                        default="1e-12,1e-10,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--seeds", default="7,13,19")
    parser.add_argument("--penal", default=4.5, type=float)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results" / "direct_floor_conditioning"),
    )
    args = parser.parse_args()

    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh, spsolve

    from gpu_fem.bc_generator import generate_bc
    from gpu_fem.presets import get_preset

    paper4 = _load_paper4_runner()
    spec = get_preset(args.preset)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    Ff = bc.F[free].astype(np.float64)
    b_norm = float(np.linalg.norm(Ff))

    n_elem = spec.nelx * spec.nely * spec.nelz
    ndof = 3 * (spec.nelx + 1) * (spec.nely + 1) * (spec.nelz + 1)
    edof = paper4._edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    row_idx, col_idx = paper4._build_sparse_indices(edof)
    ke_tiled = np.tile(paper4.KE_UNIT_3D.ravel(), n_elem)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print("seed,p,rho_min,lambda_min,lambda_max,kappa,kappa_eps,rel_residual", flush=True)

    for seed in _ints(args.seeds):
        field = np.random.default_rng(seed).random(n_elem)
        for probability in _floats(args.probabilities):
            solid = field < probability
            for rho_min in _floats(args.rho_min_values):
                rho = np.where(solid, 1.0, rho_min)
                E = rho_min + (1.0 - rho_min) * rho ** args.penal
                data = ke_tiled * np.repeat(E, 576)
                K = sp.csr_matrix((data, (row_idx, col_idx)), shape=(ndof, ndof))
                K.sum_duplicates()
                Kff = K[free][:, free].tocsc()

                t0 = time.perf_counter()
                lam_max = float(eigsh(Kff, k=1, which="LA", return_eigenvectors=False)[0])
                # shift-invert at zero: the smallest eigenvalue is the one that
                # collapses with the floor, so it cannot be reached by a plain
                # largest-magnitude iteration
                lam_min = float(eigsh(Kff, k=1, sigma=0.0, which="LM",
                                      return_eigenvectors=False)[0])
                eig_s = time.perf_counter() - t0

                x = spsolve(Kff.tocsr(), Ff)
                rel_residual = float(np.linalg.norm(Ff - Kff @ x) / max(b_norm, 1e-300))

                kappa = lam_max / lam_min if lam_min > 0 else float("inf")
                rows.append({
                    "preset": args.preset,
                    "seed": seed,
                    "solid_probability": probability,
                    "rho_min": rho_min,
                    "penal": args.penal,
                    "lambda_min": lam_min,
                    "lambda_max": lam_max,
                    "kappa": kappa,
                    "kappa_times_eps": kappa * EPS,
                    "rel_residual": rel_residual,
                    "converged": int(rel_residual <= max(args.tol * 1.05, 1e-12)),
                    "n_free": int(Kff.shape[0]),
                    "eig_time_s": eig_s,
                })
                print(f"{seed},{probability:.4g},{rho_min:.0e},{lam_min:.6g},"
                      f"{lam_max:.6g},{kappa:.6g},{kappa * EPS:.3g},{rel_residual:.6g}",
                      flush=True)

    path = out_dir / "direct_floor_conditioning.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
