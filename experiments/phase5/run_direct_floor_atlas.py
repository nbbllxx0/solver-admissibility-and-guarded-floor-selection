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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="cantilever_3d")
    parser.add_argument("--probabilities", default="0.10,0.12,0.15,0.18,0.20,0.35")
    parser.add_argument("--rho-min-values", default="1e-12,1e-10,1e-8,1e-6,1e-4,1e-3")
    parser.add_argument("--seeds", default="19")
    parser.add_argument("--penal", default=4.5, type=float)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "phase5" / "results" / "direct_floor_atlas"),
    )
    args = parser.parse_args()

    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from gpu_fem.bc_generator import generate_bc
    from gpu_fem.presets import get_preset

    paper4 = _load_paper4_runner()
    spec = get_preset(args.preset)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    Ff = bc.F[free].astype(np.float64)
    b_norm = np.linalg.norm(Ff)

    n_elem = spec.nelx * spec.nely * spec.nelz
    ndof = 3 * (spec.nelx + 1) * (spec.nely + 1) * (spec.nelz + 1)
    edof = paper4._edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    row_idx, col_idx = paper4._build_sparse_indices(edof)
    ke_tiled = np.tile(paper4.KE_UNIT_3D.ravel(), n_elem)

    probabilities = _parse_float_list(args.probabilities)
    rho_min_values = _parse_float_list(args.rho_min_values)
    seeds = _parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(
        "seed,p,rho_min,converged,rel_residual,x_norm,diag_min,diag_max,diag_ratio,wall_s",
        flush=True,
    )
    for seed in seeds:
        random_field = np.random.default_rng(seed).random(n_elem)
        for probability in probabilities:
            solid = random_field < probability
            for rho_min in rho_min_values:
                rho = np.where(solid, 1.0, rho_min)
                E = rho_min + (1.0 - rho_min) * rho**args.penal
                data = ke_tiled * np.repeat(E, 576)
                K = sp.csr_matrix((data, (row_idx, col_idx)), shape=(ndof, ndof))
                K.sum_duplicates()
                Kff = K[free][:, free].tocsr()
                t0 = time.perf_counter()
                x = spsolve(Kff, Ff)
                wall_s = time.perf_counter() - t0
                residual = Ff - Kff @ x
                rel_residual = float(np.linalg.norm(residual) / max(b_norm, 1e-300))
                x_norm = float(np.linalg.norm(x))
                diag = Kff.diagonal()
                diag_min = float(np.min(np.abs(diag)))
                diag_max = float(np.max(np.abs(diag)))
                diag_ratio = diag_max / max(diag_min, 1e-300)
                row = {
                    "preset": args.preset,
                    "seed": seed,
                    "solid_probability": probability,
                    "rho_min": rho_min,
                    "penal": args.penal,
                    "tol": args.tol,
                    "converged": int(rel_residual <= max(args.tol * 1.05, 1e-12)),
                    "rel_residual": rel_residual,
                    "x_norm": x_norm,
                    "diag_min": diag_min,
                    "diag_max": diag_max,
                    "diag_ratio": diag_ratio,
                    "wall_s": wall_s,
                    "n_free": len(free),
                    "nnz": int(Kff.nnz),
                }
                rows.append(row)
                print(
                    f"{seed},{probability:.4g},{rho_min:.0e},{row['converged']},"
                    f"{rel_residual:.12g},{x_norm:.12g},{diag_min:.3g},{diag_max:.3g},"
                    f"{diag_ratio:.3g},{wall_s:.3f}",
                    flush=True,
                )

    summary_path = out_dir / "direct_floor_atlas.csv"
    fieldnames = [
        "preset",
        "seed",
        "solid_probability",
        "rho_min",
        "penal",
        "tol",
        "converged",
        "rel_residual",
        "x_norm",
        "diag_min",
        "diag_max",
        "diag_ratio",
        "wall_s",
        "n_free",
        "nnz",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Minimal floor summary: first rho_min that reaches the target tolerance.
    crit_rows = []
    for seed in seeds:
        for probability in probabilities:
            subset = [
                row for row in rows
                if row["seed"] == seed and abs(row["solid_probability"] - probability) < 1e-15
            ]
            subset.sort(key=lambda row: row["rho_min"])
            admissible = [row for row in subset if row["converged"]]
            crit_rows.append(
                {
                    "preset": args.preset,
                    "seed": seed,
                    "solid_probability": probability,
                    "critical_rho_min": admissible[0]["rho_min"] if admissible else "",
                    "critical_rel_residual": admissible[0]["rel_residual"] if admissible else "",
                    "critical_x_norm": admissible[0]["x_norm"] if admissible else "",
                    "all_failed": int(not admissible),
                }
            )
    crit_path = out_dir / "direct_floor_critical.csv"
    with crit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "preset",
                "seed",
                "solid_probability",
                "critical_rho_min",
                "critical_rel_residual",
                "critical_x_norm",
                "all_failed",
            ],
        )
        writer.writeheader()
        writer.writerows(crit_rows)

    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {crit_path}", flush=True)


if __name__ == "__main__":
    main()
