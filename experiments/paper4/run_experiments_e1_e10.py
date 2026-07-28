"""
Phase 2 experiment runner for paper4 E1 through E10.

E1  Heterogeneous 27-case cantilever sweep: outer iteration count vs mesh size.
E2  Per-linear-solve wall time: Jacobi-PCG vs FP32-GMG/BF16-GMG.
E3  Archived fixed-penalty 30-step OC schedule: same-schedule execution speedup.
E4  Fine-operator proxy throughput and roofline context.
E5  Heterogeneous admissibility sweep: eps_BF16 * kappa_eff over 18 cases.
E6  Ablations: (a) FP64 vs FP32, (b) FP32 depth sweep, (c) V vs W cycle,
               (d) Chebyshev vs Jacobi smoother, plus the F15 sensitivity screen.
E7  Large-scale single solves at 125k / 512k / 1M elements.
E8  External post-assembly baseline: CPU PyAMG-SA vs GPU FP32-GMG.
E9  Approximate energy from NVML power sampling.
E10 Robustness edges: document GMG failure modes.

Usage:
    python run_experiments_e1_e10.py [--experiments E1 E3 E6a] [--out dir]
    python run_experiments_e1_e10.py --experiments all
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path


# ── env bootstrap ──────────────────────────────────────────────────────────────

def _prefer_pytorch_env() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_tmp = root / "tmp" / "phase5_runtime_tmp"
    cupy_cache = root / "tmp" / "phase5_cupy_cache"
    for d in [runtime_tmp, cupy_cache]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "TMP": str(runtime_tmp),
        "TEMP": str(runtime_tmp),
        "CUPY_CACHE_DIR": str(cupy_cache),
        "CUPY_TEMPDIR": str(runtime_tmp),
        "CUPY_CACHE_IN_MEMORY": "1",
    })
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

from gpu_fem.bc_generator import generate_bc
from gpu_fem.presets import get_preset
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _edof_table_3d, _build_sparse_indices

OUT_DIR = Path(__file__).parent
N_WARMUP = int(os.environ.get("PAPER4_N_WARMUP", "2"))
N_TRIALS = int(os.environ.get("PAPER4_N_TRIALS", "10"))


# ── shared helpers ─────────────────────────────────────────────────────────────

def _vram_mb() -> float:
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return (total - free) / 1024**2
    except Exception:
        return float("nan")


def _sync():
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def _free_gpu():
    """Release all cached GPU memory blocks back to the OS between experiments."""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _build_components(preset_name: str, *, n_levels: int = 4,
                       fine_smoother: str = "fp32",
                       smoother_type: str = "chebyshev",
                       level_precisions=None,
                       cycle_type: str = "v",
                       coarse_correction_policy: str = "none",
                       coarse_correction_acceptance_ratio: float = 1.0,
                       coarse_correction_max_scale: float = 1.0,
                       root_enrichment_mode: str = "none",
                       root_enrichment_weight_floor: float = 1e-6,
                       root_enrichment_weight_power: float = 0.5,
                       root_local_correction_mode: str = "none",
                       root_local_correction_max_scale: float = 1.0,
                       root_local_correction_inner_steps: int = 10,
                       root_local_correction_inner_restart=None,
                       inner_krylov_steps: int = 0,
                       inner_krylov_restart=None,
                       fused_op=None):
    import cupy as cp
    from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
    from gpu_fem.solver_v2 import MatrixFreeKff

    spec = get_preset(preset_name)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    edof_gpu = cp.asarray(_edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32))
    free_gpu = cp.asarray(free)
    F_free_gpu = cp.asarray(bc.F[free].astype(np.float64))

    mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                           free_gpu=free_gpu, n_free=len(free), ndof=bc.ndof)
    gmg = GalerkinMatFreeGMG(
        mf_op=mf_op, free=free, free_gpu=free_gpu,
        nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
        KE_UNIT=KE_UNIT_3D, n_levels=n_levels,
        fine_smoother=fine_smoother, smoother_type=smoother_type,
        level_precisions=level_precisions, cycle_type=cycle_type,
        coarse_correction_policy=coarse_correction_policy,
        coarse_correction_acceptance_ratio=coarse_correction_acceptance_ratio,
        coarse_correction_max_scale=coarse_correction_max_scale,
        root_enrichment_mode=root_enrichment_mode,
        root_enrichment_weight_floor=root_enrichment_weight_floor,
        root_enrichment_weight_power=root_enrichment_weight_power,
        root_local_correction_mode=root_local_correction_mode,
        root_local_correction_max_scale=root_local_correction_max_scale,
        root_local_correction_inner_steps=root_local_correction_inner_steps,
        root_local_correction_inner_restart=root_local_correction_inner_restart,
        inner_krylov_steps=inner_krylov_steps,
        inner_krylov_restart=inner_krylov_restart,
        fused_op=fused_op,
    )
    return spec, bc, free, free_gpu, edof_gpu, F_free_gpu, mf_op, gmg


def _E_e(spec, rho_scalar: float = 0.5, penal: float = 3.0, E0=1.0, Emin=1e-9):
    """Uniform-density SIMP stiffness; retained for experiments that
    specifically test uniform fields (E2, E4, E6 ablations)."""
    import cupy as cp
    n = spec.nelx * spec.nely * spec.nelz
    rho = cp.full(n, rho_scalar, dtype=cp.float64)
    return Emin + (E0 - Emin) * rho**penal


def _E_e_heterogeneous(spec, volfrac: float = 0.5, penal: float = 3.0,
                        rho_min: float = 0.01, rho_max: float = 1.0,
                        seed: int = 0, E0: float = 1.0, Emin: float = 1e-9):
    """Binary-contrast heterogeneous density field: each element is independently
    rho_max with probability volfrac, rho_min otherwise. Mimics a late-SIMP
    state where most of the field has saturated. Used by E1 and E5 so that
    the (volfrac, penal) sweep probes genuine high-contrast distributions
    rather than a global scalar multiplier.

    The archived paper4 E1/E5 CSVs were generated with rho_min=1e-2, so that
    remains the default here. Experiments that target lower floors override
    rho_min explicitly.
    """
    import cupy as cp
    n = spec.nelx * spec.nely * spec.nelz
    rng = np.random.default_rng(seed)
    rho_np = np.where(rng.random(n) < volfrac, rho_max, rho_min).astype(np.float64)
    rho = cp.asarray(rho_np)
    return Emin + (E0 - Emin) * rho**penal


def _pcg(A_op, b, M_op, tol=1e-6, maxiter=200, history=None):
    from gpu_fem.solver_v2 import _cupy_pcg
    return _cupy_pcg(A_op, b, M_op, tol=tol, maxiter=maxiter, history=history)


def _fgmres(A_op, b, M_op, tol=1e-6, maxiter=200, restart=32, history=None):
    from gpu_fem.multigrid_v4 import _cupy_fgmres
    return _cupy_fgmres(A_op, b, M_op, tol=tol, maxiter=maxiter, restart=restart, history=history)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def _write_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _repeat_measure(run_once, *, n_warmup: int = N_WARMUP, n_trials: int = N_TRIALS):
    for _ in range(n_warmup):
        run_once()
    records = []
    for _ in range(n_trials):
        _sync()
        t0 = time.perf_counter()
        rec = dict(run_once())
        _sync()
        rec["wall_s"] = time.perf_counter() - t0
        records.append(rec)
    return records


def _tag_trial_rows(records, **meta):
    rows = []
    trial_rows_all = []
    for i, rec in enumerate(records, start=1):
        row = dict(meta)
        row["trial"] = i
        row.update(rec)
        rows.append(row)
    return rows


def _avg_power_interval(samples, t_start: float, t_end: float) -> float:
    """Average timestamped NVML power samples over a target interval."""
    if t_end <= t_start or not samples:
        return float("nan")
    pts = sorted((float(t), float(w)) for t, w in samples)
    if len(pts) == 1:
        return pts[0][1]
    times = np.array([t for t, _ in pts], dtype=np.float64)
    watts = np.array([w for _, w in pts], dtype=np.float64)
    interior = times[(times > t_start) & (times < t_end)]
    grid = np.concatenate(([t_start], interior, [t_end]))
    watts_grid = np.interp(grid, times, watts, left=watts[0], right=watts[-1])
    return float(np.trapezoid(watts_grid, grid) / (t_end - t_start))


def _history_rows(label: str, group: str, history: list[float]):
    return [
        {"group": group, "solver": label, "iter": i, "rel_residual": float(val)}
        for i, val in enumerate(history)
    ]


# ── E1: V-cycle iteration count ───────────────────────────────────────────────

def e1_vcycle_iteration_count():
    """E1: outer iterations across the heterogeneous 27-case cantilever sweep."""
    import cupy as cp
    _free_gpu()
    print("\n[E1] V-cycle iteration count")
    presets = [
        ("cantilever_gpu_medium",  "cantilever", "64k"),
        ("cantilever_gpu_large",   "cantilever", "216k"),
        ("cantilever_gpu_xlarge",  "cantilever", "512k"),
    ]
    penals = [1.5, 3.0, 4.5]
    rhos   = [0.8, 0.5, 0.2]    # early / mid / late SIMP
    rows = []

    for preset_name, bench, size_label in presets:
        _free_gpu()
        for rho in rhos:
            for penal in penals:
                spec, bc, free, free_gpu, edof_gpu, F_free_gpu, mf_op, gmg = \
                    _build_components(preset_name, fine_smoother="fp64")
                # Heterogeneous (binary-contrast) field: probes real late-SIMP
                # distributions rather than a global scalar density multiplier.
                E_e = _E_e_heterogeneous(spec, volfrac=rho, penal=penal, seed=42)
                gmg.setup(E_e)
                def A_op(v): return mf_op.matvec(v, E_e)
                x, iters, conv = _pcg(A_op, F_free_gpu, gmg.apply, tol=1e-6)
                rows.append({
                    "benchmark": bench, "size": size_label,
                    "n_elem": spec.nelx * spec.nely * spec.nelz,
                    "volfrac": rho, "penal": penal,
                    "iters": iters, "converged": int(conv),
                })
                print(f"  {bench} {size_label} vf={rho} p={penal}  iters={iters} conv={conv}")

    _write_csv(OUT_DIR / "e1_vcycle_iters.csv", rows)
    return rows


# ── E2: Per-linear-solve wall time ────────────────────────────────────────────

def e2_per_solve_wall_time():
    """E2: Paper3 FP32 Jacobi-PCG vs BF16-GMG (or FP32-GMG) wall time."""
    import cupy as cp
    _free_gpu()
    print("\n[E2] Per-linear-solve wall time")
    presets = [
        ("cantilever_gpu_medium",  "64k"),
        ("cantilever_gpu_large",   "216k"),
        ("cantilever_gpu_xlarge",  "512k"),
    ]
    rows = []
    trial_rows = []
    history_rows = []

    for preset_name, size_label in presets:
        _free_gpu()
        spec = get_preset(preset_name)
        bc = generate_bc(spec)
        free = bc.free_dofs.astype(np.int32)
        edof_gpu = cp.asarray(_edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32))
        free_gpu = cp.asarray(free)
        F_free_gpu = cp.asarray(bc.F[free].astype(np.float64))
        E_e = _E_e(spec)

        # Paper3 baseline: Jacobi-preconditioned PCG (diagonal preconditioner)
        from gpu_fem.solver_v2 import MatrixFreeKff
        mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                               free_gpu=free_gpu, n_free=len(free), ndof=bc.ndof)
        diag = mf_op.extract_diagonal(E_e)
        diag_inv = 1.0 / cp.where(cp.abs(diag) > 1e-12, diag, cp.ones_like(diag))

        def A_op(v): return mf_op.matvec(v, E_e)
        def M_jac(v): return diag_inv * v

        def _run_jac():
            _, iters, conv = _pcg(A_op, F_free_gpu, M_jac, tol=1e-6)
            return {"iters": iters, "converged": int(conv)}

        jac_trials = _repeat_measure(
            _run_jac
        )
        trial_rows.extend(_tag_trial_rows(
            jac_trials,
            size=size_label,
            n_elem=spec.nelx * spec.nely * spec.nelz,
            solver="Jacobi-PCG",
        ))
        # Re-run once with history capture to avoid timing perturbation.
        jac_hist = []
        _, iters_j_hist, conv_j_hist = _pcg(A_op, F_free_gpu, M_jac, tol=1e-6, history=jac_hist)
        history_rows.extend(_history_rows("Jacobi-PCG", size_label, jac_hist))
        t_jac_mean, t_jac_std = _mean_std([r["wall_s"] for r in jac_trials])
        it_jac_mean, it_jac_std = _mean_std([r["iters"] for r in jac_trials])

        # GMG FP32
        from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
        gmg32 = GalerkinMatFreeGMG(
            mf_op=mf_op, free=free, free_gpu=free_gpu,
            nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
            KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp32",
        )
        gmg32.setup(E_e)
        def _run_fp32():
            _, iters, conv = _pcg(A_op, F_free_gpu, gmg32.apply, tol=1e-6)
            return {"iters": iters, "converged": int(conv)}
        fp32_trials = _repeat_measure(
            _run_fp32
        )
        trial_rows.extend(_tag_trial_rows(
            fp32_trials,
            size=size_label,
            n_elem=spec.nelx * spec.nely * spec.nelz,
            solver="FP32-GMG",
        ))
        fp32_hist = []
        _, iters_g_hist, conv_g_hist = _pcg(A_op, F_free_gpu, gmg32.apply, tol=1e-6, history=fp32_hist)
        history_rows.extend(_history_rows("FP32-GMG", size_label, fp32_hist))
        t_gmg_mean, t_gmg_std = _mean_std([r["wall_s"] for r in fp32_trials])
        it_gmg_mean, it_gmg_std = _mean_std([r["iters"] for r in fp32_trials])

        speedup = t_jac_mean / max(t_gmg_mean, 1e-9)
        row = {
            "size": size_label, "n_elem": spec.nelx * spec.nely * spec.nelz,
            "n_warmup": N_WARMUP, "n_trials": N_TRIALS,
            "t_jacobi_pcg_s": t_jac_mean, "t_jacobi_pcg_std_s": t_jac_std,
            "iters_jacobi": it_jac_mean, "iters_jacobi_std": it_jac_std,
            "t_gmg_fp32_s": t_gmg_mean, "t_gmg_fp32_std_s": t_gmg_std,
            "iters_gmg_fp32": it_gmg_mean, "iters_gmg_fp32_std": it_gmg_std,
            "speedup_gmg_fp32": speedup,
        }

        # BF16 path
        try:
            from gpu_fem.cuda_fused_matvec import FusedMatvec
            fused = FusedMatvec(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                                ndof=bc.ndof)
            if fused._bf16_available:
                gmg16 = GalerkinMatFreeGMG(
                    mf_op=mf_op, free=free, free_gpu=free_gpu,
                    nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
                    KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="bf16", fused_op=fused,
                )
                gmg16.setup(E_e)
                def _run_bf16():
                    _, iters, conv = _fgmres(A_op, F_free_gpu, gmg16.apply, tol=1e-6)
                    return {"iters": iters, "converged": int(conv)}
                bf16_trials = _repeat_measure(
                    _run_bf16
                )
                trial_rows.extend(_tag_trial_rows(
                    bf16_trials,
                    size=size_label,
                    n_elem=spec.nelx * spec.nely * spec.nelz,
                    solver="BF16-GMG",
                ))
                bf16_hist = []
                _, iters16_hist, conv16_hist = _fgmres(
                    A_op, F_free_gpu, gmg16.apply, tol=1e-6, history=bf16_hist
                )
                history_rows.extend(_history_rows("BF16-GMG", size_label, bf16_hist))
                t_bf16_mean, t_bf16_std = _mean_std([r["wall_s"] for r in bf16_trials])
                it_bf16_mean, it_bf16_std = _mean_std([r["iters"] for r in bf16_trials])
                row["t_gmg_bf16_s"] = t_bf16_mean
                row["t_gmg_bf16_std_s"] = t_bf16_std
                row["iters_gmg_bf16"] = it_bf16_mean
                row["iters_gmg_bf16_std"] = it_bf16_std
                row["speedup_gmg_bf16"] = t_jac_mean / max(t_bf16_mean, 1e-9)
        except Exception as exc:
            row["bf16_error"] = str(exc)

        rows.append(row)
        print(
            f"  {size_label}: Jacobi={t_jac_mean:.3f}±{t_jac_std:.3f}s/{it_jac_mean:.1f}it  "
            f"GMG-FP32={t_gmg_mean:.3f}±{t_gmg_std:.3f}s/{it_gmg_mean:.1f}it  "
            f"speedup={speedup:.2f}x"
        )

        try:
            del gmg32
        except Exception:
            pass
        try:
            del gmg16
        except Exception:
            pass
        try:
            del fused
        except Exception:
            pass
        del mf_op, diag, diag_inv, edof_gpu, free_gpu, F_free_gpu, E_e
        gc.collect()
        _free_gpu()

    _write_csv(OUT_DIR / "e2_per_solve_wall_time.csv", rows)
    _write_csv(OUT_DIR / "e2_per_solve_wall_time_trials.csv", trial_rows)
    _write_csv(OUT_DIR / "e2_residual_histories.csv", history_rows)
    return rows


# ── E3: End-to-end SIMP speedup ───────────────────────────────────────────────

def e3_simp_speedup(n_simp_iters: int = 30):
    """E3: End-to-end SIMP-{n_simp_iters} speedup with the current solver-default
    linear-solve tolerances in both stacks."""
    _free_gpu()
    print(f"\n[E3] End-to-end SIMP-{n_simp_iters} speedup")
    from gpu_fem.solver_v2 import SolverV2
    from gpu_fem.solver_v4 import SolverV4

    presets = [
        ("cantilever_gpu_large", "cantilever_216k"),
        ("torsion_3d",           "torsion_small"),
        ("mbb_3d",               "mbb_small"),
    ]
    rows = []
    traj_rows = []

    def _oc_update(rho, dc, volfrac, move=0.15):
        lam_lo, lam_hi = 1e-40, 1e40
        for _ in range(100):
            lmid = 0.5 * (lam_lo + lam_hi)
            rho_new = np.clip(
                rho * np.sqrt(np.maximum(-dc / lmid, 0.0)),
                np.maximum(rho - move, 1e-3),
                np.minimum(rho + move, 1.0),
            )
            if rho_new.mean() > volfrac:
                lam_lo = lmid
            else:
                lam_hi = lmid
        return rho_new

    for preset_name, label in presets:
        _free_gpu()
        spec = get_preset(preset_name)
        bc = generate_bc(spec)
        edof = _edof_table_3d(spec.nelx, spec.nely, spec.nelz)
        row_idx, col_idx = _build_sparse_indices(edof)
        free = bc.free_dofs.astype(np.int32)
        n_elem = spec.nelx * spec.nely * spec.nelz

        common_kwargs = dict(
            edof=edof, row_idx=row_idx, col_idx=col_idx,
            KE_UNIT=KE_UNIT_3D, free=free, F=bc.F, ndof=bc.ndof,
            backend="cupy", enable_warm_start=True,
        )

        # Paper3 baseline: FP32 Jacobi-PCG (no GMG)
        sv2 = SolverV2(
            enable_matrix_free=True,
            enable_mixed_precision=True,
            enable_fused_cuda=True,
            **common_kwargs,
        )
        rho = np.full(n_elem, spec.volfrac)
        t0 = time.perf_counter()
        compliance_v2_k0 = None
        compliance_v2_final = None
        for k in range(n_simp_iters):
            c, dc = sv2.solve(rho, penal=3.0)
            if k == 0:
                compliance_v2_k0 = c
            compliance_v2_final = c
            traj_rows.append({
                "preset": label, "solver": "paper3_jacobi", "step": k,
                "compliance": c, "outer_iters": int(getattr(sv2, "last_cg_iters", -1)),
            })
            rho = _oc_update(rho, dc, spec.volfrac)
        t_v2 = time.perf_counter() - t0

        # SolverV4 with FP32-GMG
        sv4 = SolverV4(
            enable_matrix_free=True,
            enable_fused_cuda=True,
            enable_matfree_gmg=True,
            matfree_gmg_levels=4,
            gmg_fine_smoother="fp32",
            gmg_smoother_type="chebyshev",
            grid_dims=(spec.nelx, spec.nely, spec.nelz),
            **common_kwargs,
        )
        rho = np.full(n_elem, spec.volfrac)
        t0 = time.perf_counter()
        compliance_v4_k0 = None
        compliance_v4_final = None
        for k in range(n_simp_iters):
            c, dc = sv4.solve(rho, penal=3.0)
            if k == 0:
                compliance_v4_k0 = c
            compliance_v4_final = c
            traj_rows.append({
                "preset": label, "solver": "paper4_gmg_fp32", "step": k,
                "compliance": c, "outer_iters": int(getattr(sv4, "last_cg_iters", -1)),
            })
            rho = _oc_update(rho, dc, spec.volfrac)
        t_v4 = time.perf_counter() - t0

        speedup = t_v2 / max(t_v4, 1e-9)
        err_final = abs(compliance_v4_final - compliance_v2_final) \
            / max(abs(compliance_v2_final), 1e-300)
        err_k0 = abs(compliance_v4_k0 - compliance_v2_k0) \
            / max(abs(compliance_v2_k0), 1e-300)
        row = {
            "preset": label,
            "n_simp_iters": n_simp_iters,
            "t_paper3_s": t_v2,
            "t_gmg_fp32_s": t_v4,
            "speedup": speedup,
            "compliance_ref_final": compliance_v2_final,
            "compliance_gmg_final": compliance_v4_final,
            "compliance_err_final_pct": err_final * 100,
            "compliance_ref_k0": compliance_v2_k0,
            "compliance_gmg_k0": compliance_v4_k0,
            "compliance_err_k0_pct": err_k0 * 100,
        }
        rows.append(row)
        print(f"  {label}: paper3={t_v2:.1f}s  GMG-FP32={t_v4:.1f}s  "
              f"speedup={speedup:.2f}x  err_final={err_final*100:.3f}%  "
              f"err_k0={err_k0*100:.3f}%")

    _write_csv(OUT_DIR / "e3_simp_speedup.csv", rows)
    _write_csv(OUT_DIR / "e3_simp_trajectory.csv", traj_rows)
    return rows


# ── E4: Tensor-core throughput ────────────────────────────────────────────────

def e4_tc_throughput():
    """E4: Realized tensor-core FLOP/s for BF16 smoother kernel."""
    import cupy as cp
    print("\n[E4] Tensor-core throughput (BF16 vs FP32 matvec)")

    try:
        from gpu_fem.cuda_fused_matvec import FusedMatvec
    except ImportError:
        print("  SKIP: FusedMatvec not available")
        return {}

    preset_name = "cantilever_gpu_large"
    spec = get_preset(preset_name)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    edof_gpu = cp.asarray(_edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32))
    free_gpu = cp.asarray(free)
    E_e = cp.full(spec.nelx * spec.nely * spec.nelz, 1.0, dtype=cp.float32)
    u = cp.ones(len(free), dtype=cp.float32)

    fused = FusedMatvec(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D), ndof=bc.ndof)

    n_rep = 200
    n_elem = spec.nelx * spec.nely * spec.nelz
    rows = []
    roof_rows = []
    measured = {}
    for dtype in (["fp32"] + (["bf16"] if fused._bf16_available else [])):
        # Warmup
        for _ in range(5):
            fused.matvec(u, E_e, free_gpu, dtype=dtype)
        _sync()
        t0 = time.perf_counter()
        for _ in range(n_rep):
            fused.matvec(u, E_e, free_gpu, dtype=dtype)
        _sync()
        elapsed = time.perf_counter() - t0

        n_elem = spec.nelx * spec.nely * spec.nelz
        # 2 * 24 * 24 FLOPs per element (matmul) × n_elem
        flops_per_call = 2 * 24 * 24 * n_elem
        gflops = flops_per_call * n_rep / elapsed / 1e9
        t_ms = elapsed / n_rep * 1e3
        rows.append({
            "dtype": dtype,
            "preset": preset_name,
            "n_elem": n_elem,
            "nelx": spec.nelx,
            "nely": spec.nely,
            "nelz": spec.nelz,
            "field_modulus": "uniform_E=1.0",
            "input_vector": "all_ones_free_dofs",
            "t_ms": t_ms,
            "gflops": gflops,
            "n_rep": n_rep,
        })
        measured[dtype] = {"t_ms": t_ms, "gflops": gflops}
        print(f"  {dtype:>4}: {t_ms:.3f} ms/call  {gflops:.1f} GFLOP/s")

    # Representative roofline data on the same 216k case.
    from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
    from gpu_fem.solver_v2 import MatrixFreeKff
    mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                          free_gpu=free_gpu, n_free=len(free), ndof=bc.ndof)
    gmg = GalerkinMatFreeGMG(
        mf_op=mf_op, free=free, free_gpu=free_gpu,
        nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
        KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp32",
    )
    gmg.setup(E_e.astype(cp.float64))
    level_stats = gmg.level_stats
    level1 = level_stats[1] if len(level_stats) > 1 else None

    # Fine-level smoother proxy from the fused matvec benchmark.
    # The roofline reference lines use RTX 4090 vendor-spec theoretical peaks
    # (see nvidia2022ada in the manuscript bibliography); they are qualitative
    # guide rails, not measured saturation ceilings.
    fine_bytes = n_elem * (24 * 4 + 24 * 4 + 24 * 4 + 4 + 24 * 2)
    fine_flops = 2 * 24 * 24 * n_elem
    roof_rows.append({
        "kernel": "fine_bf16_matvec",
        "precision": "bf16",
        "operational_intensity": fine_flops / max(fine_bytes, 1),
        "achieved_gflops": measured.get("bf16", measured["fp32"])["gflops"],
        "peak_compute_gflops": 1.32e6,
        "peak_bandwidth_gbs": 1008.0,
    })
    roof_rows.append({
        "kernel": "fine_fp32_matvec",
        "precision": "fp32",
        "operational_intensity": fine_flops / max(fine_bytes, 1),
        "achieved_gflops": measured["fp32"]["gflops"],
        "peak_compute_gflops": 82.0e3,
        "peak_bandwidth_gbs": 1008.0,
    })

    if level1 is not None:
        K1 = gmg._K_gpu[1]
        x1 = cp.ones(K1.shape[0], dtype=cp.float32)
        for _ in range(5):
            _ = gmg._K_fp32_gpu[1] @ x1 if gmg._K_fp32_gpu[1] is not None else K1 @ x1.astype(cp.float64)
        _sync()
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = gmg._K_fp32_gpu[1] @ x1 if gmg._K_fp32_gpu[1] is not None else K1 @ x1.astype(cp.float64)
        _sync()
        elapsed = time.perf_counter() - t0
        spmv_gflops = (2 * int(K1.nnz) * n_rep) / elapsed / 1e9
        value_bytes = (gmg._K_fp32_gpu[1].data.nbytes if gmg._K_fp32_gpu[1] is not None else K1.data.nbytes)
        index_bytes = (gmg._K_fp32_gpu[1].indices.nbytes + gmg._K_fp32_gpu[1].indptr.nbytes) \
            if gmg._K_fp32_gpu[1] is not None else (K1.indices.nbytes + K1.indptr.nbytes)
        vec_bytes = 2 * x1.nbytes
        spmv_bytes = value_bytes + index_bytes + vec_bytes
        roof_rows.append({
            "kernel": "level1_spmv",
            "precision": "fp32",
            "operational_intensity": (2 * int(K1.nnz)) / max(spmv_bytes, 1),
            "achieved_gflops": spmv_gflops,
            "peak_compute_gflops": 82.0e3,
            "peak_bandwidth_gbs": 1008.0,
        })

    rhs_c = cp.ones(gmg._n_free[gmg._n_levels - 1], dtype=cp.float64)
    for _ in range(5):
        _ = gmg._solve_coarsest(rhs_c)
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        _ = gmg._solve_coarsest(rhs_c)
    _sync()
    elapsed = time.perf_counter() - t0
    n_c = rhs_c.shape[0]
    if gmg._coarse_chol_mode == "dense":
        coarse_flops = 2 * n_c * n_c
        coarse_bytes = int(gmg._coarse_chol_L.nbytes + 2 * rhs_c.nbytes)
    else:
        Kc = gmg._K_gpu[gmg._n_levels - 1]
        coarse_flops = 2 * int(Kc.nnz) * gmg._coarse_pcg_iters
        coarse_bytes = int(Kc.data.nbytes + Kc.indices.nbytes + Kc.indptr.nbytes + 2 * rhs_c.nbytes)
    roof_rows.append({
        "kernel": "coarsest_solve",
        "precision": "fp64" if gmg._coarse_chol_mode == "dense" else "fp32",
        "operational_intensity": coarse_flops / max(coarse_bytes, 1),
        "achieved_gflops": coarse_flops * n_rep / elapsed / 1e9,
        "peak_compute_gflops": 1289.0,
        "peak_bandwidth_gbs": 1008.0,
    })

    _write_csv(OUT_DIR / "e4_tc_throughput.csv", rows)
    _write_csv(OUT_DIR / "e4_roofline.csv", roof_rows)
    return rows


# ── E5: kappa_eff empirical ───────────────────────────────────────────────────

def e5_kappa_eff():
    """E5: eps_BF16 * kappa_eff < 1 at every level/SIMP state."""
    import cupy as cp
    _free_gpu()
    print("\n[E5] kappa_eff empirical vs. theoretical bound")
    EPS_BF16 = 3.906e-3   # BF16 unit roundoff (2^{-8})

    presets = [
        ("cantilever_gpu_medium", "64k"),
        ("cantilever_gpu_large",  "216k"),
    ]
    penals = [1.5, 3.0, 4.5]
    rhos   = [0.8, 0.5, 0.2]
    rows = []

    for preset_name, size_label in presets:
        spec = get_preset(preset_name)
        bc = generate_bc(spec)
        free = bc.free_dofs.astype(np.int32)
        edof_gpu = cp.asarray(_edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32))
        free_gpu = cp.asarray(free)
        from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
        from gpu_fem.solver_v2 import MatrixFreeKff
        mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                               free_gpu=free_gpu, n_free=len(free), ndof=bc.ndof)

        for rho in rhos:
            for penal in penals:
                # Heterogeneous late-SIMP distribution (matches E1).
                E_e = _E_e_heterogeneous(spec, volfrac=rho, penal=penal, seed=42)
                gmg = GalerkinMatFreeGMG(
                    mf_op=mf_op, free=free, free_gpu=free_gpu,
                    nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
                    KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp64",
                )
                gmg.setup(E_e)
                kappa = gmg.estimate_kappa_eff(n_iter=25)
                lam_max = gmg._lambda_max_est
                bound = EPS_BF16 * kappa
                satisfied = bound < 1.0
                row = {
                    "preset": size_label, "volfrac": rho, "penal": penal,
                    "seed": 42,
                    "kappa_eff": kappa, "lam_max_fine": lam_max,
                    "eps_bf16": EPS_BF16, "eps_kappa": bound, "bound_lt_1": int(satisfied),
                }
                rows.append(row)
                print(f"  {size_label} vf={rho} p={penal}  kappa={kappa:.1f}  "
                      f"eps*kappa={bound:.4f} (<1: {satisfied})")

    _write_csv(OUT_DIR / "e5_kappa_eff.csv", rows)
    return rows


# ── E6: Ablations ─────────────────────────────────────────────────────────────

def e6_ablations():
    """E6: (a) FP64 vs FP32, (b) FP32 depth sweep, (c) V vs W, (d) Cheb vs Jac."""
    import cupy as cp
    _free_gpu()
    print("\n[E6] Ablations")
    preset_name = "cantilever_gpu_large"
    spec = get_preset(preset_name)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    edof_gpu = cp.asarray(_edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32))
    free_gpu = cp.asarray(free)
    F_free_gpu = cp.asarray(bc.F[free].astype(np.float64))
    E_e = _E_e(spec)
    rho_scalar = 0.5
    penal = 3.0
    sample_dt_s = 0.05

    from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG, _cupy_fgmres
    from gpu_fem.solver_v2 import MatrixFreeKff, _cupy_pcg
    mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                           free_gpu=free_gpu, n_free=len(free), ndof=bc.ndof)
    def A_op(v): return mf_op.matvec(v, E_e)

    all_rows = {}
    sensitivity_rows = []

    def _teardown_locals(*objs):
        for obj in objs:
            try:
                del obj
            except Exception:
                pass
        gc.collect()
        _free_gpu()

    # (a) FP64 vs FP32
    rows_a = []
    trial_rows_a = []
    for label, prec in [("fp64", "fp64"), ("fp32", "fp32")]:
        try:
            del gmg
        except Exception:
            pass
        gc.collect()
        _free_gpu()
        gmg = GalerkinMatFreeGMG(
            mf_op=mf_op, free=free, free_gpu=free_gpu,
            nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
            KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother=prec,
        )
        gmg.setup(E_e)
        trials = _repeat_measure(
            lambda: (lambda out: {"iters": out[1], "converged": int(out[2])})(
                _pcg(A_op, F_free_gpu, gmg.apply, tol=1e-6)
            )
        )
        trial_rows_a.extend(_tag_trial_rows(trials, config=label))
        t_mean, t_std = _mean_std([r["wall_s"] for r in trials])
        it_mean, it_std = _mean_std([r["iters"] for r in trials])
        rows_a.append({
            "config": label, "iters": it_mean, "iters_std": it_std,
            "time_s": t_mean, "time_std_s": t_std, "converged": int(all(r["converged"] for r in trials)),
            "n_trials": N_TRIALS, "n_warmup": N_WARMUP,
        })
        print(f"  E6a {label}: iters={it_mean:.1f}  t={t_mean:.3f}±{t_std:.3f}s")
    _teardown_locals(gmg)
    all_rows["E6a"] = rows_a
    _write_csv(OUT_DIR / "e6a_precision_ablation.csv", rows_a)
    _write_csv(OUT_DIR / "e6a_precision_ablation_trials.csv", trial_rows_a)

    # (b) FP32 depth sweep (how many levels keep FP32 before reverting to FP64)
    rows_b = []
    trial_rows_b = []
    for n_fp32_levels in range(1, 5):
        try:
            del gmg
        except Exception:
            pass
        gc.collect()
        _free_gpu()
        precs = ["fp32"] * min(n_fp32_levels, 4) + ["fp64"] * max(0, 4 - n_fp32_levels)
        gmg = GalerkinMatFreeGMG(
            mf_op=mf_op, free=free, free_gpu=free_gpu,
            nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
            KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp32",
            level_precisions=precs,
        )
        gmg.setup(E_e)
        trials = _repeat_measure(
            lambda: (lambda out: {"iters": out[1], "converged": int(out[2])})(
                _pcg(A_op, F_free_gpu, gmg.apply, tol=1e-6)
            )
        )
        trial_rows_b.extend(_tag_trial_rows(trials, n_fp32_levels=n_fp32_levels, precs=str(precs)))
        t_mean, t_std = _mean_std([r["wall_s"] for r in trials])
        it_mean, it_std = _mean_std([r["iters"] for r in trials])
        rows_b.append({
            "n_fp32_levels": n_fp32_levels, "precs": str(precs),
            "iters": it_mean, "iters_std": it_std,
            "time_s": t_mean, "time_std_s": t_std,
            "converged": int(all(r["converged"] for r in trials)),
            "n_trials": N_TRIALS, "n_warmup": N_WARMUP,
        })
        print(f"  E6b n_fp32_levels={n_fp32_levels} precs={precs}: iters={it_mean:.1f}  t={t_mean:.3f}±{t_std:.3f}s")
    _teardown_locals(gmg)
    all_rows["E6b"] = rows_b
    _write_csv(OUT_DIR / "e6b_depth_sweep.csv", rows_b)
    _write_csv(OUT_DIR / "e6b_depth_sweep_trials.csv", trial_rows_b)

    # (c) V-cycle vs W-cycle
    rows_c = []
    trial_rows_c = []
    for cycle in ["v", "w"]:
        try:
            del gmg
        except Exception:
            pass
        gc.collect()
        _free_gpu()
        gmg = GalerkinMatFreeGMG(
            mf_op=mf_op, free=free, free_gpu=free_gpu,
            nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
            KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp64", cycle_type=cycle,
        )
        gmg.setup(E_e)
        trials = _repeat_measure(
            lambda: (lambda out: {"iters": out[1], "converged": int(out[2])})(
                _pcg(A_op, F_free_gpu, gmg.apply, tol=1e-6)
            )
        )
        trial_rows_c.extend(_tag_trial_rows(trials, cycle=cycle))
        t_mean, t_std = _mean_std([r["wall_s"] for r in trials])
        it_mean, it_std = _mean_std([r["iters"] for r in trials])
        rows_c.append({
            "cycle": cycle, "iters": it_mean, "iters_std": it_std,
            "time_s": t_mean, "time_std_s": t_std,
            "converged": int(all(r["converged"] for r in trials)),
            "n_trials": N_TRIALS, "n_warmup": N_WARMUP,
        })
        print(f"  E6c {cycle}-cycle: iters={it_mean:.1f}  t={t_mean:.3f}±{t_std:.3f}s")
    _teardown_locals(gmg)
    all_rows["E6c"] = rows_c
    _write_csv(OUT_DIR / "e6c_vcycle_vs_wcycle.csv", rows_c)
    _write_csv(OUT_DIR / "e6c_vcycle_vs_wcycle_trials.csv", trial_rows_c)

    # (d) Chebyshev vs Jacobi
    rows_d = []
    trial_rows_d = []
    for smtype in ["chebyshev", "jacobi"]:
        for deg in [2, 4]:
            try:
                del gmg
            except Exception:
                pass
            gc.collect()
            _free_gpu()
            gmg = GalerkinMatFreeGMG(
                mf_op=mf_op, free=free, free_gpu=free_gpu,
                nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
                KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp64",
                smoother_type=smtype, fine_smoother_degree=deg,
            )
            gmg.setup(E_e)
            trials = _repeat_measure(
                lambda: (lambda out: {"iters": out[1], "converged": int(out[2])})(
                    _pcg(A_op, F_free_gpu, gmg.apply, tol=1e-6)
                )
            )
            trial_rows_d.extend(_tag_trial_rows(trials, smoother=smtype, degree=deg))
            t_mean, t_std = _mean_std([r["wall_s"] for r in trials])
            it_mean, it_std = _mean_std([r["iters"] for r in trials])
            rows_d.append({
                "smoother": smtype, "degree": deg, "iters": it_mean, "iters_std": it_std,
                "time_s": t_mean, "time_std_s": t_std,
                "converged": int(all(r["converged"] for r in trials)),
                "n_trials": N_TRIALS, "n_warmup": N_WARMUP,
            })
            print(f"  E6d {smtype} deg={deg}: iters={it_mean:.1f}  t={t_mean:.3f}±{t_std:.3f}s")
    _teardown_locals(gmg)
    all_rows["E6d"] = rows_d
    _write_csv(OUT_DIR / "e6d_smoother_type.csv", rows_d)
    _write_csv(OUT_DIR / "e6d_smoother_type_trials.csv", trial_rows_d)

    # Joint sensitivity surface on the representative 216k case.
    fused = None
    try:
        from gpu_fem.cuda_fused_matvec import FusedMatvec
        fused = FusedMatvec(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D), ndof=bc.ndof)
    except Exception:
        fused = None

    sensitivity_trial_rows = []
    for fine_smoother, outer_solver, levels in [
        ("fp32", "fgmres", [3, 4]),
        ("bf16", "fgmres", [3, 4]),
    ]:
        for degree in [1, 2, 4]:
            for n_levels in levels:
                for restart in [16, 32, 50]:
                    try:
                        del gmg
                    except Exception:
                        pass
                    gc.collect()
                    _free_gpu()
                    if fine_smoother == "bf16" and fused is None:
                        continue
                    gmg = GalerkinMatFreeGMG(
                        mf_op=mf_op, free=free, free_gpu=free_gpu,
                        nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
                        KE_UNIT=KE_UNIT_3D, n_levels=n_levels,
                        fine_smoother=fine_smoother, smoother_type="chebyshev",
                        fine_smoother_degree=degree,
                        fused_op=fused if fine_smoother == "bf16" else None,
                    )
                    gmg.setup(E_e)
                    trials = _repeat_measure(
                        lambda: (lambda out: {"iters": out[1], "converged": int(out[2])})(
                            _fgmres(A_op, F_free_gpu, gmg.apply, tol=1e-6, restart=restart)
                        ),
                        n_warmup=1,
                        n_trials=3,
                    )
                    sensitivity_trial_rows.extend(_tag_trial_rows(
                        trials,
                        fine_smoother=fine_smoother,
                        degree=degree,
                        n_levels=n_levels,
                        restart=restart,
                    ))
                    t_mean, t_std = _mean_std([r["wall_s"] for r in trials])
                    it_mean, it_std = _mean_std([r["iters"] for r in trials])
                    sensitivity_rows.append({
                        "fine_smoother": fine_smoother,
                        "degree": degree,
                        "n_levels": n_levels,
                        "restart": restart,
                        "iters": it_mean,
                        "iters_std": it_std,
                        "time_s": t_mean,
                        "time_std_s": t_std,
                        "converged": int(all(r["converged"] for r in trials)),
                    })

    _write_csv(OUT_DIR / "e6_sensitivity_surface.csv", sensitivity_rows)
    _write_csv(OUT_DIR / "e6_sensitivity_surface_trials.csv", sensitivity_trial_rows)
    _teardown_locals(gmg, fused, mf_op, edof_gpu, free_gpu, F_free_gpu, E_e)
    return all_rows


# ── E7: Scaling to 8M+ ────────────────────────────────────────────────────────

def e7_large_scale():
    """E7: Single linear-solve timings at 125k / 512k / 1M with uniform modulus."""
    import cupy as cp
    _free_gpu()
    print("\n[E7] Large-scale single-solve timings (125k / 512k / 1M, FP32-GMG)")
    from gpu_fem.solver_v4 import SolverV4

    # Build large-mesh cases dynamically. The archived paper4 scaling study
    # reports only the three points below.
    scales = [
        (100, 50, 25,  125_000, "125k"),
        (160, 80, 40,  512_000, "512k"),
        (200, 100, 50, 1_000_000, "1M"),
    ]
    import gc
    rows = []
    trial_rows_all = []
    for nelx, nely, nelz, n_est, label in scales:
        gc.collect(); _free_gpu()
        trial_rows = []
        for _ in range(N_WARMUP + N_TRIALS):
            import cupy as cp
            from gpu_fem.solver_v2 import MatrixFreeKff
            from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG

            ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
            fixed = []
            for iy in range(nely + 1):
                for iz in range(nelz + 1):
                    nid = iy * (nelz + 1) + iz
                    fixed.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])
            fixed = np.unique(np.array(fixed, dtype=np.int32))
            free = np.setdiff1d(np.arange(ndof, dtype=np.int32), fixed)

            F_np = np.zeros(ndof)
            tip_node = nelx * (nely + 1) * (nelz + 1) + (nely // 2) * (nelz + 1) + (nelz // 2)
            F_np[3 * tip_node + 1] = -1.0
            F_free_gpu = cp.asarray(F_np[free])

            edof = _edof_table_3d(nelx, nely, nelz)
            edof_gpu = cp.asarray(edof.astype(np.int32))
            free_gpu = cp.asarray(free)
            # Table 7 is a uniform-modulus solve, not a SIMP field recovered
            # from (V_f, p).
            E_e = cp.full(nelx * nely * nelz, 0.5, dtype=cp.float64)

            mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                                   free_gpu=free_gpu, n_free=len(free), ndof=ndof)
            n_levels = 4 if nelx >= 80 else 3
            gmg = GalerkinMatFreeGMG(
                mf_op=mf_op, free=free, free_gpu=free_gpu,
                nelx=nelx, nely=nely, nelz=nelz,
                KE_UNIT=KE_UNIT_3D, n_levels=n_levels, fine_smoother="fp32",
            )
            vram_before = _vram_mb()
            t_setup0 = time.perf_counter()
            gmg.setup(E_e)
            t_setup = time.perf_counter() - t_setup0
            vram_after = _vram_mb()

            def A_op(v): return mf_op.matvec(v, E_e)
            _sync(); t0 = time.perf_counter()
            x, iters, conv = _fgmres(A_op, F_free_gpu, gmg.apply, tol=1e-6, maxiter=500, restart=50)
            _sync(); t_solve = time.perf_counter() - t0

            if len(trial_rows) >= N_WARMUP:
                trial_rows.append({
                    "t_setup_s": t_setup, "t_solve_s": t_solve,
                    "iters": iters, "converged": int(conv),
                    "vram_delta_mb": vram_after - vram_before,
                })
            else:
                trial_rows.append(None)

            del mf_op, gmg, edof_gpu, free_gpu, F_free_gpu, E_e, x
            gc.collect(); _free_gpu()

        trial_rows = [r for r in trial_rows if r is not None]
        trial_rows_all.extend(_tag_trial_rows(
            trial_rows,
            size=label,
            nelx=nelx,
            nely=nely,
            nelz=nelz,
            n_elem=nelx * nely * nelz,
            n_free=len(free),
        ))
        t_setup_mean, t_setup_std = _mean_std([r["t_setup_s"] for r in trial_rows])
        t_solve_mean, t_solve_std = _mean_std([r["t_solve_s"] for r in trial_rows])
        it_mean, it_std = _mean_std([r["iters"] for r in trial_rows])
        vram_mean, vram_std = _mean_std([r["vram_delta_mb"] for r in trial_rows])
        rows.append({
            "size": label, "nelx": nelx, "nely": nely, "nelz": nelz,
            "n_elem": nelx * nely * nelz, "n_free": len(free),
            "t_setup_s": t_setup_mean, "t_setup_std_s": t_setup_std,
            "t_solve_s": t_solve_mean, "t_solve_std_s": t_solve_std,
            "iters": it_mean, "iters_std": it_std,
            "converged": int(all(r["converged"] for r in trial_rows)),
            "vram_delta_mb": vram_mean, "vram_delta_std_mb": vram_std,
            "n_trials": N_TRIALS, "n_warmup": N_WARMUP,
        })
        print(f"  {label}: setup={t_setup_mean:.2f}±{t_setup_std:.2f}s  solve={t_solve_mean:.2f}±{t_solve_std:.2f}s  "
              f"iters={it_mean:.1f}  VRAM_delta={vram_mean:.0f}±{vram_std:.0f} MB")

    _write_csv(OUT_DIR / "e7_large_scale.csv", rows)
    _write_csv(OUT_DIR / "e7_large_scale_trials.csv", trial_rows_all)
    return rows


# ── E8: External baseline ─────────────────────────────────────────────────────

def e8_external_baseline():
    """E8: PyAMG smoothed-aggregation as CPU reference baseline."""
    import cupy as cp
    print("\n[E8] External baseline (PyAMG smoothed-aggregation AMG)")
    rows = []
    trial_rows_all = []

    try:
        import pyamg
    except ImportError:
        print("  SKIP: pyamg not installed")
        return rows

    from gpu_fem.pub_simp_solver import _build_sparse_indices
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import cg

    preset_name = "cantilever_gpu_medium"
    spec = get_preset(preset_name)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    n_elem = spec.nelx * spec.nely * spec.nelz

    rho_arr = np.full(n_elem, 0.5)
    E_arr = 1e-9 + (1.0 - 1e-9) * rho_arr**3.0

    edof = _edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    data = np.tile(KE_UNIT_3D.ravel(), n_elem) * np.repeat(E_arr, 576)
    ndof = 3 * (spec.nelx + 1) * (spec.nely + 1) * (spec.nelz + 1)
    import scipy.sparse as sp
    K = csr_matrix((data, (row_idx, col_idx)), shape=(ndof, ndof))
    K.sum_duplicates()
    Kff = K[free][:, free].tocsr()
    Ff = bc.F[free]

    py_trials = []
    for _ in range(N_WARMUP + N_TRIALS):
        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(Kff)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_amg, info = cg(Kff, Ff, M=ml.aspreconditioner(), rtol=1e-6, maxiter=200)
        t_solve = time.perf_counter() - t0
        res = np.linalg.norm(Ff - Kff @ x_amg) / np.linalg.norm(Ff)
        if len(py_trials) >= N_WARMUP:
            py_trials.append({"t_build_s": t_build, "t_solve_s": t_solve, "rel_res": res, "info": info})
        else:
            py_trials.append(None)
    py_trials = [r for r in py_trials if r is not None]
    trial_rows_all.extend(_tag_trial_rows(py_trials, solver="PyAMG-SA", preset="64k"))
    py_build_mean, py_build_std = _mean_std([r["t_build_s"] for r in py_trials])
    py_solve_mean, py_solve_std = _mean_std([r["t_solve_s"] for r in py_trials])
    py_res_mean, py_res_std = _mean_std([r["rel_res"] for r in py_trials])

    rows.append({
        "solver": "PyAMG-SA",
        "preset": "64k",
        "t_build_s": py_build_mean, "t_build_std_s": py_build_std,
        "t_solve_s": py_solve_mean, "t_solve_std_s": py_solve_std,
        "rel_res": py_res_mean, "rel_res_std": py_res_std,
        "info": int(all(int(r["info"]) == 0 for r in py_trials)),
        "iters": float("nan"), "iters_std": float("nan"),
        "n_trials": N_TRIALS, "n_warmup": N_WARMUP,
    })
    print(f"  PyAMG SA: build={py_build_mean:.2f}±{py_build_std:.2f}s  solve={py_solve_mean:.2f}±{py_solve_std:.2f}s  res={py_res_mean:.2e}")

    # --- GMG reference row at the same 64k problem, measured end-to-end ---
    edof_gpu = cp.asarray(edof.astype(np.int32))
    free_gpu = cp.asarray(free)
    F_free_gpu = cp.asarray(Ff.astype(np.float64))
    E_e = _E_e(spec, rho_scalar=0.5, penal=3.0)

    from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
    from gpu_fem.solver_v2 import MatrixFreeKff
    KE_unit_gpu = cp.asarray(KE_UNIT_3D)
    gmg_trials = []
    for _ in range(N_WARMUP + N_TRIALS):
        _sync(); t0 = time.perf_counter()
        mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=KE_unit_gpu,
                              free_gpu=free_gpu, n_free=len(free), ndof=ndof)
        gmg = GalerkinMatFreeGMG(
            mf_op=mf_op, free=free, free_gpu=free_gpu,
            nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
            KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp32",
        )
        gmg.setup(E_e)
        _sync(); t_gmg_build = time.perf_counter() - t0

        def A_op(v): return mf_op.matvec(v, E_e)
        _sync(); t0 = time.perf_counter()
        x_gmg, iters_gmg, conv_gmg = _pcg(A_op, F_free_gpu, gmg.apply, tol=1e-6)
        _sync(); t_gmg_solve = time.perf_counter() - t0
        r_gmg = A_op(x_gmg) - F_free_gpu
        res_gmg = float(cp.linalg.norm(r_gmg) / cp.linalg.norm(F_free_gpu))
        if len(gmg_trials) >= N_WARMUP:
            gmg_trials.append({
                "t_build_s": t_gmg_build, "t_solve_s": t_gmg_solve,
                "rel_res": res_gmg, "iters": iters_gmg, "conv": int(conv_gmg),
            })
        else:
            gmg_trials.append(None)
        del gmg, mf_op, x_gmg, r_gmg
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    gmg_trials = [r for r in gmg_trials if r is not None]
    trial_rows_all.extend(_tag_trial_rows(gmg_trials, solver="FP32-GMG", preset="64k"))
    gmg_build_mean, gmg_build_std = _mean_std([r["t_build_s"] for r in gmg_trials])
    gmg_solve_mean, gmg_solve_std = _mean_std([r["t_solve_s"] for r in gmg_trials])
    gmg_res_mean, gmg_res_std = _mean_std([r["rel_res"] for r in gmg_trials])

    rows.append({
        "solver": "FP32-GMG",
        "preset": "64k",
        "t_build_s": gmg_build_mean, "t_build_std_s": gmg_build_std,
        "t_solve_s": gmg_solve_mean, "t_solve_std_s": gmg_solve_std,
        "rel_res": gmg_res_mean, "rel_res_std": gmg_res_std,
        "info": int(all(r["conv"] for r in gmg_trials)),
        "iters": _mean_std([r["iters"] for r in gmg_trials])[0],
        "iters_std": _mean_std([r["iters"] for r in gmg_trials])[1],
        "n_trials": N_TRIALS, "n_warmup": N_WARMUP,
    })
    print(f"  GMG FP32 : build={gmg_build_mean:.3f}±{gmg_build_std:.3f}s  solve={gmg_solve_mean:.3f}±{gmg_solve_std:.3f}s "
          f" res={gmg_res_mean:.2e}")

    _write_csv(OUT_DIR / "e8_external_baseline.csv", rows)
    _write_csv(OUT_DIR / "e8_external_baseline_trials.csv", trial_rows_all)
    return rows


# ── E9: Energy/power ──────────────────────────────────────────────────────────

def e9_energy():
    """E9: Approximate energy via GPU power × wall time (pynvml)."""
    import cupy as cp
    print("\n[E9] Energy/power measurement")
    rows = []
    trial_rows_all = []

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        def power_w(): return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        print("  SKIP: pynvml not available")
        return rows

    preset_name = "cantilever_gpu_large"
    spec = get_preset(preset_name)
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    edof_gpu = cp.asarray(_edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32))
    free_gpu = cp.asarray(free)
    F_free_gpu = cp.asarray(bc.F[free].astype(np.float64))
    E_e = _E_e(spec)
    rho_scalar = 0.5
    penal = 3.0
    sample_dt_s = 0.05

    from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
    from gpu_fem.solver_v2 import MatrixFreeKff, _cupy_pcg
    mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                           free_gpu=free_gpu, n_free=len(free), ndof=bc.ndof)
    def A_op(v): return mf_op.matvec(v, E_e)

    for label, prec in [("fp64-gmg", "fp64"), ("fp32-gmg", "fp32")]:
        try:
            del gmg
        except Exception:
            pass
        gc.collect()
        _free_gpu()
        gmg = GalerkinMatFreeGMG(
            mf_op=mf_op, free=free, free_gpu=free_gpu,
            nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
            KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother=prec,
        )
        gmg.setup(E_e)

        import threading
        trial_rows = []
        for _ in range(N_WARMUP + N_TRIALS):
            samples = [(time.perf_counter(), power_w())]
            stop_evt = threading.Event()

            def _sample():
                while not stop_evt.wait(sample_dt_s):
                    samples.append((time.perf_counter(), power_w()))

            sampler = threading.Thread(target=_sample, daemon=True)
            sampler.start()
            _sync(); t0 = time.perf_counter()
            x, iters, conv = _pcg(A_op, F_free_gpu, gmg.apply, tol=1e-6)
            _sync(); elapsed = time.perf_counter() - t0
            t1 = t0 + elapsed
            samples.append((time.perf_counter(), power_w()))
            stop_evt.set()
            sampler.join(timeout=0.5)
            avg_w = _avg_power_interval(samples, t0, t1)
            joules = avg_w * elapsed
            if len(trial_rows) >= N_WARMUP:
                trial_rows.append({
                    "config": label,
                    "preset": preset_name,
                    "n_elem": spec.nelx * spec.nely * spec.nelz,
                    "nelx": spec.nelx,
                    "nely": spec.nely,
                    "nelz": spec.nelz,
                    "rho": rho_scalar,
                    "penal": penal,
                    "sampling_dt_s": sample_dt_s,
                    "idle_subtracted": 0,
                    "t_s": elapsed,
                    "iters": iters,
                    "avg_power_w": avg_w,
                    "energy_j": joules,
                    "converged": int(conv),
                })
            else:
                trial_rows.append(None)

        trial_rows = [r for r in trial_rows if r is not None]
        trial_rows_all.extend(_tag_trial_rows(
            trial_rows,
            config=label,
            preset=preset_name,
            n_elem=spec.nelx * spec.nely * spec.nelz,
            nelx=spec.nelx,
            nely=spec.nely,
            nelz=spec.nelz,
            rho=rho_scalar,
            penal=penal,
            sampling_dt_s=sample_dt_s,
            idle_subtracted=0,
        ))
        t_mean, t_std = _mean_std([r["t_s"] for r in trial_rows])
        it_mean, it_std = _mean_std([r["iters"] for r in trial_rows])
        p_mean, p_std = _mean_std([r["avg_power_w"] for r in trial_rows])
        e_mean, e_std = _mean_std([r["energy_j"] for r in trial_rows])

        rows.append({
            "config": label,
            "preset": preset_name,
            "n_elem": spec.nelx * spec.nely * spec.nelz,
            "nelx": spec.nelx,
            "nely": spec.nely,
            "nelz": spec.nelz,
            "rho": rho_scalar,
            "penal": penal,
            "sampling_dt_s": sample_dt_s,
            "idle_subtracted": 0,
            "t_s": t_mean,
            "t_std_s": t_std,
            "iters": it_mean,
            "iters_std": it_std,
            "avg_power_w": p_mean,
            "avg_power_std_w": p_std,
            "energy_j": e_mean,
            "energy_std_j": e_std,
            "converged": int(all(r["converged"] for r in trial_rows)),
            "n_trials": N_TRIALS,
            "n_warmup": N_WARMUP,
        })
        print(f"  {label}: t={t_mean:.3f}±{t_std:.3f}s  P_avg={p_mean:.1f}±{p_std:.1f}W  E={e_mean:.2f}±{e_std:.2f}J")

    try:
        del gmg
    except Exception:
        pass
    del mf_op, edof_gpu, free_gpu, F_free_gpu, E_e
    gc.collect()
    _free_gpu()
    _write_csv(OUT_DIR / "e9_energy.csv", rows)
    _write_csv(OUT_DIR / "e9_energy_trials.csv", trial_rows_all)
    return rows


# ── E10: Robustness edges ─────────────────────────────────────────────────────

def e10_robustness_edges():
    """E10: Document where GMG fails (high-contrast, aggressive continuation, etc.)."""
    import cupy as cp
    _free_gpu()
    from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
    from gpu_fem.solver_v2 import MatrixFreeKff
    print("\n[E10] Robustness edge cases")

    spec = get_preset("cantilever_gpu_medium")
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    edof_gpu = cp.asarray(_edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32))
    free_gpu = cp.asarray(free)
    F_free_gpu = cp.asarray(bc.F[free].astype(np.float64))
    n_elem = spec.nelx * spec.nely * spec.nelz

    mf_op = MatrixFreeKff(edof_gpu=edof_gpu, KE_unit_gpu=cp.asarray(KE_UNIT_3D),
                           free_gpu=free_gpu, n_free=len(free), ndof=bc.ndof)

    eps_bf16 = 3.906e-3
    cases = [
        ("uniform-vf0.2",            {"volfrac": 0.2, "rho_min": 0.2,  "penal": 3.0, "seed": "deterministic"}, lambda: _E_e(spec, rho_scalar=0.2, penal=3.0)),
        ("uniform-vf0.5",            {"volfrac": 0.5, "rho_min": 0.5,  "penal": 3.0, "seed": "deterministic"}, lambda: _E_e(spec, rho_scalar=0.5, penal=3.0)),
        ("uniform-vf0.8",            {"volfrac": 0.8, "rho_min": 0.8,  "penal": 3.0, "seed": "deterministic"}, lambda: _E_e(spec, rho_scalar=0.8, penal=3.0)),
        ("binary-vf0.2-p1.5",        {"volfrac": 0.2, "rho_min": 1e-9, "penal": 1.5, "seed": 7}, lambda: _E_e_heterogeneous(spec, volfrac=0.2, penal=1.5, rho_min=1e-9, seed=7)),
        ("binary-vf0.5-p3.0",        {"volfrac": 0.5, "rho_min": 1e-9, "penal": 3.0, "seed": 11}, lambda: _E_e_heterogeneous(spec, volfrac=0.5, penal=3.0, rho_min=1e-9, seed=11)),
        ("binary-vf0.8-p4.5",        {"volfrac": 0.8, "rho_min": 1e-9, "penal": 4.5, "seed": 13}, lambda: _E_e_heterogeneous(spec, volfrac=0.8, penal=4.5, rho_min=1e-9, seed=13)),
        ("checkerboard",             {"volfrac": 0.5, "rho_min": 1e-9, "penal": 3.0, "seed": "deterministic"}, lambda: (lambda r: 1e-9 + (1.0 - 1e-9) * r**3.0)(
            cp.asarray((np.arange(n_elem) % 2).astype(np.float64)))),
        ("layered-band",             {"volfrac": 0.5, "rho_min": 1e-9, "penal": 3.0, "seed": "deterministic"}, lambda: (lambda r: 1e-9 + (1.0 - 1e-9) * r**3.0)(
            cp.asarray(np.repeat((np.arange(spec.nelx) < spec.nelx // 2).astype(np.float64), spec.nely * spec.nelz)))),
        ("rho-min-1e-12",            {"volfrac": 0.5, "rho_min": 1e-12, "penal": 3.0, "seed": 17}, lambda: _E_e_heterogeneous(spec, volfrac=0.5, penal=3.0, rho_min=1e-12, seed=17)),
        ("mixed-very-low",           {"volfrac": 0.1, "rho_min": 1e-12, "penal": 4.5, "seed": 19}, lambda: (lambda r: 1e-12 + (1.0 - 1e-12) * r**4.5)(
            cp.asarray(np.where(np.random.default_rng(19).random(n_elem) < 0.1, 1.0, 1e-12)))),
    ]

    rows = []
    for label, meta, E_fn in cases:
        try:
            E_e = E_fn()
            gmg = GalerkinMatFreeGMG(
                mf_op=mf_op, free=free, free_gpu=free_gpu,
                nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
                KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp64",
            )
            gmg.setup(E_e)
            def A_op(v): return mf_op.matvec(v, E_e)
            # FGMRES with the same outer budget as the manuscript; solver
            # label kept consistent for downstream text.
            x, iters, conv = _fgmres(A_op, F_free_gpu, gmg.apply,
                                      tol=1e-6, maxiter=500, restart=50)
            kappa = gmg.estimate_kappa_eff(n_iter=20)
            adapt = gmg.adaptive_params
            rows.append({
                "case": label, "solver": "FGMRES", "iters": iters,
                "converged": int(conv), "kappa_eff": kappa,
                "eps_kappa": eps_bf16 * kappa,
                "volfrac": meta["volfrac"], "rho_min": meta["rho_min"],
                "penal": meta["penal"], "seed": meta["seed"], "notes": "",
                "contrast_ratio": adapt["contrast_ratio"],
                "is_high_contrast": adapt["is_high_contrast"],
                "fine_degree_adapt": adapt["fine_degree_adapt"],
                "coarse_iters_adapt": adapt["coarse_iters_adapt"],
            })
            print(f"  {label:<30} iters={iters:4d}  conv={conv}  kappa={kappa:.1f}  "
                  f"CR={adapt['contrast_ratio']:.1f}  HC={adapt['is_high_contrast']}")
        except Exception as exc:
            rows.append({
                "case": label, "solver": "FGMRES", "iters": -1,
                "converged": 0, "kappa_eff": float("nan"),
                "eps_kappa": float("nan"),
                "volfrac": meta["volfrac"], "rho_min": meta["rho_min"],
                "penal": meta["penal"], "seed": meta["seed"], "notes": str(exc),
                "contrast_ratio": float("nan"),
                "is_high_contrast": False,
                "fine_degree_adapt": -1,
                "coarse_iters_adapt": -1,
            })
            print(f"  {label:<30} ERROR: {exc}")

    basin_rows = []
    for volfrac in [0.2, 0.4, 0.6, 0.8]:
        for rho_min in [1e-12, 1e-9, 1e-6]:
            for penal in [3.0, 4.5]:
                try:
                    E_e = _E_e_heterogeneous(
                        spec, volfrac=volfrac, penal=penal, rho_min=rho_min, seed=23
                    )
                    gmg = GalerkinMatFreeGMG(
                        mf_op=mf_op, free=free, free_gpu=free_gpu,
                        nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
                        KE_UNIT=KE_UNIT_3D, n_levels=4, fine_smoother="fp64",
                    )
                    gmg.setup(E_e)
                    def A_op(v): return mf_op.matvec(v, E_e)
                    _, iters, conv = _fgmres(A_op, F_free_gpu, gmg.apply,
                                             tol=1e-6, maxiter=300, restart=50)
                    kappa = gmg.estimate_kappa_eff(n_iter=15)
                    adapt = gmg.adaptive_params
                    basin_rows.append({
                        "volfrac": volfrac, "rho_min": rho_min, "penal": penal,
                        "seed": 23,
                        "iters": iters, "converged": int(conv),
                        "kappa_eff": kappa, "eps_kappa": eps_bf16 * kappa,
                        "contrast_ratio": adapt["contrast_ratio"],
                        "is_high_contrast": adapt["is_high_contrast"],
                        "fine_degree_adapt": adapt["fine_degree_adapt"],
                        "coarse_iters_adapt": adapt["coarse_iters_adapt"],
                    })
                except Exception as exc:
                    basin_rows.append({
                        "volfrac": volfrac, "rho_min": rho_min, "penal": penal,
                        "seed": 23,
                        "iters": -1, "converged": 0,
                        "kappa_eff": float("nan"), "eps_kappa": float("nan"),
                        "contrast_ratio": float("nan"),
                        "is_high_contrast": False,
                        "fine_degree_adapt": -1,
                        "coarse_iters_adapt": -1,
                    })

    _write_csv(OUT_DIR / "e10_robustness.csv", rows)
    _write_csv(OUT_DIR / "e10_basin.csv", basin_rows)
    return rows


# ── Dispatch ──────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "E1":  e1_vcycle_iteration_count,
    "E2":  e2_per_solve_wall_time,
    "E3":  e3_simp_speedup,
    "E4":  e4_tc_throughput,
    "E5":  e5_kappa_eff,
    "E6":  e6_ablations,
    "E7":  e7_large_scale,
    "E8":  e8_external_baseline,
    "E9":  e9_energy,
    "E10": e10_robustness_edges,
}


def main():
    global OUT_DIR   # declared first so all subsequent uses in this scope see it
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        help="E1 E2 ... or 'all'")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory")
    args = parser.parse_args()

    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(exist_ok=True)

    keys = list(EXPERIMENTS.keys()) if "all" in args.experiments else \
           [k.upper() for k in args.experiments]

    all_results = {}
    for key in keys:
        if key not in EXPERIMENTS:
            print(f"Unknown experiment {key}")
            continue
        try:
            r = EXPERIMENTS[key]()
            all_results[key] = r
        except Exception as exc:
            print(f"  ERROR in {key}: {exc}")
            all_results[key] = {"error": str(exc)}

    if set(keys) == set(EXPERIMENTS.keys()):
        _write_json(OUT_DIR / "results_e1_e10_summary.json", all_results)
    else:
        _write_json(OUT_DIR / "results_selected_summary.json", all_results)
    print(f"\nAll results written to {OUT_DIR}")


if __name__ == "__main__":
    main()
