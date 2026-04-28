"""
Paper 4 production solver.

SolverV4 keeps SolverV2 as the paper-3 baseline and adds a clean paper-4 path
for matrix-free full-Galerkin GMG.  The multigrid hierarchy lives in
`multigrid_v4.py`; this wrapper only integrates that hierarchy into the
existing SIMP and warm-start machinery.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .multigrid_v4 import GalerkinMatFreeGMG, _cupy_fgmres
from .solver_v2 import MatrixFreeKff, SolverV2, _cupy_pcg


class SolverV4(SolverV2):
    """
    Paper 4 drop-in solver.

    New flags
    ---------
    enable_matfree_gmg : bool
        Activate the paper-4 full-Galerkin matrix-free hierarchy.
    matfree_gmg_levels : int
        Number of multigrid levels including the fine level.
    gmg_fine_smoother : {"fp64", "fp32", "bf16"}
        Precision used by the fine-level smoother only.
    gmg_fine_degree : int
        Polynomial smoother degree per pre/post pass on the fine level.
    gmg_outer_solver : {"auto", "pcg", "fgmres"}
        "auto" selects PCG for fp64/fp32 smoothers and FGMRES for bf16.
    """

    def __init__(
        self,
        *args,
        enable_matfree_gmg: bool = False,
        matfree_gmg_levels: int = 4,
        gmg_fine_smoother: str = "fp32",
        gmg_fine_degree: int = 2,
        gmg_outer_solver: str = "auto",
        gmg_restart: int = 32,
        gmg_smoother_type: str = "chebyshev",
        gmg_cheb_lower_frac: float = 1.0 / 30.0,
        gmg_level_precisions=None,
        gmg_cycle_type: str = "v",
        **kwargs,
    ) -> None:
        self._enable_matfree_gmg = enable_matfree_gmg
        self._matfree_gmg_levels = matfree_gmg_levels
        self._gmg_fine_smoother = gmg_fine_smoother
        self._gmg_fine_degree = gmg_fine_degree
        self._gmg_outer_solver = gmg_outer_solver
        self._gmg_restart = gmg_restart
        self._gmg_n_smooth = kwargs.get("gmg_smooth_iters", 2)
        self._grid_dims = kwargs.get("grid_dims")
        self._gmg_smoother_type = gmg_smoother_type
        self._gmg_cheb_lower_frac = gmg_cheb_lower_frac
        self._gmg_level_precisions = gmg_level_precisions
        self._gmg_cycle_type = gmg_cycle_type

        if enable_matfree_gmg:
            if not kwargs.get("enable_matrix_free", False):
                raise ValueError("SolverV4 requires enable_matrix_free=True when enable_matfree_gmg=True.")
            if self._grid_dims is None:
                raise ValueError("SolverV4 requires grid_dims=(nelx, nely, nelz) for the GMG path.")
            if gmg_fine_smoother not in {"fp64", "fp32", "bf16"}:
                raise ValueError(
                    f"gmg_fine_smoother must be one of 'fp64', 'fp32', or 'bf16'; got {gmg_fine_smoother!r}."
                )
            if gmg_outer_solver not in {"auto", "pcg", "fgmres"}:
                raise ValueError(
                    f"gmg_outer_solver must be 'auto', 'pcg', or 'fgmres'; got {gmg_outer_solver!r}."
                )
            if gmg_fine_smoother == "bf16" and not kwargs.get("enable_fused_cuda", False):
                raise ValueError(
                    "SolverV4 BF16 fine smoothing requires enable_fused_cuda=True so the WMMA path is available."
                )
            if kwargs.get("enable_fused_cuda", False) and not kwargs.get("enable_mixed_precision", False):
                # SolverV2 only enables the fused-kernel hook when mixed precision is on.
                # SolverV4 uses that fused path only inside the fine-level smoother, so
                # force the flag here instead of modifying the paper-3 baseline.
                kwargs["enable_mixed_precision"] = True

        super().__init__(*args, **kwargs)

        self._matfree_gmg: Optional[GalerkinMatFreeGMG] = None
        self.last_outer_solver: str = ""
        self.last_multigrid_quality: dict = {}
        self.last_true_rel_residual: float = float("nan")
        self.last_cg_history: list[float] = []

    def _ensure_matfree_components(self) -> None:
        if self._matfree_op is None:
            self._matfree_op = MatrixFreeKff(
                edof_gpu=self._edof_gpu,
                KE_unit_gpu=self._KE_unit_gpu,
                free_gpu=self._free_gpu,
                n_free=self._n_free,
                ndof=self.ndof,
            )
            print(
                f"[SolverV4] MatrixFreeKff ready - n_elem={self._n_elem:,} "
                f"ke_size={self._matfree_op._ke_size} n_free={self._n_free:,} ndof={self.ndof:,}"
            )

        if self._enable_fused_cuda and self._fused_op is None:
            from .cuda_fused_matvec import FusedMatvec

            self._fused_op = FusedMatvec(
                edof_gpu=self._edof_gpu,
                KE_unit_gpu=self._KE_unit_gpu,
                ndof=self.ndof,
            )
            print("[SolverV4] FusedMatvec ready for paper-4 fine smoothing.")

        if self._matfree_gmg is None:
            from .pub_simp_solver import KE_UNIT_3D

            nelx, nely, nelz = self._grid_dims
            print(
                f"[SolverV4] Building GalerkinMatFreeGMG: {nelx}x{nely}x{nelz}, "
                f"{self._matfree_gmg_levels} levels, fine={self._gmg_fine_smoother}, "
                f"degree={self._gmg_fine_degree}"
            )
            self._matfree_gmg = GalerkinMatFreeGMG(
                mf_op=self._matfree_op,
                free=self._free,
                free_gpu=self._free_gpu,
                nelx=nelx,
                nely=nely,
                nelz=nelz,
                KE_UNIT=KE_UNIT_3D,
                n_levels=self._matfree_gmg_levels,
                coarse_smooth_iters=self._gmg_n_smooth,
                fine_smoother=self._gmg_fine_smoother,
                fine_smoother_degree=self._gmg_fine_degree,
                fused_op=self._fused_op if self._enable_fused_cuda else None,
                smoother_type=self._gmg_smoother_type,
                cheb_lower_frac=self._gmg_cheb_lower_frac,
                level_precisions=self._gmg_level_precisions,
                cycle_type=self._gmg_cycle_type,
            )

    def _solve_cupy_matfree(self, rho_phys: np.ndarray, penal: float) -> tuple:
        if not self._enable_matfree_gmg:
            return super()._solve_cupy_matfree(rho_phys, penal)

        import time as _time

        cp = self._cp

        def _sync_time():
            if self._enable_profiling:
                cp.cuda.Stream.null.synchronize()
            return _time.perf_counter()

        self._ensure_matfree_components()

        penal_jumped = self._prev_penal is not None and abs(penal - self._prev_penal) > 0.5
        if penal_jumped and self._u_prev_cupy is not None:
            self._u_prev_cupy = None
        self._prev_penal = penal

        rho_gpu = cp.asarray(rho_phys, dtype=cp.float64)
        E_e = self.Emin + (self.E0 - self.Emin) * rho_gpu ** penal

        t0 = _sync_time()
        fine_diag = self._matfree_op.extract_diagonal(E_e)
        t1 = _sync_time()

        self._matfree_gmg.setup(E_e, fine_diag=fine_diag)
        t2 = _sync_time()

        def _Kop(v):
            return self._matfree_op.matvec(v, E_e)

        x0 = (
            self._u_prev_cupy
            if (self._enable_warm_start and self._u_prev_cupy is not None)
            else None
        )
        F_cg = self._F_free_gpu

        outer_solver = self._gmg_outer_solver
        if outer_solver == "auto":
            outer_solver = "fgmres" if self._matfree_gmg.uses_bf16_smoother() else "pcg"
        self.last_outer_solver = outer_solver

        if outer_solver == "pcg":
            history: list[float] = []
            U_free_gpu, n_outer, converged = _cupy_pcg(
                _Kop,
                F_cg,
                self._matfree_gmg.apply,
                x0=x0,
                tol=self.cg_tol,
                maxiter=self.cg_maxiter,
                history=history,
            )
            if not converged and x0 is not None:
                history = []
                U_free_gpu, n_outer, converged = _cupy_pcg(
                    _Kop,
                    F_cg,
                    self._matfree_gmg.apply,
                    x0=None,
                    tol=self.cg_tol,
                    maxiter=self.cg_maxiter,
                    history=history,
                )
        else:
            history = []
            U_free_gpu, n_outer, converged = _cupy_fgmres(
                _Kop,
                F_cg,
                self._matfree_gmg.apply,
                x0=x0,
                tol=self.cg_tol,
                maxiter=self.cg_maxiter,
                restart=self._gmg_restart,
                history=history,
            )
            if not converged and x0 is not None:
                history = []
                U_free_gpu, n_outer, converged = _cupy_fgmres(
                    _Kop,
                    F_cg,
                    self._matfree_gmg.apply,
                    x0=None,
                    tol=self.cg_tol,
                    maxiter=self.cg_maxiter,
                    restart=self._gmg_restart,
                    history=history,
                )

        self.last_cg_iters = n_outer
        self.last_cg_history = list(history)
        b_norm = float(cp.linalg.norm(F_cg))
        self.last_true_rel_residual = float(cp.linalg.norm(F_cg - _Kop(U_free_gpu)) / max(b_norm, 1e-300))
        self.last_multigrid_quality = self._matfree_gmg.probe_quality(F_cg)
        if not converged:
            print(
                f"[SolverV4] WARNING: {outer_solver} did not converge in {self.cg_maxiter} iters "
                f"(n_free={self._n_free:,})."
            )

        t3 = _sync_time()

        if self._enable_warm_start:
            self._u_prev_cupy = U_free_gpu.copy()

        compliance = float(cp.dot(self._F_free_gpu, U_free_gpu).get())

        U_gpu = cp.zeros(self.ndof, dtype=cp.float64)
        U_gpu[self._free_gpu] = U_free_gpu
        Ue = U_gpu[self._edof_gpu]
        KUe = Ue @ self._KE_unit_gpu
        ce = (KUe * Ue).sum(axis=1)
        dc_phys_gpu = -penal * (self.E0 - self.Emin) * rho_gpu ** (penal - 1) * ce

        t4 = _sync_time()

        if self._enable_profiling:
            self.last_timing = {
                "assemble_ms": 0.0,
                "diag_ms": (t1 - t0) * 1e3,
                "gmg_setup_ms": (t2 - t1) * 1e3,
                "cg_ms": (t3 - t2) * 1e3,
                "sensitivity_ms": (t4 - t3) * 1e3,
                "total_ms": (t4 - t0) * 1e3,
                "used_gmg": True,
                "matrix_free": True,
                "outer_solver": outer_solver,
                "cg_iters": n_outer,
            }

        return compliance, cp.asnumpy(dc_phys_gpu)

    def invalidate(self) -> None:
        super().invalidate()
        if self._matfree_gmg is not None:
            self._matfree_gmg._is_setup = False
