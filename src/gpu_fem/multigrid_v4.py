"""
Paper 4 multigrid support.

This module isolates the matrix-free full-Galerkin hierarchy from the paper-3
production path.  The fine operator remains matrix-free; level 1 is assembled
elementwise as P^T K_f P without ever forming K_f; coarser levels are built by
exact sparse triple products.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from .solver_v2 import (
    MatrixFreeKff,
    NodeBlockJacobiMF,
    _build_scalar_prolongation,
    _coarse_free_dofs_injection,
    _fixed_pcg_coarse,
)


@dataclass
class MultigridLevelStats:
    level: int
    kind: str
    n_elem: int
    n_free: int
    nnz: int
    estimated_vram_bytes: int


@dataclass
class MultigridCycleVisit:
    visit_id: int
    level: int
    cycle_type: str
    is_coarsest: bool
    n_coarse_corrections: int
    rhs_norm: float
    jacobi_action_norm: float
    presmooth_solution_norm: float
    presmooth_residual_norm: float
    presmooth_residual_ratio: float
    restricted_rhs_norm: float
    restricted_rhs_ratio: float
    projector_norm: float
    coarse_space_gap_norm: float
    coarse_space_gap_ratio: float
    coarse_correction_norm: float
    prolongated_correction_norm: float
    correction_alignment: float
    corrected_residual_norm: float
    corrected_residual_vs_presmooth: float
    correction_scale_raw: Optional[float] = None
    correction_scale_applied: float = 1.0
    second_restricted_rhs_norm: Optional[float] = None
    second_coarse_correction_norm: Optional[float] = None
    second_prolongated_correction_norm: Optional[float] = None
    second_correction_alignment: Optional[float] = None
    second_correction_scale_raw: Optional[float] = None
    second_correction_scale_applied: Optional[float] = None
    second_corrected_residual_norm: Optional[float] = None
    second_corrected_residual_vs_corrected: Optional[float] = None
    postsmooth_solution_norm: float = 0.0
    postsmooth_residual_norm: float = 0.0
    postsmooth_residual_ratio: float = 0.0
    coarsest_mode: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultigridCycleTrace:
    cycle_type: str
    rhs_norm: float
    jacobi_action_norm: float
    output_norm: float
    z_over_jacobi: float
    rz: float
    pd: bool
    output_residual_norm: float
    output_residual_ratio: float
    contrast_ratio: float
    is_high_contrast: bool
    fine_degree_adapt: int
    coarse_iters_adapt: int
    coarse_correction_policy: str
    coarse_correction_acceptance_ratio: float
    coarse_correction_max_scale: float
    root_enrichment_mode: str
    root_enrichment_weight_floor: float
    root_enrichment_weight_power: float
    root_local_correction_mode: str
    root_local_correction_inner_steps: int
    root_local_correction_inner_restart: int
    root_local_correction_raw_scale: Optional[float]
    root_local_correction_applied_scale: float
    root_local_correction_residual_ratio: float
    inner_krylov_steps: int
    inner_krylov_restart: int
    visits: list[MultigridCycleVisit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class _GalerkinCoarseLevel:
    __slots__ = (
        "nelx",
        "nely",
        "nelz",
        "n_elem",
        "n_free",
        "Kff_indptr",
        "Kff_indices",
        "diag_pos",
        "diag_row",
        "Q_stu_sorted",
        "fine_elem_sorted_stu",
        "Q_stu_stack",          # (8, nnz_raw) stacked for vectorised assembly
        "fine_elem_stack",      # (8, nnz_raw) int32 stacked
        "slot_map",             # (nnz_raw,) int32 — maps raw slot → unique slot
        "Kff_indptr_u",         # (n_free+1,) int32 — unique CSR indptr
        "Kff_indices_u",        # (n_nnz_u,) int32 — unique CSR col indices
        "n_nnz_u",              # int — number of unique nonzeros
    )


@dataclass
class _CorrectionDecision:
    x: object
    residual: object
    raw_scale: Optional[float]
    applied_scale: float
    note: str = ""


def _coarsen_density_v4(
    rho_f,
    nelx_f: int,
    nely_f: int,
    nelz_f: int,
    nelx_c: int,
    nely_c: int,
    nelz_c: int,
):
    """
    Odd-dimension-safe 2:1 coarsening used by paper-4 diagnostics.

    The production Galerkin hierarchy does not use re-discretized coarse
    operators, but the same density pooling is useful for side-by-side
    ablations and per-level reports.
    """
    try:
        import cupy as cp

        is_cupy = isinstance(rho_f, cp.ndarray)
    except ImportError:
        cp = None
        is_cupy = False

    ix_c = np.minimum(np.arange(nelx_f, dtype=np.intp) // 2, nelx_c - 1)
    iy_c = np.minimum(np.arange(nely_f, dtype=np.intp) // 2, nely_c - 1)
    iz_c = np.minimum(np.arange(nelz_f, dtype=np.intp) // 2, nelz_c - 1)
    coarse_lin = (
        ix_c[:, None, None] * (nely_c * nelz_c)
        + iy_c[None, :, None] * nelz_c
        + iz_c[None, None, :]
    ).ravel()
    n_c = nelx_c * nely_c * nelz_c
    if is_cupy:
        coarse_gpu = cp.asarray(coarse_lin.astype(np.int64))
        rho_flat = rho_f.ravel().astype(cp.float64)
        rho_sum = cp.bincount(coarse_gpu, weights=rho_flat, minlength=n_c)
        cnt = cp.bincount(coarse_gpu, minlength=n_c).astype(cp.float64)
        return rho_sum / cp.maximum(cnt, 1.0)

    rho_flat = np.asarray(rho_f, dtype=float).ravel()
    rho_sum = np.bincount(coarse_lin, weights=rho_flat, minlength=n_c)
    cnt = np.bincount(coarse_lin, minlength=n_c).astype(float)
    return rho_sum / np.maximum(cnt, 1.0)


def _local_prolongation_scalar(s: int, t: int, u: int) -> np.ndarray:
    node_pos = np.array(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        dtype=np.float64,
    )
    P = np.zeros((8, 8), dtype=np.float64)
    stu = np.array([s, t, u], dtype=np.float64)
    for fn in range(8):
        coarse_coords = (stu + node_pos[fn]) / 2.0
        for cn in range(8):
            cx, cy, cz = node_pos[cn]
            wx = (1.0 - coarse_coords[0]) if cx == 0.0 else coarse_coords[0]
            wy = (1.0 - coarse_coords[1]) if cy == 0.0 else coarse_coords[1]
            wz = (1.0 - coarse_coords[2]) if cz == 0.0 else coarse_coords[2]
            P[fn, cn] = wx * wy * wz
    return P


def _build_level1_galerkin_struct(
    nelx_c: int,
    nely_c: int,
    nelz_c: int,
    nelx_f: int,
    nely_f: int,
    nelz_f: int,
    free_c: np.ndarray,
    KE_UNIT: np.ndarray,
) -> _GalerkinCoarseLevel:
    import cupy as cp

    from .pub_simp_solver import _build_sparse_indices, _edof_table_3d

    ndof_c = 3 * (nelx_c + 1) * (nely_c + 1) * (nelz_c + 1)
    n_free_c = len(free_c)

    edof_c = _edof_table_3d(nelx_c, nely_c, nelz_c)
    row_idx, col_idx = _build_sparse_indices(edof_c)

    free_mask = np.zeros(ndof_c, dtype=bool)
    free_mask[free_c] = True
    keep = free_mask[row_idx] & free_mask[col_idx]
    keep_idx = np.nonzero(keep)[0]

    free_local = np.full(ndof_c, -1, dtype=np.int32)
    free_local[free_c] = np.arange(n_free_c, dtype=np.int32)
    rows_local = free_local[row_idx[keep_idx]]
    cols_local = free_local[col_idx[keep_idx]]

    elem_of_kept = (keep_idx // 576).astype(np.int32)
    local_ke_of_kept = (keep_idx % 576).astype(np.int32)

    sort_idx = np.lexsort((cols_local, rows_local))
    rows_s = rows_local[sort_idx]
    cols_s = cols_local[sort_idx]
    elem_s = elem_of_kept[sort_idx]
    local_ke_s = local_ke_of_kept[sort_idx]

    indptr = np.searchsorted(
        rows_s.astype(np.int64),
        np.arange(n_free_c + 1, dtype=np.int64),
        side="left",
    ).astype(np.int32)

    row_expanded = np.repeat(np.arange(n_free_c, dtype=np.int32), np.diff(indptr))
    is_diag = cols_s == row_expanded
    diag_pos = np.nonzero(is_diag)[0].astype(np.int32)
    diag_row = row_expanded[is_diag].astype(np.int32)

    stu_list = [(s, t, u) for s in range(2) for t in range(2) for u in range(2)]
    q_flat_list = []
    for s, t, u in stu_list:
        P_sc = _local_prolongation_scalar(s, t, u)
        P_24 = np.kron(P_sc, np.eye(3, dtype=np.float64))
        q_flat_list.append((P_24.T @ KE_UNIT @ P_24).ravel().astype(np.float64))

    nyz_c = nely_c * nelz_c
    cx = elem_s // nyz_c
    cy = (elem_s // nelz_c) % nely_c
    cz = elem_s % nelz_c

    q_stu_sorted_cpu = []
    fine_elem_sorted_stu_cpu = []
    for stu_idx, (s, t, u) in enumerate(stu_list):
        q_stu_sorted_cpu.append(q_flat_list[stu_idx][local_ke_s])
        fx = np.minimum(2 * cx + s, nelx_f - 1)
        fy = np.minimum(2 * cy + t, nely_f - 1)
        fz = np.minimum(2 * cz + u, nelz_f - 1)
        fine_elem_sorted_stu_cpu.append(
            (fx * nely_f * nelz_f + fy * nelz_f + fz).astype(np.int32)
        )

    # Build unique-slot mapping: rows_s/cols_s are lex-sorted but may have duplicate
    # (row, col) pairs from different coarse elements sharing a sparsity entry.
    # slot_map[j] = index into the deduplicated CSR arrays for raw slot j.
    nnz_raw = len(rows_s)
    if nnz_raw > 0:
        is_new = np.ones(nnz_raw, dtype=bool)
        is_new[1:] = (rows_s[1:] != rows_s[:-1]) | (cols_s[1:] != cols_s[:-1])
        unique_slot = (np.cumsum(is_new) - 1).astype(np.int32)
        n_nnz_u = int(unique_slot[-1]) + 1
        unique_cols = cols_s[is_new].astype(np.int32)
        unique_rows = rows_s[is_new]
    else:
        unique_slot = np.empty(0, dtype=np.int32)
        n_nnz_u = 0
        unique_cols = np.empty(0, dtype=np.int32)
        unique_rows = np.empty(0, dtype=np.int32)
    unique_indptr = np.searchsorted(
        unique_rows.astype(np.int64),
        np.arange(n_free_c + 1, dtype=np.int64),
        side="left",
    ).astype(np.int32)

    level = _GalerkinCoarseLevel()
    level.nelx = nelx_c
    level.nely = nely_c
    level.nelz = nelz_c
    level.n_elem = nelx_c * nely_c * nelz_c
    level.n_free = n_free_c
    level.Kff_indptr = cp.asarray(indptr)
    level.Kff_indices = cp.asarray(cols_s.astype(np.int32))
    level.diag_pos = cp.asarray(diag_pos)
    level.diag_row = cp.asarray(diag_row)
    level.Q_stu_sorted = [cp.asarray(q) for q in q_stu_sorted_cpu]
    level.fine_elem_sorted_stu = [cp.asarray(fe) for fe in fine_elem_sorted_stu_cpu]
    level.Q_stu_stack     = cp.stack(level.Q_stu_sorted)          # (8, nnz_raw)
    level.fine_elem_stack = cp.stack(level.fine_elem_sorted_stu)  # (8, nnz_raw) int32
    level.slot_map        = cp.asarray(unique_slot)               # (nnz_raw,) int32
    level.Kff_indptr_u    = cp.asarray(unique_indptr)             # (n_free+1,)
    level.Kff_indices_u   = cp.asarray(unique_cols)               # (n_nnz_u,)
    level.n_nnz_u         = n_nnz_u
    return level


def assemble_level1_galerkin(
    level: _GalerkinCoarseLevel,
    E_e_gpu,
):
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    # Step 1: gather-multiply-sum over 8 sub-elements per raw slot (~3 kernel launches).
    raw = (E_e_gpu[level.fine_elem_stack] * level.Q_stu_stack).sum(axis=0)

    # Step 2: scatter-add raw contributions into unique (row,col) slots.
    # Use COO-based segment reduction instead of cp.add.at() for better
    # performance on WDDM platforms (avoids per-element atomic overhead).
    coo_data = cpsp.coo_matrix(
        (raw, (cp.zeros(len(raw), dtype=cp.int32), level.slot_map)),
        shape=(1, level.n_nnz_u),
    )
    data = cpsp.csr_matrix(coo_data).data

    K_c = cpsp.csr_matrix(
        (data, level.Kff_indices_u, level.Kff_indptr_u),
        shape=(level.n_free, level.n_free),
    )
    K_c.has_canonical_format = True  # no duplicates by construction; indices sorted by lexsort
    diag = K_c.diagonal()
    diag = cp.where(cp.abs(diag) > 1e-12, diag, cp.ones_like(diag))
    return K_c, diag


def _estimate_fine_lambda_max(
    mf_op: MatrixFreeKff,
    E_e_gpu,
    d_inv_gpu,
    n_iter: int = 20,
) -> float:
    import cupy as cp

    d_gpu = 1.0 / d_inv_gpu
    v = cp.ones(int(d_inv_gpu.shape[0]), dtype=cp.float64)
    v /= cp.sqrt(cp.dot(v * d_gpu, v))
    lam = 1.0
    for _ in range(n_iter):
        Kv = mf_op.matvec(v, E_e_gpu)
        vDv = float(cp.dot(v * d_gpu, v))
        vKv = float(cp.dot(v, Kv))
        lam = vKv / max(vDv, 1e-300)
        Av = d_inv_gpu * Kv
        nrm_D = float(cp.sqrt(cp.dot(Av * d_gpu, Av)))
        if nrm_D <= 1e-300:
            break
        v = Av / nrm_D
    return lam


def _estimate_sparse_lambda_max(K_gpu, diag_inv_gpu, n_iter: int = 10) -> float:
    """Power iteration for λ_max of D^{-1}K at a coarse sparse level."""
    import cupy as cp

    v = cp.ones(int(diag_inv_gpu.shape[0]), dtype=cp.float64)
    v /= float(cp.linalg.norm(v))
    lam = 1.0
    for _ in range(n_iter):
        Kv = K_gpu @ v
        vKv = float(cp.dot(v, Kv))
        Av = diag_inv_gpu * Kv
        nrm = float(cp.linalg.norm(Av))
        if nrm <= 1e-300:
            break
        lam = float(cp.dot(v, Kv)) / max(float(cp.dot(v, v / diag_inv_gpu)), 1e-300)
        v = Av / nrm
    return max(lam, 1e-6)


def _estimate_sparse_vram_bytes(K_gpu) -> int:
    return int(K_gpu.data.nbytes + K_gpu.indices.nbytes + K_gpu.indptr.nbytes)


def _estimate_transfer_vram_bytes(P_gpu, R_gpu) -> int:
    return _estimate_sparse_vram_bytes(P_gpu) + _estimate_sparse_vram_bytes(R_gpu)


def _cupy_fgmres(
    A_op: Callable,
    b,
    M_op: Callable,
    x0=None,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 32,
    history=None,
):
    """
    Flexible restarted GMRES with right preconditioning and Givens rotations.

    Uses Givens rotations (not lstsq) to maintain the upper-triangular
    Hessenberg factorization incrementally.  This is numerically stable even
    when the Krylov basis is nearly linearly dependent (e.g. near-converged
    systems or badly-conditioned MBB / torsion problems where lstsq diverges).

    The preconditioner M_op may be non-symmetric and lower-precision (FGMRES
    allows a variable/nonlinear preconditioner); only A_op must be fixed.
    """
    import cupy as cp

    x = cp.zeros_like(b) if x0 is None else x0.copy()
    b_norm = float(cp.linalg.norm(b))
    if b_norm <= 1e-30:
        return x, 0, True

    total_iters = 0
    converged = False

    while total_iters < maxiter:
        r = b - A_op(x)
        beta = float(cp.linalg.norm(r))
        if history is not None:
            history.append(beta / b_norm)
        if beta / b_norm <= tol:
            converged = True
            break

        V = [r / beta]
        Z = []
        # Upper Hessenberg (filled column by column via Givens)
        H = np.zeros((restart + 1, restart), dtype=np.float64)
        # Givens rotation parameters accumulated across the restart cycle
        cs = np.zeros(restart, dtype=np.float64)
        sn = np.zeros(restart, dtype=np.float64)
        # Residual RHS vector in the rotated frame  g[0] = beta, rest = 0
        g = np.zeros(restart + 1, dtype=np.float64)
        g[0] = beta

        inner_iters = min(restart, maxiter - total_iters)
        j_final = 0

        for j in range(inner_iters):
            z_j = M_op(V[j])
            Z.append(z_j)
            w = A_op(z_j)

            # Modified Gram-Schmidt orthogonalisation
            for i in range(j + 1):
                H[i, j] = float(cp.dot(V[i], w))
                w = w - H[i, j] * V[i]
            H[j + 1, j] = float(cp.linalg.norm(w))
            h_norm = H[j + 1, j]   # save before Givens zeros this entry

            if h_norm > 1e-14:
                V.append(w / h_norm)

            # Apply all previous Givens rotations to column j
            for i in range(j):
                tmp         =  cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j]     = tmp

            # Compute new Givens rotation to zero out H[j+1, j]
            denom = np.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
            if denom < 1e-300:
                cs[j], sn[j] = 1.0, 0.0
            else:
                cs[j] = H[j, j]     / denom
                sn[j] = H[j + 1, j] / denom

            # Apply the new rotation to H and g
            H[j, j]     = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1]    = -sn[j] * g[j]
            g[j]        =  cs[j] * g[j]

            total_iters += 1
            j_final = j + 1
            if history is not None:
                history.append(abs(g[j + 1]) / b_norm)

            if abs(g[j + 1]) / b_norm <= tol:
                converged = True
                break

            # Lucky breakdown — Krylov basis exhausted; back-solve gives exact answer
            if h_norm <= 1e-14:
                converged = True
                break

        # Back-solve upper triangular H[:j_final, :j_final] y = g[:j_final]
        if j_final > 0:
            R = np.triu(H[:j_final, :j_final])
            # Guard against exact breakdown (lucky convergence) where R is singular
            rhs_g = g[:j_final].copy()
            try:
                y = np.linalg.solve(R, rhs_g)
            except np.linalg.LinAlgError:
                y = np.linalg.lstsq(R, rhs_g, rcond=None)[0]
            for coeff, z_j in zip(y, Z[:j_final]):
                x = x + float(coeff) * z_j

        if converged:
            break

    return x, total_iters, converged


class GalerkinMatFreeGMG:
    """
    Production paper-4 multigrid hierarchy.

    The outer operator remains FP64 matrix-free.  Lower precision is used only
    inside the fine-level smoother, which is the part the paper targets for
    tensor-core acceleration.
    """

    def __init__(
        self,
        *,
        mf_op: MatrixFreeKff,
        free: np.ndarray,
        free_gpu,
        nelx: int,
        nely: int,
        nelz: int,
        KE_UNIT: np.ndarray,
        n_levels: int = 4,
        coarse_smooth_iters: int = 2,
        fine_smoother: str = "fp32",
        fine_smoother_degree: int = 2,
        omega: float = 0.5,
        fused_op=None,
        dense_chol_max: int = 5000,
        coarse_pcg_iters: int = 80,
        smoother_type: str = "chebyshev",
        cheb_lower_frac: float = 1.0 / 30.0,
        level_precisions: Optional[list] = None,
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
        root_local_correction_inner_restart: Optional[int] = None,
        inner_krylov_steps: int = 0,
        inner_krylov_restart: Optional[int] = None,
    ) -> None:
        self._mf_op = mf_op
        self._free = free
        self._free_gpu = free_gpu
        self._fine_smoother = fine_smoother
        self._fine_degree = max(1, int(fine_smoother_degree))
        self._coarse_smooth_iters = max(1, int(coarse_smooth_iters))
        self._omega_max = float(omega)
        self._omega_used = float(omega)
        self._fused_op = fused_op
        self._dense_chol_max = int(dense_chol_max)
        self._coarse_pcg_iters = int(coarse_pcg_iters)
        self._n_levels = max(1, int(n_levels))
        self._gpu_uploaded = False
        self._coarse_chol_mode = "none"
        self._coarse_chol_L = None
        self._is_setup = False
        self._level_stats: list[MultigridLevelStats] = []
        self._lambda_max_est = float("nan")
        # Chebyshev smoother controls
        if smoother_type not in {"chebyshev", "jacobi"}:
            raise ValueError(f"smoother_type must be 'chebyshev' or 'jacobi'; got {smoother_type!r}")
        self._smoother_type = smoother_type
        self._cheb_lower_frac = float(cheb_lower_frac)
        # Per-level precision: None → the paper's default schedule.
        # BF16 uses BF16 at level 0, FP32 at level 1, and FP64 thereafter.
        # FP32/FP64-only variants keep the coarser levels in FP64.
        if level_precisions is not None:
            self._level_prec = list(level_precisions)
        else:
            self._level_prec = None  # resolved after n_levels is known
        # Cycle type: "v" (default) or "w"
        if cycle_type not in {"v", "w"}:
            raise ValueError(f"cycle_type must be 'v' or 'w'; got {cycle_type!r}")
        self._cycle_type = cycle_type
        if coarse_correction_policy not in {"none", "residual_line_search"}:
            raise ValueError(
                "coarse_correction_policy must be 'none' or 'residual_line_search'; "
                f"got {coarse_correction_policy!r}"
            )
        if coarse_correction_acceptance_ratio <= 0.0:
            raise ValueError(
                "coarse_correction_acceptance_ratio must be positive; "
                f"got {coarse_correction_acceptance_ratio!r}"
            )
        if coarse_correction_max_scale <= 0.0:
            raise ValueError(
                f"coarse_correction_max_scale must be positive; got {coarse_correction_max_scale!r}"
            )
        self._coarse_correction_policy = coarse_correction_policy
        self._coarse_correction_acceptance_ratio = float(coarse_correction_acceptance_ratio)
        self._coarse_correction_max_scale = float(coarse_correction_max_scale)
        if root_enrichment_mode not in {
            "none",
            "weighted_residual_second",
            "weighted_gap_second",
            "void_residual_second",
            "void_gap_second",
        }:
            raise ValueError(
                "root_enrichment_mode must be 'none', 'weighted_residual_second', "
                "'weighted_gap_second', 'void_residual_second', or 'void_gap_second'; "
                f"got {root_enrichment_mode!r}"
            )
        if not (0.0 < root_enrichment_weight_floor <= 1.0):
            raise ValueError(
                "root_enrichment_weight_floor must lie in (0, 1]; "
                f"got {root_enrichment_weight_floor!r}"
            )
        if root_enrichment_weight_power <= 0.0:
            raise ValueError(
                f"root_enrichment_weight_power must be positive; got {root_enrichment_weight_power!r}"
            )
        self._root_enrichment_mode = root_enrichment_mode
        self._root_enrichment_weight_floor = float(root_enrichment_weight_floor)
        self._root_enrichment_weight_power = float(root_enrichment_weight_power)
        if root_local_correction_mode not in {
            "none",
            "node_block_line_search",
            "node_block_inner_fgmres",
        }:
            raise ValueError(
                "root_local_correction_mode must be 'none', 'node_block_line_search', "
                "or 'node_block_inner_fgmres'; "
                f"got {root_local_correction_mode!r}"
            )
        if root_local_correction_max_scale <= 0.0:
            raise ValueError(
                f"root_local_correction_max_scale must be positive; got {root_local_correction_max_scale!r}"
            )
        if root_local_correction_inner_steps <= 0:
            raise ValueError(
                "root_local_correction_inner_steps must be positive; "
                f"got {root_local_correction_inner_steps!r}"
            )
        if (
            root_local_correction_inner_restart is not None
            and root_local_correction_inner_restart <= 0
        ):
            raise ValueError(
                "root_local_correction_inner_restart must be positive when provided; "
                f"got {root_local_correction_inner_restart!r}"
            )
        self._root_local_correction_mode = root_local_correction_mode
        self._root_local_correction_max_scale = float(root_local_correction_max_scale)
        self._root_local_correction_inner_steps = int(root_local_correction_inner_steps)
        self._root_local_correction_inner_restart = (
            int(root_local_correction_inner_restart)
            if root_local_correction_inner_restart is not None
            else max(1, self._root_local_correction_inner_steps)
        )
        if inner_krylov_steps < 0:
            raise ValueError(f"inner_krylov_steps must be non-negative; got {inner_krylov_steps!r}")
        if inner_krylov_restart is not None and inner_krylov_restart <= 0:
            raise ValueError(
                f"inner_krylov_restart must be positive when provided; got {inner_krylov_restart!r}"
            )
        self._inner_krylov_steps = int(inner_krylov_steps)
        self._inner_krylov_restart = (
            int(inner_krylov_restart) if inner_krylov_restart is not None else max(1, self._inner_krylov_steps)
        )
        # Per-level spectral estimates (filled in setup())
        self._lambda_max_per_level: list[float] = []
        # FP32 K copies for mid-level precision-descent (filled in setup())
        self._K_fp32_gpu: list = []
        self._diag_inv_fp32_gpu: list = []
        # κ_eff estimate (filled by estimate_kappa_eff())
        self._kappa_eff: float = float("nan")
        # Lambda-max caching: track max(E_e) at last estimation.
        # Re-estimate when max(E_e) changes by more than 10% to avoid
        # Chebyshev miscalibration during SIMP density evolution.
        self._lambda_max_call_count: int = 0
        self._lambda_max_E_e_max: float = float("nan")
        # Contrast-adaptive smoother controls
        self._contrast_ratio: float = float("nan")
        self._is_high_contrast: bool = False
        self._coarse_smooth_iters_adapt: int = coarse_smooth_iters
        self._fine_degree_adapt: int = max(1, int(fine_smoother_degree))

        dims = [(nelx, nely, nelz)]
        for _ in range(1, self._n_levels):
            nx, ny, nz = dims[-1]
            dims.append((max(1, nx // 2), max(1, ny // 2), max(1, nz // 2)))
        self._dims = dims

        self._free_per_level: list[np.ndarray] = [free]
        self._n_free: list[int] = [len(free)]
        self._P_scipy: list[sp.csr_matrix] = []
        for lv in range(self._n_levels - 1):
            nx_f, ny_f, nz_f = dims[lv]
            nx_c, ny_c, nz_c = dims[lv + 1]
            free_c = _coarse_free_dofs_injection(
                nx_f,
                ny_f,
                nz_f,
                nx_c,
                ny_c,
                nz_c,
                self._free_per_level[lv],
            )
            self._free_per_level.append(free_c)
            self._n_free.append(len(free_c))
            P_sc = _build_scalar_prolongation(nx_f, ny_f, nz_f, nx_c, ny_c, nz_c)
            P_vec = sp.kron(P_sc, sp.eye(3, format="csr", dtype=np.float64), format="csr")
            del P_sc
            P_free = P_vec[self._free_per_level[lv], :][:, free_c].tocsr()
            self._P_scipy.append(P_free)
            del P_vec, P_free
            gc.collect()

        self._P_gpu = [None] * max(0, self._n_levels - 1)
        self._R_gpu = [None] * max(0, self._n_levels - 1)
        self._K_gpu = [None] * self._n_levels
        self._diag_inv_gpu = [None] * self._n_levels
        self._E_e_fp64 = None
        self._E_e_fp32 = None
        self._level1_struct = None
        self._root_enrichment_weights_gpu = None
        self._root_block_jacobi = None
        if self._n_levels > 1:
            nx_c, ny_c, nz_c = dims[1]
            self._level1_struct = _build_level1_galerkin_struct(
                nx_c,
                ny_c,
                nz_c,
                nelx,
                nely,
                nelz,
                self._free_per_level[1],
                KE_UNIT,
            )

    @property
    def level_stats(self) -> list[MultigridLevelStats]:
        return list(self._level_stats)

    @property
    def omega_used(self) -> float:
        return self._omega_used

    @property
    def lambda_max_estimate(self) -> float:
        return self._lambda_max_est

    def uses_bf16_smoother(self) -> bool:
        return self._fine_smoother == "bf16"

    def supports_spd_outer(self) -> bool:
        return (
            self._fine_smoother in {"fp64", "fp32"}
            and self._coarse_correction_policy == "none"
            and self._root_enrichment_mode == "none"
            and self._root_local_correction_mode == "none"
            and self._inner_krylov_steps == 0
        )

    @property
    def kappa_eff(self) -> float:
        return self._kappa_eff

    @property
    def lambda_max_per_level(self) -> list:
        return list(self._lambda_max_per_level)

    @property
    def level_precisions(self) -> list:
        return list(self._level_prec) if self._level_prec else []

    @property
    def contrast_ratio(self) -> float:
        return self._contrast_ratio

    @property
    def is_high_contrast(self) -> bool:
        return self._is_high_contrast

    @property
    def adaptive_params(self) -> dict:
        return {
            "contrast_ratio": self._contrast_ratio,
            "is_high_contrast": self._is_high_contrast,
            "fine_degree_base": self._fine_degree,
            "fine_degree_adapt": self._fine_degree_adapt,
            "coarse_iters_base": self._coarse_smooth_iters,
            "coarse_iters_adapt": self._coarse_smooth_iters_adapt,
            "coarse_correction_policy": self._coarse_correction_policy,
            "coarse_correction_acceptance_ratio": self._coarse_correction_acceptance_ratio,
            "coarse_correction_max_scale": self._coarse_correction_max_scale,
            "root_enrichment_mode": self._root_enrichment_mode,
            "root_enrichment_weight_floor": self._root_enrichment_weight_floor,
            "root_enrichment_weight_power": self._root_enrichment_weight_power,
            "root_local_correction_mode": self._root_local_correction_mode,
            "root_local_correction_max_scale": self._root_local_correction_max_scale,
            "root_local_correction_inner_steps": self._root_local_correction_inner_steps,
            "root_local_correction_inner_restart": self._root_local_correction_inner_restart,
            "inner_krylov_steps": self._inner_krylov_steps,
            "inner_krylov_restart": self._inner_krylov_restart,
        }

    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        if abs(den) <= 1e-300:
            return 0.0 if abs(num) <= 1e-300 else float("inf")
        return float(num / den)

    def _vector_norm(self, v) -> float:
        import cupy as cp

        return float(cp.linalg.norm(v))

    def _level_matvec(self, lv: int, v):
        if lv == 0:
            return self.apply_fine_operator(v)
        return self._K_gpu[lv] @ v

    def _correction_alignment(self, reference, correction) -> float:
        import cupy as cp

        ref_norm = self._vector_norm(reference)
        corr_norm = self._vector_norm(correction)
        if ref_norm <= 1e-300 or corr_norm <= 1e-300:
            return 0.0
        return float(cp.dot(reference, correction) / (ref_norm * corr_norm))

    def _residual_for_level(self, lv: int, rhs, x):
        return rhs - self._level_matvec(lv, x)

    def _apply_coarse_correction(
        self,
        lv: int,
        x,
        residual,
        correction,
    ) -> _CorrectionDecision:
        import cupy as cp

        if self._coarse_correction_policy == "none":
            return _CorrectionDecision(
                x=x + correction,
                residual=residual - self._level_matvec(lv, correction),
                raw_scale=1.0,
                applied_scale=1.0,
            )

        correction_norm = self._vector_norm(correction)
        residual_norm = self._vector_norm(residual)
        if correction_norm <= 1e-300 or residual_norm <= 1e-300:
            return _CorrectionDecision(
                x=x,
                residual=residual,
                raw_scale=0.0,
                applied_scale=0.0,
                note="degenerate_correction",
            )

        if self._coarse_correction_policy == "residual_line_search":
            A_correction = self._level_matvec(lv, correction)
            denom = float(cp.dot(A_correction, A_correction))
            if denom <= 1e-300:
                return _CorrectionDecision(
                    x=x,
                    residual=residual,
                    raw_scale=0.0,
                    applied_scale=0.0,
                    note="degenerate_correction",
                )
            raw_scale = float(cp.dot(residual, A_correction) / denom)
            applied_scale = min(self._coarse_correction_max_scale, max(0.0, raw_scale))
            trial_residual = residual - applied_scale * A_correction
            trial_norm = self._vector_norm(trial_residual)
            note = ""
            if trial_norm > self._coarse_correction_acceptance_ratio * residual_norm:
                return _CorrectionDecision(
                    x=x,
                    residual=residual,
                    raw_scale=raw_scale,
                    applied_scale=0.0,
                    note="rejected_correction",
                )
            if applied_scale <= 1e-12 and raw_scale <= 0.0:
                note = "nonpositive_correction_scale"
            elif abs(applied_scale - raw_scale) > 1e-12:
                note = "clipped_correction_scale"
            return _CorrectionDecision(
                x=x + applied_scale * correction,
                residual=trial_residual,
                raw_scale=raw_scale,
                applied_scale=applied_scale,
                note=note,
            )

        raise RuntimeError(f"Unsupported coarse_correction_policy {self._coarse_correction_policy!r}")

    @staticmethod
    def _merge_notes(*notes: str) -> str:
        merged = [note for note in notes if note]
        return "|".join(merged)

    def _root_enrichment_rhs(self, residual):
        if self._root_enrichment_mode == "none":
            return None, ""
        if self._root_enrichment_weights_gpu is None:
            raise RuntimeError("Root enrichment weights are not initialised; call setup() first.")

        stiff_weighted_residual = self._root_enrichment_weights_gpu * residual
        void_weights = 1.0 - self._root_enrichment_weights_gpu
        void_weights = void_weights.clip(self._root_enrichment_weight_floor, 1.0)
        void_weighted_residual = void_weights * residual
        if self._root_enrichment_mode == "weighted_residual_second":
            return self._R_gpu[0] @ stiff_weighted_residual, "root_enrichment:weighted_residual_second"
        if self._root_enrichment_mode == "weighted_gap_second":
            base_rhs = self._R_gpu[0] @ residual
            projector = self._P_gpu[0] @ base_rhs
            weighted_gap = self._root_enrichment_weights_gpu * (residual - projector)
            return self._R_gpu[0] @ weighted_gap, "root_enrichment:weighted_gap_second"
        if self._root_enrichment_mode == "void_residual_second":
            return self._R_gpu[0] @ void_weighted_residual, "root_enrichment:void_residual_second"
        if self._root_enrichment_mode == "void_gap_second":
            base_rhs = self._R_gpu[0] @ residual
            projector = self._P_gpu[0] @ base_rhs
            void_gap = void_weights * (residual - projector)
            return self._R_gpu[0] @ void_gap, "root_enrichment:void_gap_second"
        raise RuntimeError(f"Unsupported root_enrichment_mode {self._root_enrichment_mode!r}")

    def _apply_root_local_correction(self, x, residual) -> _CorrectionDecision:
        import cupy as cp

        if self._root_local_correction_mode == "none":
            return _CorrectionDecision(x=x, residual=residual, raw_scale=None, applied_scale=0.0)
        if self._root_local_correction_mode == "node_block_inner_fgmres":
            correction, _, _ = _cupy_fgmres(
                self.apply_fine_operator,
                residual,
                self._root_block_jacobi.apply,
                tol=0.0,
                maxiter=self._root_local_correction_inner_steps,
                restart=self._root_local_correction_inner_restart,
            )
            correction_norm = self._vector_norm(correction)
            residual_norm = self._vector_norm(residual)
            if correction_norm <= 1e-300 or residual_norm <= 1e-300:
                return _CorrectionDecision(
                    x=x,
                    residual=residual,
                    raw_scale=0.0,
                    applied_scale=0.0,
                    note="degenerate_root_local_correction",
                )
            trial_residual = residual - self.apply_fine_operator(correction)
            trial_norm = self._vector_norm(trial_residual)
            if trial_norm > residual_norm:
                return _CorrectionDecision(
                    x=x,
                    residual=residual,
                    raw_scale=None,
                    applied_scale=0.0,
                    note="rejected_root_local_inner_fgmres",
                )
            return _CorrectionDecision(
                x=x + correction,
                residual=trial_residual,
                raw_scale=None,
                applied_scale=1.0,
                note="root_local_inner_fgmres",
            )

        if self._root_local_correction_mode != "node_block_line_search":
            raise RuntimeError(
                f"Unsupported root_local_correction_mode {self._root_local_correction_mode!r}"
            )
        if self._root_block_jacobi is None:
            raise RuntimeError("Root-local correction requested before NodeBlockJacobiMF setup.")

        correction = self._root_block_jacobi.apply(residual)
        correction_norm = self._vector_norm(correction)
        residual_norm = self._vector_norm(residual)
        if correction_norm <= 1e-300 or residual_norm <= 1e-300:
            return _CorrectionDecision(
                x=x,
                residual=residual,
                raw_scale=0.0,
                applied_scale=0.0,
                note="degenerate_root_local_correction",
            )

        A_correction = self.apply_fine_operator(correction)
        denom = float(cp.dot(A_correction, A_correction))
        if denom <= 1e-300:
            return _CorrectionDecision(
                x=x,
                residual=residual,
                raw_scale=0.0,
                applied_scale=0.0,
                note="degenerate_root_local_correction",
            )
        raw_scale = float(cp.dot(residual, A_correction) / denom)
        applied_scale = min(self._root_local_correction_max_scale, max(0.0, raw_scale))
        if applied_scale <= 1e-12:
            return _CorrectionDecision(
                x=x,
                residual=residual,
                raw_scale=raw_scale,
                applied_scale=0.0,
                note="nonpositive_root_local_scale",
            )
        trial_residual = residual - applied_scale * A_correction
        note = ""
        if abs(applied_scale - raw_scale) > 1e-12:
            note = "clipped_root_local_scale"
        return _CorrectionDecision(
            x=x + applied_scale * correction,
            residual=trial_residual,
            raw_scale=raw_scale,
            applied_scale=applied_scale,
            note=note,
        )

    def estimate_kappa_eff(self, n_iter: int = 30) -> float:
        """
        Estimate κ_eff = λ_max(M^{-1}A) via Lanczos on the preconditioned operator.

        Runs n_iter steps of Lanczos and returns λ_max/λ_min of the resulting
        tridiagonal matrix.  This reflects the condition number seen by the outer
        PCG/FGMRES solver and is the central quantity for the ε·κ_eff < 1 bound.
        """
        import cupy as cp
        # Empirical kappa_eff probe on the applied left-preconditioned operator M K.
        # Use a fixed CuPy RNG seed so the paper-facing CSV values are exactly reproducible.

        if not self._is_setup:
            raise RuntimeError("Call setup() before estimate_kappa_eff().")
        if (
            self._coarse_correction_policy != "none"
            or self._root_enrichment_mode != "none"
            or self._root_local_correction_mode != "none"
            or self._inner_krylov_steps > 0
        ):
            raise RuntimeError(
                "estimate_kappa_eff requires a linear preconditioner; disable safeguarded "
                "coarse correction, root enrichment, root-local correction, and inner Krylov "
                "wrapping first."
            )

        n = self._n_free[0]
        v = cp.random.RandomState(0).standard_normal(n, dtype=cp.float64)
        v /= float(cp.linalg.norm(v))

        alpha_vec = np.zeros(n_iter, dtype=np.float64)
        beta_vec = np.zeros(n_iter - 1, dtype=np.float64)
        v_prev = cp.zeros(n, dtype=cp.float64)

        for j in range(n_iter):
            # w = M K v for the frozen hierarchy viewed as an applied
            # left-preconditioning operator.
            w = self.apply(self.apply_fine_operator(v))
            alpha_j = float(cp.dot(v, w))
            alpha_vec[j] = alpha_j
            w = w - alpha_j * v
            if j > 0:
                w = w - beta_vec[j - 1] * v_prev
            if j < n_iter - 1:
                beta_j = float(cp.linalg.norm(w))
                beta_vec[j] = beta_j
                if beta_j < 1e-14:
                    alpha_vec = alpha_vec[: j + 1]
                    beta_vec = beta_vec[:j]
                    break
                v_prev = v
                v = w / beta_j

        T = np.diag(alpha_vec) + np.diag(beta_vec, 1) + np.diag(beta_vec, -1)
        eigs = np.linalg.eigvalsh(T)
        eigs = eigs[eigs > 0]
        if len(eigs) == 0:
            self._kappa_eff = float("nan")
        else:
            self._kappa_eff = float(eigs[-1] / max(eigs[0], 1e-300))
        return self._kappa_eff

    def setup(self, E_e_gpu, fine_diag=None) -> None:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        if not self._gpu_uploaded:
            for lv in range(self._n_levels - 1):
                self._P_gpu[lv] = cpsp.csr_matrix(self._P_scipy[lv])
                self._P_gpu[lv].sum_duplicates()  # sets has_canonical_format=True once
                self._R_gpu[lv] = self._P_gpu[lv].T.tocsr()
                self._R_gpu[lv].sum_duplicates()
            self._gpu_uploaded = True

        self._E_e_fp64 = E_e_gpu.astype(cp.float64, copy=False)
        self._E_e_fp32 = self._E_e_fp64.astype(cp.float32)

        # Resolve level_precisions now that n_levels is known
        if self._level_prec is None:
            if self._fine_smoother == "bf16":
                self._level_prec = [self._fine_smoother]
                if self._n_levels > 1:
                    self._level_prec.append("fp32")
                self._level_prec.extend(["fp64"] * max(0, self._n_levels - len(self._level_prec)))
            else:
                self._level_prec = [self._fine_smoother] + ["fp64"] * max(0, self._n_levels - 1)
        while len(self._level_prec) < self._n_levels:
            self._level_prec.append("fp64")

        diag0 = fine_diag if fine_diag is not None else self._mf_op.extract_diagonal(self._E_e_fp64)
        diag0 = cp.where(cp.abs(diag0) > 1e-12, diag0, cp.ones_like(diag0))
        self._diag_inv_gpu[0] = 1.0 / diag0
        if self._root_enrichment_mode != "none":
            diag0_max = float(cp.max(cp.abs(diag0)))
            weight_base = cp.abs(diag0) / max(diag0_max, 1e-300)
            weight_base = cp.clip(weight_base, self._root_enrichment_weight_floor, 1.0)
            if abs(self._root_enrichment_weight_power - 1.0) > 1e-12:
                weight_base = weight_base ** self._root_enrichment_weight_power
            self._root_enrichment_weights_gpu = weight_base.astype(cp.float64, copy=False)
        else:
            self._root_enrichment_weights_gpu = None
        if self._root_local_correction_mode != "none":
            if self._root_block_jacobi is None:
                self._root_block_jacobi = NodeBlockJacobiMF(
                    edof_gpu=self._mf_op._edof_gpu,
                    KE_unit_gpu=self._mf_op._KE_unit,
                    free_gpu=self._mf_op._free_gpu,
                    n_free=self._mf_op._n_free,
                    ndof=self._mf_op._ndof,
                )
            self._root_block_jacobi.update(self._E_e_fp64, scalar_diag=diag0)
        else:
            self._root_block_jacobi = None
        self._lambda_max_call_count += 1
        # Re-estimate lambda_max when max(E_e) shifts by >10% from the last estimation.
        # max(E_e) is a good proxy for lambda_max since K ~ E_e * KE_ref element-wise.
        # This avoids Chebyshev miscalibration during rapid SIMP density changes while
        # caching across solve calls where rho is nearly constant.
        E_e_max = float(cp.max(self._E_e_fp64))
        prev_max = self._lambda_max_E_e_max
        reestimate_lam = (
            np.isnan(prev_max)
            or abs(E_e_max - prev_max) / max(abs(prev_max), 1e-12) > 0.10
        )
        if reestimate_lam:
            self._lambda_max_est = _estimate_fine_lambda_max(
                self._mf_op,
                self._E_e_fp64,
                self._diag_inv_gpu[0],
            )
            self._lambda_max_E_e_max = E_e_max
        self._omega_used = min(self._omega_max, 0.9 * 2.0 / max(self._lambda_max_est, 1e-6))
        # Preserve cached coarse estimates; only reset if the list is the wrong length.
        if len(self._lambda_max_per_level) != self._n_levels:
            self._lambda_max_per_level = [float("nan")] * self._n_levels
        self._lambda_max_per_level[0] = self._lambda_max_est
        self._K_fp32_gpu = [None] * self._n_levels
        self._diag_inv_fp32_gpu = [None] * self._n_levels

        # ── Material contrast detection and adaptive smoother ──
        # Estimate effective contrast ratio from E_e statistics.
        # For SIMP fields, the full dynamic range E_max / E_min controls
        # the solver's sensitivity to smoother strength. We use two diagnostics:
        #   (1) Full E_e ratio: max / min across all elements (captures bimodality)
        #   (2) Fraction of near-void elements
        # High contrast triggers stronger smoothing at coarse levels and
        # a wider Chebyshev targeting interval.
        E_e = self._E_e_fp64
        E_e_min = float(cp.min(E_e))
        self._contrast_ratio = E_e_max / max(E_e_min, 1e-300)
        void_floor = float(E_e_max * 1e-8)
        solid_mask = E_e > void_floor
        n_solid = int(cp.count_nonzero(solid_mask))
        void_frac = 1.0 - n_solid / max(len(E_e), 1)

        # Adaptive parameter selection based on contrast ratio
        # Thresholds chosen from the failure analysis in E1/E10:
        #   - CR > 10:  moderate contrast, boost coarse smoothing
        #   - CR > 100: high contrast, boost all smoothers + widen Chebyshev
        #   - CR > 1e5: extreme, use maximum smoothing budget
        base_coarse = self._coarse_smooth_iters
        base_fine = self._fine_degree

        if self._contrast_ratio > 1e5 or void_frac > 0.8:
            # Extreme contrast or near-void: maximum smoothing
            self._coarse_smooth_iters_adapt = max(base_coarse, 6)
            self._fine_degree_adapt = max(base_fine, 4)
            self._is_high_contrast = True
        elif self._contrast_ratio > 100 or void_frac > 0.5:
            # High contrast: strengthen coarse correction
            self._coarse_smooth_iters_adapt = max(base_coarse, 4)
            self._fine_degree_adapt = max(base_fine, 3)
            self._is_high_contrast = True
        elif self._contrast_ratio > 10:
            # Moderate contrast: mild boost
            self._coarse_smooth_iters_adapt = max(base_coarse, 3)
            self._fine_degree_adapt = self._fine_degree
            self._is_high_contrast = False
        else:
            # Uniform or mild field: keep defaults
            self._coarse_smooth_iters_adapt = base_coarse
            self._fine_degree_adapt = base_fine
            self._is_high_contrast = False

        self._level_stats = [
            MultigridLevelStats(
                level=0,
                kind=f"fine-matrixfree-{self._fine_smoother}",
                n_elem=self._dims[0][0] * self._dims[0][1] * self._dims[0][2],
                n_free=self._n_free[0],
                nnz=0,
                estimated_vram_bytes=int(diag0.nbytes),
            )
        ]

        if self._n_levels > 1:
            K1, diag1 = assemble_level1_galerkin(self._level1_struct, self._E_e_fp64)
            self._K_gpu[1] = K1
            self._diag_inv_gpu[1] = 1.0 / diag1
            if reestimate_lam or len(self._lambda_max_per_level) < 2 or np.isnan(self._lambda_max_per_level[1]):
                lam1 = _estimate_sparse_lambda_max(K1, self._diag_inv_gpu[1])
                self._lambda_max_per_level[1] = lam1
            if self._level_prec[1] == "fp32":
                self._K_fp32_gpu[1] = K1.astype(cp.float32)
                self._diag_inv_fp32_gpu[1] = self._diag_inv_gpu[1].astype(cp.float32)
            self._level_stats.append(
                MultigridLevelStats(
                    level=1,
                    kind="level1-galerkin",
                    n_elem=self._dims[1][0] * self._dims[1][1] * self._dims[1][2],
                    n_free=self._n_free[1],
                    nnz=int(K1.nnz),
                    estimated_vram_bytes=_estimate_sparse_vram_bytes(K1)
                    + int(diag1.nbytes)
                    + _estimate_transfer_vram_bytes(self._P_gpu[0], self._R_gpu[0]),
                )
            )

        for lv in range(2, self._n_levels):
            K_prev = self._K_gpu[lv - 1]
            K_c = (self._R_gpu[lv - 1] @ (K_prev @ self._P_gpu[lv - 1])).tocsr()
            K_c.sum_duplicates()
            diag_c = K_c.diagonal()
            diag_c = cp.where(cp.abs(diag_c) > 1e-12, diag_c, cp.ones_like(diag_c))
            self._K_gpu[lv] = K_c
            self._diag_inv_gpu[lv] = 1.0 / diag_c
            if reestimate_lam or len(self._lambda_max_per_level) <= lv or np.isnan(self._lambda_max_per_level[lv]):
                lam_c = _estimate_sparse_lambda_max(K_c, self._diag_inv_gpu[lv])
                self._lambda_max_per_level[lv] = lam_c
            if self._level_prec[lv] == "fp32":
                self._K_fp32_gpu[lv] = K_c.astype(cp.float32)
                self._diag_inv_fp32_gpu[lv] = self._diag_inv_gpu[lv].astype(cp.float32)
            self._level_stats.append(
                MultigridLevelStats(
                    level=lv,
                    kind="galerkin-triple-product",
                    n_elem=self._dims[lv][0] * self._dims[lv][1] * self._dims[lv][2],
                    n_free=self._n_free[lv],
                    nnz=int(K_c.nnz),
                    estimated_vram_bytes=_estimate_sparse_vram_bytes(K_c)
                    + int(diag_c.nbytes)
                    + _estimate_transfer_vram_bytes(self._P_gpu[lv - 1], self._R_gpu[lv - 1]),
                )
            )

        if self._n_levels == 1:
            self._coarse_chol_mode = "none"
            self._coarse_chol_L = None
        else:
            K_coarsest = self._K_gpu[self._n_levels - 1]
            n_coarsest = int(K_coarsest.shape[0])
            if n_coarsest <= self._dense_chol_max:
                K_dense = K_coarsest.toarray()
                try:
                    L = cp.linalg.cholesky(K_dense)
                    if not bool(cp.all(cp.isfinite(L))):
                        raise RuntimeError("non-finite L")
                    self._coarse_chol_L = L
                    self._coarse_chol_mode = "dense"
                except Exception:
                    # Near-singular coarsest matrix (e.g. nz=1 with no z-BCs).
                    # Regularise with eps·I to damp the rigid-body null mode, then
                    # retry Cholesky.  eps = mean_diag * 1e-8 barely affects the
                    # physics but removes the near-zero eigenvalue.
                    mean_diag = float(K_dense.diagonal().mean())
                    eps = max(mean_diag * 1e-8, 1e-14)
                    K_reg = K_dense + eps * cp.eye(n_coarsest, dtype=K_dense.dtype)
                    try:
                        L = cp.linalg.cholesky(K_reg)
                        if bool(cp.all(cp.isfinite(L))):
                            self._coarse_chol_L = L
                            self._coarse_chol_mode = "dense"
                            print(f"[GMG] coarsest matrix regularised (eps={eps:.2e}): Cholesky OK")
                        else:
                            raise RuntimeError("still non-finite after regularisation")
                    except Exception:
                        self._coarse_chol_L = None
                        self._coarse_chol_mode = "iterative"
                        print("[GMG] WARNING: coarsest Cholesky failed even after regularisation "
                              "— using iterative coarse solve")
            else:
                self._coarse_chol_L = None
                self._coarse_chol_mode = "iterative"

        self._is_setup = True

    def apply_fine_operator(self, v):
        return self._mf_op.matvec(v, self._E_e_fp64)

    def apply_preconditioned_operator(self, v):
        return self.apply(self.apply_fine_operator(v))

    def probe_quality(self, rhs) -> dict:
        import cupy as cp

        z = self.apply(rhs)
        z_jac = self._diag_inv_gpu[0] * rhs
        return {
            "finite": bool(cp.all(cp.isfinite(z))),
            "pd": float(cp.dot(rhs, z)) > 0.0,
            "z_norm": float(cp.linalg.norm(z)),
            "jacobi_norm": float(cp.linalg.norm(z_jac)),
            "z_over_jacobi": float(cp.linalg.norm(z) / max(float(cp.linalg.norm(z_jac)), 1e-30)),
        }

    def cycle_trace(self, rhs, *, cycle_type: Optional[str] = None) -> MultigridCycleTrace:
        """
        Return a structured trace of one multigrid cycle on the supplied RHS.

        This is intended for targeted diagnostics on hard cases.  It does not
        change the default solve path unless called explicitly.
        """
        import cupy as cp

        if not self._is_setup:
            raise RuntimeError("GalerkinMatFreeGMG.setup() must be called before cycle_trace().")

        rhs64 = rhs.astype(np.float64, copy=False) if hasattr(rhs, "astype") else cp.asarray(rhs, dtype=np.float64)
        cycle = self._cycle_type if cycle_type is None else cycle_type
        if cycle not in {"v", "w"}:
            raise ValueError(f"cycle_type must be 'v' or 'w'; got {cycle!r}")

        visits: list[MultigridCycleVisit] = []
        visit_counter = [0]
        if cycle == "w":
            out = self._trace_wcycle_level(0, rhs64, visits, visit_counter, cycle)
        else:
            out = self._trace_vcycle_level(0, rhs64, visits, visit_counter, cycle)

        pre_local_residual = rhs64 - self.apply_fine_operator(out)
        local_decision = self._apply_root_local_correction(out, pre_local_residual)
        out = local_decision.x
        rhs_norm = self._vector_norm(rhs64)
        jacobi_action = self._diag_inv_gpu[0] * rhs64
        jacobi_norm = self._vector_norm(jacobi_action)
        out_norm = self._vector_norm(out)
        out_residual = local_decision.residual
        out_residual_norm = self._vector_norm(out_residual)
        rz = float(cp.dot(rhs64, out))
        visits.sort(key=lambda visit: visit.visit_id)
        if local_decision.note:
            root_visit = next((visit for visit in visits if visit.level == 0 and not visit.is_coarsest), None)
            if root_visit is not None:
                root_visit.notes = self._merge_notes(root_visit.notes, local_decision.note)

        return MultigridCycleTrace(
            cycle_type=cycle,
            rhs_norm=rhs_norm,
            jacobi_action_norm=jacobi_norm,
            output_norm=out_norm,
            z_over_jacobi=self._safe_ratio(out_norm, jacobi_norm),
            rz=rz,
            pd=bool(rz > 0.0),
            output_residual_norm=out_residual_norm,
            output_residual_ratio=self._safe_ratio(out_residual_norm, rhs_norm),
            contrast_ratio=self._contrast_ratio,
            is_high_contrast=self._is_high_contrast,
            fine_degree_adapt=self._fine_degree_adapt,
            coarse_iters_adapt=self._coarse_smooth_iters_adapt,
            coarse_correction_policy=self._coarse_correction_policy,
            coarse_correction_acceptance_ratio=self._coarse_correction_acceptance_ratio,
            coarse_correction_max_scale=self._coarse_correction_max_scale,
            root_enrichment_mode=self._root_enrichment_mode,
            root_enrichment_weight_floor=self._root_enrichment_weight_floor,
            root_enrichment_weight_power=self._root_enrichment_weight_power,
            root_local_correction_mode=self._root_local_correction_mode,
            root_local_correction_inner_steps=self._root_local_correction_inner_steps,
            root_local_correction_inner_restart=self._root_local_correction_inner_restart,
            root_local_correction_raw_scale=local_decision.raw_scale,
            root_local_correction_applied_scale=local_decision.applied_scale,
            root_local_correction_residual_ratio=self._safe_ratio(out_residual_norm, self._vector_norm(pre_local_residual)),
            inner_krylov_steps=self._inner_krylov_steps,
            inner_krylov_restart=self._inner_krylov_restart,
            visits=visits,
        )

    def _trace_vcycle_level(
        self,
        lv: int,
        rhs,
        visits: list[MultigridCycleVisit],
        visit_counter: list[int],
        cycle_type: str,
    ):
        import cupy as cp

        visit_id = visit_counter[0]
        visit_counter[0] += 1

        rhs_norm = self._vector_norm(rhs)
        jacobi_action_norm = self._vector_norm(self._diag_inv_gpu[lv] * rhs)

        if lv == self._n_levels - 1:
            x = self._solve_coarsest(rhs)
            residual = rhs - self._level_matvec(lv, x)
            residual_norm = self._vector_norm(residual)
            visits.append(
                MultigridCycleVisit(
                    visit_id=visit_id,
                    level=lv,
                    cycle_type=cycle_type,
                    is_coarsest=True,
                    n_coarse_corrections=0,
                    rhs_norm=rhs_norm,
                    jacobi_action_norm=jacobi_action_norm,
                    presmooth_solution_norm=0.0,
                    presmooth_residual_norm=rhs_norm,
                    presmooth_residual_ratio=1.0,
                    restricted_rhs_norm=0.0,
                    restricted_rhs_ratio=0.0,
                    projector_norm=0.0,
                    coarse_space_gap_norm=0.0,
                    coarse_space_gap_ratio=0.0,
                    coarse_correction_norm=self._vector_norm(x),
                    prolongated_correction_norm=self._vector_norm(x),
                    correction_alignment=0.0,
                    correction_scale_raw=None,
                    correction_scale_applied=1.0,
                    corrected_residual_norm=residual_norm,
                    corrected_residual_vs_presmooth=self._safe_ratio(residual_norm, rhs_norm),
                    postsmooth_solution_norm=self._vector_norm(x),
                    postsmooth_residual_norm=residual_norm,
                    postsmooth_residual_ratio=self._safe_ratio(residual_norm, rhs_norm),
                    coarsest_mode=self._coarse_chol_mode,
                )
            )
            return x

        x0 = cp.zeros_like(rhs)
        if lv == 0:
            x_smooth = self._smooth_fine(x0, rhs)
        else:
            x_smooth = self._smooth_coarse(lv, x0, rhs)
        presmooth_residual = rhs - self._level_matvec(lv, x_smooth)
        presmooth_residual_norm = self._vector_norm(presmooth_residual)

        r_c = self._R_gpu[lv] @ presmooth_residual
        restricted_rhs_norm = self._vector_norm(r_c)
        projector = self._P_gpu[lv] @ r_c
        projector_norm = self._vector_norm(projector)
        coarse_space_gap = presmooth_residual - projector
        coarse_space_gap_norm = self._vector_norm(coarse_space_gap)

        e_c = self._trace_vcycle_level(lv + 1, r_c, visits, visit_counter, cycle_type)
        P_e_c = self._P_gpu[lv] @ e_c
        correction_alignment = self._correction_alignment(presmooth_residual, P_e_c)
        correction_decision = self._apply_coarse_correction(lv, x_smooth, presmooth_residual, P_e_c)
        x_corrected = correction_decision.x
        corrected_residual = correction_decision.residual
        corrected_residual_norm = self._vector_norm(corrected_residual)
        first_corrected_residual_norm = corrected_residual_norm
        n_coarse_corrections = 1
        restricted_rhs_norm_2 = None
        e_c2 = None
        P_e_c2 = None
        correction_alignment_2 = None
        second_decision = None
        enrichment_note = ""

        if lv == 0 and self._root_enrichment_mode != "none":
            r_c2, enrichment_note = self._root_enrichment_rhs(corrected_residual)
            restricted_rhs_norm_2 = self._vector_norm(r_c2)
            e_c2 = self._trace_vcycle_level(lv + 1, r_c2, visits, visit_counter, cycle_type)
            P_e_c2 = self._P_gpu[lv] @ e_c2
            correction_alignment_2 = self._correction_alignment(corrected_residual, P_e_c2)
            second_decision = self._apply_coarse_correction(lv, x_corrected, corrected_residual, P_e_c2)
            x_corrected = second_decision.x
            corrected_residual = second_decision.residual
            corrected_residual_norm = self._vector_norm(corrected_residual)
            n_coarse_corrections = 2

        notes = ""
        if lv == 0:
            x_final = self._smooth_fine(x_corrected, rhs)
            if self._is_high_contrast:
                x_final = self._smooth_fine(x_final, rhs)
                notes = "extra_fine_postsmooth"
        else:
            x_final = self._smooth_coarse(lv, x_corrected, rhs)
        postsmooth_residual = rhs - self._level_matvec(lv, x_final)
        postsmooth_residual_norm = self._vector_norm(postsmooth_residual)

        visits.append(
            MultigridCycleVisit(
                visit_id=visit_id,
                level=lv,
                cycle_type=cycle_type,
                is_coarsest=False,
                n_coarse_corrections=n_coarse_corrections,
                rhs_norm=rhs_norm,
                jacobi_action_norm=jacobi_action_norm,
                presmooth_solution_norm=self._vector_norm(x_smooth),
                presmooth_residual_norm=presmooth_residual_norm,
                presmooth_residual_ratio=self._safe_ratio(presmooth_residual_norm, rhs_norm),
                restricted_rhs_norm=restricted_rhs_norm,
                restricted_rhs_ratio=self._safe_ratio(restricted_rhs_norm, presmooth_residual_norm),
                projector_norm=projector_norm,
                coarse_space_gap_norm=coarse_space_gap_norm,
                coarse_space_gap_ratio=self._safe_ratio(coarse_space_gap_norm, presmooth_residual_norm),
                coarse_correction_norm=self._vector_norm(e_c),
                prolongated_correction_norm=self._vector_norm(P_e_c),
                correction_alignment=correction_alignment,
                correction_scale_raw=correction_decision.raw_scale,
                correction_scale_applied=correction_decision.applied_scale,
                corrected_residual_norm=corrected_residual_norm,
                corrected_residual_vs_presmooth=self._safe_ratio(corrected_residual_norm, presmooth_residual_norm),
                second_restricted_rhs_norm=restricted_rhs_norm_2,
                second_coarse_correction_norm=self._vector_norm(e_c2) if e_c2 is not None else None,
                second_prolongated_correction_norm=self._vector_norm(P_e_c2) if P_e_c2 is not None else None,
                second_correction_alignment=correction_alignment_2,
                second_correction_scale_raw=second_decision.raw_scale if second_decision is not None else None,
                second_correction_scale_applied=(
                    second_decision.applied_scale if second_decision is not None else None
                ),
                second_corrected_residual_norm=self._vector_norm(corrected_residual) if second_decision is not None else None,
                second_corrected_residual_vs_corrected=(
                    self._safe_ratio(self._vector_norm(corrected_residual), first_corrected_residual_norm)
                    if second_decision is not None
                    else None
                ),
                postsmooth_solution_norm=self._vector_norm(x_final),
                postsmooth_residual_norm=postsmooth_residual_norm,
                postsmooth_residual_ratio=self._safe_ratio(postsmooth_residual_norm, rhs_norm),
                notes=self._merge_notes(
                    correction_decision.note,
                    second_decision.note if second_decision is not None else "",
                    enrichment_note,
                    notes,
                ),
            )
        )
        return x_final

    def _trace_wcycle_level(
        self,
        lv: int,
        rhs,
        visits: list[MultigridCycleVisit],
        visit_counter: list[int],
        cycle_type: str,
    ):
        import cupy as cp

        visit_id = visit_counter[0]
        visit_counter[0] += 1

        rhs_norm = self._vector_norm(rhs)
        jacobi_action_norm = self._vector_norm(self._diag_inv_gpu[lv] * rhs)

        if lv == self._n_levels - 1:
            x = self._solve_coarsest(rhs)
            residual = rhs - self._level_matvec(lv, x)
            residual_norm = self._vector_norm(residual)
            visits.append(
                MultigridCycleVisit(
                    visit_id=visit_id,
                    level=lv,
                    cycle_type=cycle_type,
                    is_coarsest=True,
                    n_coarse_corrections=0,
                    rhs_norm=rhs_norm,
                    jacobi_action_norm=jacobi_action_norm,
                    presmooth_solution_norm=0.0,
                    presmooth_residual_norm=rhs_norm,
                    presmooth_residual_ratio=1.0,
                    restricted_rhs_norm=0.0,
                    restricted_rhs_ratio=0.0,
                    projector_norm=0.0,
                    coarse_space_gap_norm=0.0,
                    coarse_space_gap_ratio=0.0,
                    coarse_correction_norm=self._vector_norm(x),
                    prolongated_correction_norm=self._vector_norm(x),
                    correction_alignment=0.0,
                    correction_scale_raw=None,
                    correction_scale_applied=1.0,
                    corrected_residual_norm=residual_norm,
                    corrected_residual_vs_presmooth=self._safe_ratio(residual_norm, rhs_norm),
                    postsmooth_solution_norm=self._vector_norm(x),
                    postsmooth_residual_norm=residual_norm,
                    postsmooth_residual_ratio=self._safe_ratio(residual_norm, rhs_norm),
                    coarsest_mode=self._coarse_chol_mode,
                )
            )
            return x

        x0 = cp.zeros_like(rhs)
        if lv == 0:
            x_smooth = self._smooth_fine(x0, rhs)
        else:
            x_smooth = self._smooth_coarse(lv, x0, rhs)
        presmooth_residual = rhs - self._level_matvec(lv, x_smooth)
        presmooth_residual_norm = self._vector_norm(presmooth_residual)

        r_c = self._R_gpu[lv] @ presmooth_residual
        restricted_rhs_norm = self._vector_norm(r_c)
        projector = self._P_gpu[lv] @ r_c
        projector_norm = self._vector_norm(projector)
        coarse_space_gap = presmooth_residual - projector
        coarse_space_gap_norm = self._vector_norm(coarse_space_gap)

        e_c = self._trace_wcycle_level(lv + 1, r_c, visits, visit_counter, cycle_type)
        P_e_c = self._P_gpu[lv] @ e_c
        correction_alignment = self._correction_alignment(presmooth_residual, P_e_c)
        first_decision = self._apply_coarse_correction(lv, x_smooth, presmooth_residual, P_e_c)
        x_first = first_decision.x
        corrected_residual = first_decision.residual
        corrected_residual_norm = self._vector_norm(corrected_residual)

        enrichment_note = ""
        if lv == 0 and self._root_enrichment_mode != "none":
            r_c2, enrichment_note = self._root_enrichment_rhs(corrected_residual)
        else:
            r_c2 = self._R_gpu[lv] @ corrected_residual
        restricted_rhs_norm_2 = self._vector_norm(r_c2)
        e_c2 = self._trace_wcycle_level(lv + 1, r_c2, visits, visit_counter, cycle_type)
        P_e_c2 = self._P_gpu[lv] @ e_c2
        correction_alignment_2 = self._correction_alignment(corrected_residual, P_e_c2)
        second_decision = self._apply_coarse_correction(lv, x_first, corrected_residual, P_e_c2)
        x_second = second_decision.x
        corrected_residual_2 = second_decision.residual
        corrected_residual_norm_2 = self._vector_norm(corrected_residual_2)

        notes = ""
        if lv == 0:
            x_final = self._smooth_fine(x_second, rhs)
            if self._is_high_contrast:
                x_final = self._smooth_fine(x_final, rhs)
                notes = "extra_fine_postsmooth"
        else:
            x_final = self._smooth_coarse(lv, x_second, rhs)
        postsmooth_residual = rhs - self._level_matvec(lv, x_final)
        postsmooth_residual_norm = self._vector_norm(postsmooth_residual)

        visits.append(
            MultigridCycleVisit(
                visit_id=visit_id,
                level=lv,
                cycle_type=cycle_type,
                is_coarsest=False,
                n_coarse_corrections=2,
                rhs_norm=rhs_norm,
                jacobi_action_norm=jacobi_action_norm,
                presmooth_solution_norm=self._vector_norm(x_smooth),
                presmooth_residual_norm=presmooth_residual_norm,
                presmooth_residual_ratio=self._safe_ratio(presmooth_residual_norm, rhs_norm),
                restricted_rhs_norm=restricted_rhs_norm,
                restricted_rhs_ratio=self._safe_ratio(restricted_rhs_norm, presmooth_residual_norm),
                projector_norm=projector_norm,
                coarse_space_gap_norm=coarse_space_gap_norm,
                coarse_space_gap_ratio=self._safe_ratio(coarse_space_gap_norm, presmooth_residual_norm),
                coarse_correction_norm=self._vector_norm(e_c),
                prolongated_correction_norm=self._vector_norm(P_e_c),
                correction_alignment=correction_alignment,
                correction_scale_raw=first_decision.raw_scale,
                correction_scale_applied=first_decision.applied_scale,
                corrected_residual_norm=corrected_residual_norm,
                corrected_residual_vs_presmooth=self._safe_ratio(corrected_residual_norm, presmooth_residual_norm),
                second_restricted_rhs_norm=restricted_rhs_norm_2,
                second_coarse_correction_norm=self._vector_norm(e_c2),
                second_prolongated_correction_norm=self._vector_norm(P_e_c2),
                second_correction_alignment=correction_alignment_2,
                second_correction_scale_raw=second_decision.raw_scale,
                second_correction_scale_applied=second_decision.applied_scale,
                second_corrected_residual_norm=corrected_residual_norm_2,
                second_corrected_residual_vs_corrected=self._safe_ratio(corrected_residual_norm_2, corrected_residual_norm),
                postsmooth_solution_norm=self._vector_norm(x_final),
                postsmooth_residual_norm=postsmooth_residual_norm,
                postsmooth_residual_ratio=self._safe_ratio(postsmooth_residual_norm, rhs_norm),
                notes=self._merge_notes(first_decision.note, second_decision.note, enrichment_note, notes),
            )
        )
        return x_final

    def _coerce_rhs64(self, rhs):
        import cupy as cp

        if hasattr(rhs, "astype"):
            return rhs.astype(np.float64, copy=False)
        return cp.asarray(rhs, dtype=np.float64)

    def _apply_base_cycle(self, rhs64):
        if self._cycle_type == "w":
            return self._wcycle(0, rhs64)
        return self._vcycle(0, rhs64)

    def _apply_inner_krylov(self, rhs64):
        x, _, _ = _cupy_fgmres(
            self.apply_fine_operator,
            rhs64,
            self._apply_base_cycle,
            tol=0.0,
            maxiter=self._inner_krylov_steps,
            restart=self._inner_krylov_restart,
        )
        return x

    def _apply_root_local_correction_to_output(self, rhs64, x):
        if self._root_local_correction_mode == "none":
            return x
        residual = rhs64 - self.apply_fine_operator(x)
        return self._apply_root_local_correction(x, residual).x

    def apply(self, rhs):
        if not self._is_setup:
            raise RuntimeError("GalerkinMatFreeGMG.setup() must be called before apply().")
        rhs64 = self._coerce_rhs64(rhs)
        if self._inner_krylov_steps > 0:
            x = self._apply_inner_krylov(rhs64)
        else:
            x = self._apply_base_cycle(rhs64)
        return self._apply_root_local_correction_to_output(rhs64, x)

    def _fine_matvec_low_precision(self, v32):
        if self._fine_smoother == "fp32":
            if self._fused_op is not None:
                return self._fused_op.matvec(v32, self._E_e_fp32, self._free_gpu, dtype="fp32")
            return self._mf_op.matvec(v32, self._E_e_fp32)
        if self._fine_smoother == "bf16":
            if self._fused_op is None:
                raise RuntimeError(
                    "BF16 fine smoothing requires enable_fused_cuda=True so the WMMA path exists."
                )
            return self._fused_op.matvec(v32, self._E_e_fp32, self._free_gpu, dtype="bf16")
        raise ValueError(f"Unsupported fine smoother '{self._fine_smoother}'")

    # ── Chebyshev helper (Saad Algorithm 12.1 with M^{-1}=D^{-1}) ─────────────
    @staticmethod
    def _chebyshev_steps(A_op, rhs, x0, d_inv, lam_max, lower_frac, degree):
        """
        Apply 'degree' steps of Chebyshev-Jacobi smoothing.

        Targets eigenvalues of D^{-1}A in [lower_frac*lam_max, lam_max].
        Implements Saad (2003) Algorithm 12.1 with M^{-1}=D^{-1}.
        Array-type agnostic: works with numpy, cupy, or any array supporting
        arithmetic operators — no explicit cupy import needed.
        """
        lam_min = lower_frac * lam_max
        sigma = (lam_max + lam_min) * 0.5
        delta = (lam_max - lam_min) * 0.5

        x = x0
        r = rhs - A_op(x)
        z = d_inv * r
        alpha = 2.0 / (lam_max + lam_min)   # = 1/sigma
        d_vec = z
        x = x + alpha * d_vec

        for _ in range(1, degree):
            r = rhs - A_op(x)
            z = d_inv * r
            alpha_new = 1.0 / (sigma - (delta ** 2) * alpha * 0.25)
            d_vec = alpha_new * (z + (delta ** 2) * alpha * 0.25 * d_vec)
            x = x + d_vec
            alpha = alpha_new

        return x

    def _smooth_fine(self, x, rhs):
        import cupy as cp

        prec = self._level_prec[0] if self._level_prec else self._fine_smoother
        lm = self._lambda_max_est

        if self._smoother_type == "chebyshev":
            if prec == "fp64":
                # Adaptive Chebyshev interval for high-contrast fields
                lf = self._cheb_lower_frac
                if self._is_high_contrast:
                    # Widen targeting interval to capture low-frequency error
                    lf = min(self._cheb_lower_frac, 1.0 / 100.0)
                return self._chebyshev_steps(
                    self.apply_fine_operator, rhs,
                    x.astype(cp.float64, copy=True),
                    self._diag_inv_gpu[0], lm, lf, self._fine_degree_adapt,
                )
            # fp32 or bf16: run Chebyshev in low precision, cast back
            x32 = x.astype(cp.float32, copy=True)
            rhs32 = rhs.astype(cp.float32, copy=False)
            d_inv32 = self._diag_inv_gpu[0].astype(cp.float32, copy=False)
            lm32 = cp.float32(lm)
            # Adaptive interval for high-contrast fields
            lf = self._cheb_lower_frac
            if self._is_high_contrast:
                lf = min(self._cheb_lower_frac, 1.0 / 100.0)
            lf32 = cp.float32(lf)

            def _A32(v):
                return self._fine_matvec_low_precision(v)

            return self._chebyshev_steps(
                _A32, rhs32, x32, d_inv32, lm32, lf32, self._fine_degree_adapt,
            ).astype(cp.float64)

        # Jacobi fallback
        if prec == "fp64":
            x_curr = x.astype(cp.float64, copy=True)
            for _ in range(self._fine_degree_adapt):
                res = rhs - self.apply_fine_operator(x_curr)
                x_curr = x_curr + self._omega_used * self._diag_inv_gpu[0] * res
            return x_curr

        x32 = x.astype(cp.float32, copy=True)
        rhs32 = rhs.astype(cp.float32, copy=False)
        d_inv32 = self._diag_inv_gpu[0].astype(cp.float32, copy=False)
        for _ in range(self._fine_degree_adapt):
            res32 = rhs32 - self._fine_matvec_low_precision(x32)
            x32 = x32 + self._omega_used * d_inv32 * res32
        return x32.astype(cp.float64)

    def _smooth_coarse(self, lv: int, x, rhs):
        import cupy as cp

        prec = self._level_prec[lv] if self._level_prec and lv < len(self._level_prec) else "fp64"
        lm = self._lambda_max_per_level[lv] if lv < len(self._lambda_max_per_level) else 1.0
        lm = lm if np.isfinite(lm) else 1.0

        # Adaptive Chebyshev interval and smoothing strength
        lf = self._cheb_lower_frac
        adapt_iters = self._coarse_smooth_iters_adapt
        if self._is_high_contrast:
            # Widen targeting interval at coarse levels for high-contrast fields
            lf = min(self._cheb_lower_frac, 1.0 / 100.0)

        if prec == "fp32" and self._K_fp32_gpu[lv] is not None:
            K32 = self._K_fp32_gpu[lv]
            d_inv32 = self._diag_inv_fp32_gpu[lv]
            x32 = x.astype(cp.float32, copy=True)
            rhs32 = rhs.astype(cp.float32, copy=False)
            if self._smoother_type == "chebyshev":
                x32 = self._chebyshev_steps(
                    lambda v: K32 @ v, rhs32, x32, d_inv32,
                    cp.float32(lm), cp.float32(lf), adapt_iters,
                )
            else:
                for _ in range(adapt_iters):
                    x32 = x32 + self._omega_used * d_inv32 * (rhs32 - K32 @ x32)
            return x32.astype(cp.float64)

        # FP64 path
        K = self._K_gpu[lv]
        d_inv = self._diag_inv_gpu[lv]
        if self._smoother_type == "chebyshev":
            return self._chebyshev_steps(
                lambda v: K @ v, rhs, x, d_inv,
                lm, lf, adapt_iters,
            )
        x_curr = x
        for _ in range(adapt_iters):
            x_curr = x_curr + self._omega_used * d_inv * (rhs - K @ x_curr)
        return x_curr

    def _solve_coarsest(self, rhs):
        import cupy as cp
        import cupyx.scipy.linalg as cpla

        if self._n_levels == 1:
            return self._smooth_fine(cp.zeros_like(rhs), rhs)
        lv = self._n_levels - 1
        if self._coarse_chol_mode == "dense":
            y = cpla.solve_triangular(self._coarse_chol_L, rhs, lower=True)
            return cpla.solve_triangular(self._coarse_chol_L, y, lower=True, trans="T")
        # For high-contrast cases, increase iterative coarse solve budget
        adapt_iters = self._coarse_pcg_iters
        if self._is_high_contrast:
            adapt_iters = max(self._coarse_pcg_iters, 200)
        return _fixed_pcg_coarse(
            self._K_gpu[lv],
            rhs,
            self._diag_inv_gpu[lv],
            n_iters=adapt_iters,
        )

    def _vcycle(self, lv: int, rhs):
        import cupy as cp

        if lv == self._n_levels - 1:
            return self._solve_coarsest(rhs)

        if lv == 0:
            x = self._smooth_fine(cp.zeros_like(rhs), rhs)
            residual = rhs - self.apply_fine_operator(x)
        else:
            x = self._smooth_coarse(lv, cp.zeros_like(rhs), rhs)
            residual = rhs - self._K_gpu[lv] @ x

        r_c = self._R_gpu[lv] @ residual
        e_c = self._vcycle(lv + 1, r_c)
        P_e_c = self._P_gpu[lv] @ e_c
        if self._coarse_correction_policy == "none":
            x = x + P_e_c
            corrected_residual = residual - self._level_matvec(lv, P_e_c)
        else:
            first_decision = self._apply_coarse_correction(lv, x, residual, P_e_c)
            x = first_decision.x
            corrected_residual = first_decision.residual

        if lv == 0 and self._root_enrichment_mode != "none":
            r_c2, _ = self._root_enrichment_rhs(corrected_residual)
            e_c2 = self._vcycle(lv + 1, r_c2)
            P_e_c2 = self._P_gpu[lv] @ e_c2
            if self._coarse_correction_policy == "none":
                x = x + P_e_c2
            else:
                x = self._apply_coarse_correction(lv, x, corrected_residual, P_e_c2).x

        if lv == 0:
            x = self._smooth_fine(x, rhs)
        else:
            x = self._smooth_coarse(lv, x, rhs)

        # Pathology-aware post-refinement for high-contrast fields at level 0
        # If the coarse correction didn't reduce residual enough, apply
        # an extra fine-level smoothing pass to clean up high-frequency error.
        if lv == 0 and self._is_high_contrast:
            x = self._smooth_fine(x, rhs)

        return x

    def _wcycle(self, lv: int, rhs):
        """W-cycle: two coarse corrections per level (γ=2)."""
        import cupy as cp

        if lv == self._n_levels - 1:
            return self._solve_coarsest(rhs)

        if lv == 0:
            x = self._smooth_fine(cp.zeros_like(rhs), rhs)
            residual = rhs - self.apply_fine_operator(x)
        else:
            x = self._smooth_coarse(lv, cp.zeros_like(rhs), rhs)
            residual = rhs - self._K_gpu[lv] @ x

        r_c = self._R_gpu[lv] @ residual
        e_c = self._wcycle(lv + 1, r_c)
        P_e_c = self._P_gpu[lv] @ e_c
        if self._coarse_correction_policy == "none":
            x = x + P_e_c
            if lv == 0:
                residual2 = rhs - self.apply_fine_operator(x)
            else:
                residual2 = rhs - self._K_gpu[lv] @ x
        else:
            first_decision = self._apply_coarse_correction(lv, x, residual, P_e_c)
            x = first_decision.x
            residual2 = first_decision.residual

        # Second coarse correction (γ=2 distinguishes W from V)
        if lv == 0 and self._root_enrichment_mode != "none":
            r_c2, _ = self._root_enrichment_rhs(residual2)
        else:
            r_c2 = self._R_gpu[lv] @ residual2
        e_c2 = self._wcycle(lv + 1, r_c2)
        P_e_c2 = self._P_gpu[lv] @ e_c2
        if self._coarse_correction_policy == "none":
            x = x + P_e_c2
        else:
            x = self._apply_coarse_correction(lv, x, residual2, P_e_c2).x

        if lv == 0:
            x = self._smooth_fine(x, rhs)
        else:
            x = self._smooth_coarse(lv, x, rhs)

        # Pathology-aware post-refinement for high-contrast fields at level 0
        if lv == 0 and self._is_high_contrast:
            x = self._smooth_fine(x, rhs)

        return x
