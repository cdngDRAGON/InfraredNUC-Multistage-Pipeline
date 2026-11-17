"""mod02_stage1_admm_destriping/src/admm_stage1_solver.py

Stage 1 ADMM Destriping Solver (Section III-C, Eqs. 5-21).

Mathematical Formulation:
    min_{C, S} 0.5 * ||D - (C + S)||_F^2
               + lambda1 * ||S||_{S_{1/2}}^{1/2}
               + lambda2 * ||grad(C)||_1
               + lambda3 * ||d2_dx2(C)||_1

Note on Eq. (7) vs. Eq. (11) sign convention:
    Physical degradation model (Eq. 5 & Eq. 7) defines the residual D - (C + S) = D - C - S
    matching Fig. 6 where C = D - S. Eq. (11) in the published text contains a typographical +C;
    the code adheres to the mathematically consistent data fidelity 0.5 * ||D - C - S||_F^2.

Variable Splitting (Eq. 12):
    - Z = sigma(S)^{1/2}   (low-rank Schatten 1/2 prior on stripe S)
    - G = grad(C)          (first-order total variation)
    - H = d2_dx2(C)        (second-order horizontal directional prior)

Augmented Lagrangian Updates (Eqs. 14-20):
    1. C^{k+1} (Eq. 15): Fast FFT-diagonalized closed-form solver.
    2. S^{k+1}, Z^{k+1} (Eqs. 16-17): Schatten 1/2 SVD shrinkage + column-mean projection.
    3. G^{k+1}, H^{k+1} (Eqs. 18-19): ISTA L1 soft-thresholding.
    4. Dual Multipliers (Eq. 20): Subgradient ascent updates.
    5. Convergence (Eq. 21): Relative change ||C^{k+1} - C^k||_F / ||C^k||_F <= tol.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from common.metrics import compute_psnr, compute_ssim
from mod02_stage1_admm_destriping.src.operators import (
    compute_fft_eigenvalues,
    d2_dx2,
    d2_dx2_adj,
    grad_x,
    grad_x_adj,
    grad_y,
    grad_y_adj,
)
from mod02_stage1_admm_destriping.src.proximal_ista import (
    project_column_stripe,
    schatten_half_shrinkage,
    soft_threshold,
)


def solve_stage1_admm(
    D: np.ndarray,
    lambda1: float = 0.5,
    lambda2: float = 0.05,
    lambda3: float = 2.0,
    rho: float = 0.5,
    maxiter: int = 100,
    tol: float = 1e-4,
    clean_ref: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Solve Stage 1 ADMM destriping problem (Eqs. 11-21).

    Parameters
    ----------
    D : np.ndarray
        Degraded input infrared image of shape (M, N) with values in [0, 255].
    lambda1 : float
        Schatten 1/2 low-rank penalty weight on stripe matrix S. Default 0.5.
    lambda2 : float
        First-order Total Variation penalty weight ||grad(C)||_1. Default 0.05.
    lambda3 : float
        Second-order horizontal directional TV penalty weight ||d2_dx2(C)||_1. Default 2.0.
    rho : float
        ADMM augmented Lagrangian quadratic penalty parameter. Default 0.5.
    maxiter : int
        Maximum number of ADMM iterations. Default 100.
    tol : float
        Relative Frobenius norm convergence tolerance. Default 1e-4.
    clean_ref : Optional[np.ndarray]
        Ground-truth clean image C for tracking PSNR/SSIM convergence trajectory.
    verbose : bool
        If True, prints progress per iteration.

    Returns
    -------
    C_stage1 : np.ndarray
        Destriped intermediate infrared image estimate in [0, 255].
    S_est : np.ndarray
        Estimated vertical stripe noise matrix.
    history_df : pd.DataFrame
        Iteration telemetry recording [iteration, rel_change, psnr, ssim, obj_val, elapsed_sec].
    """
    D_arr = np.asarray(D, dtype=np.float64)
    M, N = D_arr.shape

    # 1. Precompute FFT eigenvalues for BCCC linear system inversion (Eq. 15)
    eig_gx_sq, eig_gy_sq, eig_hxx_sq = compute_fft_eigenvalues((M, N))
    fft_denom = 1.0 + rho * (eig_gx_sq + eig_gy_sq) + rho * eig_hxx_sq

    # 2. Variable Initialization
    C = np.copy(D_arr)
    S = np.zeros((M, N), dtype=np.float64)

    # Auxiliary variables: G = [Gx, Gy] and H = d2_dx2(C)
    Gx = grad_x(C)
    Gy = grad_y(C)
    H = d2_dx2(C)

    # Dual Lagrange multipliers
    Lambda_Gx = np.zeros((M, N), dtype=np.float64)
    Lambda_Gy = np.zeros((M, N), dtype=np.float64)
    Lambda_H = np.zeros((M, N), dtype=np.float64)

    history: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for it in range(1, maxiter + 1):
        C_prev = np.copy(C)

        # ----------------------------------------------------------------------
        # Subproblem 1: Update S^{k+1} via Schatten 1/2 Shrinkage (Eqs. 16-17)
        # ----------------------------------------------------------------------
        residual = D_arr - C
        S_svd, _ = schatten_half_shrinkage(
            residual,
            tau=lambda1 / rho,
            rank_limit=2,
        )
        # Physical column-correlated subspace projection
        S_col = project_column_stripe(S_svd)
        S = 0.95 * S_col + 0.05 * S_svd

        # ----------------------------------------------------------------------
        # Subproblem 2: Update C^{k+1} via 2D FFT (Eq. 15)
        # ----------------------------------------------------------------------
        v_gx = Gx - (Lambda_Gx / rho)
        v_gy = Gy - (Lambda_Gy / rho)
        v_h = H - (Lambda_H / rho)

        div_g = grad_x_adj(v_gx) + grad_y_adj(v_gy)
        adj_h = d2_dx2_adj(v_h)

        rhs_spatial = (D_arr - S) + rho * div_g + rho * adj_h
        rhs_freq = np.fft.fft2(rhs_spatial)

        # Closed-form Fourier inversion
        C = np.real(np.fft.ifft2(rhs_freq / fft_denom))

        # ----------------------------------------------------------------------
        # Subproblem 3: Update G^{k+1} and H^{k+1} via ISTA (Eqs. 18-19)
        # ----------------------------------------------------------------------
        # First-order gradient auxiliary update (Eq. 18)
        u_gx = grad_x(C) + (Lambda_Gx / rho)
        u_gy = grad_y(C) + (Lambda_Gy / rho)
        Gx = soft_threshold(u_gx, threshold=lambda2 / rho)
        Gy = soft_threshold(u_gy, threshold=(lambda2 * 0.2) / rho)

        # Second-order horizontal directional auxiliary update (Eq. 19)
        u_h = d2_dx2(C) + (Lambda_H / rho)
        H = soft_threshold(u_h, threshold=lambda3 / rho)

        # ----------------------------------------------------------------------
        # Subproblem 4: Multiplier Updates (Eq. 20)
        # ----------------------------------------------------------------------
        res_gx = grad_x(C) - Gx
        res_gy = grad_y(C) - Gy
        res_h = d2_dx2(C) - H

        Lambda_Gx += rho * res_gx
        Lambda_Gy += rho * res_gy
        Lambda_H += rho * res_h

        # ----------------------------------------------------------------------
        # Convergence & Telemetry Check (Eq. 21)
        # ----------------------------------------------------------------------
        diff_norm = np.linalg.norm(C - C_prev, "fro")
        prev_norm = np.linalg.norm(C_prev, "fro") + 1e-10
        rel_change = float(diff_norm / prev_norm)

        elapsed = time.perf_counter() - start_time

        # Calculate metrics if clean reference provided
        current_psnr = compute_psnr(clean_ref, C) if clean_ref is not None else float("nan")
        current_ssim = compute_ssim(clean_ref, C) if clean_ref is not None else float("nan")

        # Objective function value (Eq. 11 with consistent D - C - S residual)
        fidelity = 0.5 * float(np.sum((D_arr - C - S) ** 2))
        _, s_val, _ = np.linalg.svd(S, full_matrices=False)
        reg_s = lambda1 * float(np.sum(np.sqrt(np.maximum(s_val, 0.0))))
        reg_c1 = lambda2 * float(np.sum(np.abs(grad_x(C))) + np.sum(np.abs(grad_y(C))))
        reg_c2 = lambda3 * float(np.sum(np.abs(d2_dx2(C))))
        total_obj = fidelity + reg_s + reg_c1 + reg_c2

        record = {
            "iteration": it,
            "rel_change": rel_change,
            "objective": total_obj,
            "psnr": current_psnr,
            "ssim": current_ssim,
            "elapsed_sec": elapsed,
        }
        history.append(record)

        if verbose and (it % 10 == 0 or it == 1):
            psnr_str = f" | PSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}" if clean_ref is not None else ""
            print(f"  [ADMM Iter {it:3d}/{maxiter}] RelChange: {rel_change:.2e} | Obj: {total_obj:.2e}{psnr_str}")

        if it > 5 and rel_change <= tol:
            if verbose:
                print(f"  ADMM converged at iteration {it} with relative change {rel_change:.2e} <= {tol:.2e}")
            break

    history_df = pd.DataFrame(history)
    C_stage1 = np.clip(C, 0.0, 255.0)

    return C_stage1, S, history_df
