"""Unified Wrapper for the Proposed Two-Stage Single-Frame NUC Pipeline.

Chains:
  - Stage 1: ADMM High-Order Directional Destriping (mod02, Eqs. 11-21)
  - Stage 2: Non-Local Patch Low-Rank SVD Denoiser (mod03, Eqs. 22-26)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple
import numpy as np

from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank


def solve_proposed_ours(
    D: np.ndarray,
    lambda1: float = 0.5,
    lambda2: float = 0.05,
    lambda3: float = 2.0,
    rho: float = 0.5,
    maxiter_s1: int = 40,
    patch_size: int = 11,
    stride: int = 3,
    search_radius: int = 12,
    L: int = 50,
    fast_mode: bool = False,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Execute complete two-stage NUC pipeline (Ours).

    Parameters
    ----------
    D : np.ndarray
        Degraded input infrared image of shape (H, W), float64, in [0, 255].
    lambda1 : float
        Schatten-1/2 low-rank penalty weight on stripe matrix S.
    lambda2 : float
        First-order Total Variation penalty weight ||grad(C)||_1.
    lambda3 : float
        Second-order horizontal directional TV penalty weight ||d2_dx2(C)||_1.
    rho : float
        ADMM augmented Lagrangian quadratic penalty parameter.
    maxiter_s1 : int
        Maximum iterations for Stage 1 ADMM solver.
    patch_size : int
        Stage 2 patch size B = 11.
    stride : int
        Stage 2 reference patch stride (default 3; 6 if fast_mode).
    search_radius : int
        Stage 2 local search window half-width R = 12.
    L : int
        Number of grouped similar patches L = 50.
    fast_mode : bool
        If True, uses larger stride for fast unit testing.
    n_jobs : int
        Number of parallel worker threads (-1 for all cores).

    Returns
    -------
    C_ours : np.ndarray
        Final restored thermal image in [0, 255].
    S_ours : np.ndarray
        Estimated total corruptions (stripe + environmental noise).
    diagnostics : Dict[str, Any]
        Timing, stage breakdown, and convergence diagnostics.
    """
    img = np.asarray(D, dtype=np.float64)
    s2_stride = 6 if fast_mode else stride

    t_start = time.perf_counter()

    # Stage 1: ADMM High-Order Directional Destriping
    t0_s1 = time.perf_counter()
    C_s1, S_s1, history_s1 = solve_stage1_admm(
        img,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        rho=rho,
        maxiter=maxiter_s1,
    )
    t_s1 = time.perf_counter() - t0_s1

    # Stage 2: Non-Local Patch Low-Rank SVD
    t0_s2 = time.perf_counter()
    C_ours, diag_s2 = denoise_stage2_patch_lowrank(
        C_s1,
        patch_size=patch_size,
        stride=s2_stride,
        search_radius=search_radius,
        num_similar_patches=L,
        n_jobs=n_jobs,
        return_diagnostics=True,
    )
    t_s2 = time.perf_counter() - t0_s2

    t_total = time.perf_counter() - t_start
    S_ours = img - C_ours

    diagnostics = {
        "stage1_time_sec": t_s1,
        "stage2_time_sec": t_s2,
        "total_time_sec": t_total,
        "stage1_iterations": len(history_s1),
        "stage2_mean_rank": diag_s2.get("mean_rank", 0.0),
        "C_stage1": C_s1,
        "S_stage1": S_s1,
    }

    return C_ours, S_ours, diagnostics
