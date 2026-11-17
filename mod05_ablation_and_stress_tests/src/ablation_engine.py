"""Ablation Study Engine for IEEE Sensors Journal NUC Project (Section IV-C).

Investigates the individual contribution of each core component of the proposed framework:
  1. Component A: Without Second-Order Directional TV (|d^2 C / dx^2|, lambda_3 = 0).
  2. Component B: Without Schatten-1/2 Norm (using standard Nuclear Norm for S).
  3. Component C: Without Stage 2 Non-Local Patch Low-Rank SVD (Stage 1 only).
  4. Component D: Complete Proposed Pipeline (Stage 1 with High-Order TV + Schatten-1/2 + Stage 2 SVD).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from common.metrics import evaluate_all_metrics
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod02_stage1_admm_destriping.src.operators import (
    compute_fft_eigenvalues,
    d2_dx2,
    grad_x,
    grad_y,
)
from mod02_stage1_admm_destriping.src.proximal_ista import (
    project_column_stripe,
    soft_threshold,
)
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank


def _svd_nuclear_norm_shrinkage(Y: np.ndarray, tau: float) -> np.ndarray:
    """Standard Nuclear Norm singular value thresholding (L1 on singular values)."""
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return (U * s_thresh) @ Vt


def solve_stage1_nuclear_norm(
    D: np.ndarray,
    lambda1: float = 0.5,
    lambda2: float = 0.05,
    lambda3: float = 2.0,
    rho: float = 0.5,
    maxiter: int = 50,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 1 ADMM solver substituting Schatten-1/2 with standard Nuclear Norm."""
    arr_D = np.asarray(D, dtype=np.float64)
    H, W = arr_D.shape

    eig_x, eig_y, eig_xx = compute_fft_eigenvalues((H, W))
    denom = 1.0 + rho * (np.abs(eig_x) ** 2 + np.abs(eig_y) ** 2) + rho * np.abs(eig_xx) ** 2

    C = np.copy(arr_D)
    S = np.zeros((H, W), dtype=np.float64)

    V1 = np.zeros_like(arr_D)
    V2 = np.zeros_like(arr_D)
    V3 = np.zeros_like(arr_D)

    U1 = np.zeros_like(arr_D)
    U2 = np.zeros_like(arr_D)
    U3 = np.zeros_like(arr_D)
    U_s = np.zeros_like(arr_D)

    for _ in range(maxiter):
        C_prev = np.copy(C)

        # 1. Update S with standard Nuclear Norm
        M_s = arr_D - C + (1.0 / rho) * U_s
        S = _svd_nuclear_norm_shrinkage(M_s, lambda2 / rho)
        S = project_column_stripe(S)

        # 2. Update C via 2D FFT
        rhs = (
            (arr_D - S)
            + rho
            * np.fft.ifft2(
                np.conj(eig_x) * np.fft.fft2(V1 - U1)
                + np.conj(eig_y) * np.fft.fft2(V2 - U2)
                + np.conj(eig_xx) * np.fft.fft2(V3 - U3)
            ).real
        )
        C = np.fft.ifft2(np.fft.fft2(rhs) / denom).real

        # 3. Gradients
        gx = grad_x(C)
        gy = grad_y(C)
        gxx = d2_dx2(C)

        # 4. Proximal updates for auxiliary variables
        V1 = soft_threshold(gx + U1, lambda1 / rho)
        V2 = soft_threshold(gy + U2, lambda1 / rho)
        V3 = soft_threshold(gxx + U3, lambda3 / rho)

        # 5. Multipliers
        U1 += gx - V1
        U2 += gy - V2
        U3 += gxx - V3
        U_s += rho * (arr_D - C - S)

        if np.linalg.norm(C - C_prev) / (np.linalg.norm(C_prev) + 1e-8) < tol:
            break

    return np.clip(C, 0.0, 255.0), S


def run_ablation_study(
    clean_img: np.ndarray,
    degraded_img: np.ndarray,
    fast_mode: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run the 4 ablation configurations on a given scene.

    Returns
    -------
    ablation_dict : Dict[config_name, Dict[str, Any]]
        Dictionary containing restored images, metrics, and runtime.
    """
    clean_64 = clean_img.astype(np.float64)
    deg_64 = degraded_img.astype(np.float64)
    stride = 6 if fast_mode else 3

    results: Dict[str, Dict[str, Any]] = {}

    # 1. Config A: Without Second-Order TV (lambda3 = 0.0)
    t0 = time.perf_counter()
    c_no_horder, s_no_horder, _ = solve_stage1_admm(
        deg_64,
        lambda1=0.5,
        lambda2=0.05,
        lambda3=0.0,  # Zero high-order TV weight
        rho=0.5,
        maxiter=40,
    )
    c_no_horder_s2 = denoise_stage2_patch_lowrank(
        c_no_horder,
        patch_size=11,
        stride=stride,
        search_radius=12,
        num_similar_patches=50,
        n_jobs=-1,
    )
    t_no_horder = time.perf_counter() - t0
    results["w/o High-Order TV"] = {
        "restored": c_no_horder_s2,
        "metrics": evaluate_all_metrics(clean_64, c_no_horder_s2),
        "time": t_no_horder,
    }

    # 2. Config B: Without Schatten-1/2 (Nuclear Norm)
    t0 = time.perf_counter()
    c_nuclear, s_nuclear = solve_stage1_nuclear_norm(
        deg_64,
        lambda1=0.5,
        lambda2=0.05,
        lambda3=2.0,
        rho=0.5,
        maxiter=40,
    )
    c_nuclear_s2 = denoise_stage2_patch_lowrank(
        c_nuclear,
        patch_size=11,
        stride=stride,
        search_radius=12,
        num_similar_patches=50,
        n_jobs=-1,
    )
    t_nuclear = time.perf_counter() - t0
    results["w/o Schatten-1/2 (Nuclear Norm)"] = {
        "restored": c_nuclear_s2,
        "metrics": evaluate_all_metrics(clean_64, c_nuclear_s2),
        "time": t_nuclear,
    }

    # 3. Config C: Without Stage 2 (Stage 1 Only)
    t0 = time.perf_counter()
    c_s1, s_s1, _ = solve_stage1_admm(
        deg_64,
        lambda1=0.5,
        lambda2=0.05,
        lambda3=2.0,
        rho=0.5,
        maxiter=40,
    )
    t_s1 = time.perf_counter() - t0
    results["w/o Stage 2 (Stage 1 Only)"] = {
        "restored": c_s1,
        "metrics": evaluate_all_metrics(clean_64, c_s1),
        "time": t_s1,
    }

    # 4. Config D: Proposed Full Pipeline
    t0 = time.perf_counter()
    c_full = denoise_stage2_patch_lowrank(
        c_s1,
        patch_size=11,
        stride=stride,
        search_radius=12,
        num_similar_patches=50,
        n_jobs=-1,
    )
    t_full = (time.perf_counter() - t0) + t_s1
    results["Proposed Full Pipeline"] = {
        "restored": c_full,
        "metrics": evaluate_all_metrics(clean_64, c_full),
        "time": t_full,
    }

    return results
