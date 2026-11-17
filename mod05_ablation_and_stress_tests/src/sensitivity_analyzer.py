"""Parameter Sensitivity Analysis for IEEE Sensors Journal NUC Project.

Analyzes performance response and robustness across key hyperparameters:
  - Second-order directional TV weight: lambda_3 in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
  - Stage 2 patch size: B in [7, 9, 11, 13, 15]
  - Stage 2 number of similar patches: L in [20, 35, 50, 75, 100]
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common.metrics import compute_psnr, compute_roughness_rho, compute_ssim
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank


def evaluate_lambda3_sensitivity(
    clean_img: np.ndarray,
    degraded_img: np.ndarray,
    lambda3_values: List[float] = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0],
    stride: int = 4,
) -> pd.DataFrame:
    """Evaluate PSNR, SSIM, and Roughness as a function of lambda_3."""
    clean_64 = clean_img.astype(np.float64)
    deg_64 = degraded_img.astype(np.float64)

    rows = []
    for l3 in lambda3_values:
        t0 = time.perf_counter()
        c_s1, _, _ = solve_stage1_admm(
            deg_64,
            lambda1=0.5,
            lambda2=0.05,
            lambda3=l3,
            rho=0.5,
            maxiter=35,
        )
        c_full = denoise_stage2_patch_lowrank(
            c_s1,
            patch_size=11,
            stride=stride,
            search_radius=12,
            num_similar_patches=50,
            n_jobs=-1,
        )
        elapsed = time.perf_counter() - t0

        psnr_val = compute_psnr(clean_64, c_full)
        ssim_val = compute_ssim(clean_64, c_full)
        rho_val = compute_roughness_rho(c_full)

        rows.append(
            {
                "Parameter": "lambda_3",
                "Value": l3,
                "PSNR_dB": psnr_val,
                "SSIM": ssim_val,
                "Roughness_rho": rho_val,
                "Time_sec": elapsed,
            }
        )
    return pd.DataFrame(rows)


def evaluate_patch_size_sensitivity(
    c_s1: np.ndarray,
    clean_img: np.ndarray,
    patch_sizes: List[int] = [7, 9, 11, 13, 15],
    stride: int = 4,
) -> pd.DataFrame:
    """Evaluate performance as a function of Stage 2 patch size B."""
    clean_64 = clean_img.astype(np.float64)

    rows = []
    for B in patch_sizes:
        t0 = time.perf_counter()
        c_denoised = denoise_stage2_patch_lowrank(
            c_s1,
            patch_size=B,
            stride=stride,
            search_radius=12,
            num_similar_patches=50,
            n_jobs=-1,
        )
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "Parameter": "Patch_Size_B",
                "Value": B,
                "PSNR_dB": compute_psnr(clean_64, c_denoised),
                "SSIM": compute_ssim(clean_64, c_denoised),
                "Roughness_rho": compute_roughness_rho(c_denoised),
                "Time_sec": elapsed,
            }
        )
    return pd.DataFrame(rows)


def evaluate_num_patches_sensitivity(
    c_s1: np.ndarray,
    clean_img: np.ndarray,
    num_patches_list: List[int] = [20, 35, 50, 75, 100],
    stride: int = 4,
) -> pd.DataFrame:
    """Evaluate performance as a function of number of grouped patches L."""
    clean_64 = clean_img.astype(np.float64)

    rows = []
    for L in num_patches_list:
        t0 = time.perf_counter()
        c_denoised = denoise_stage2_patch_lowrank(
            c_s1,
            patch_size=11,
            stride=stride,
            search_radius=12,
            num_similar_patches=L,
            n_jobs=-1,
        )
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "Parameter": "Num_Patches_L",
                "Value": L,
                "PSNR_dB": compute_psnr(clean_64, c_denoised),
                "SSIM": compute_ssim(clean_64, c_denoised),
                "Roughness_rho": compute_roughness_rho(c_denoised),
                "Time_sec": elapsed,
            }
        )
    return pd.DataFrame(rows)
