"""Stress Testing Suite for IEEE Sensors Journal NUC Pipeline.

Evaluates algorithmic robustness under severe and non-ideal operational extremes:
  1. High Stripe Noise Variance: sigma_s in [5, 10, 15, 20, 25, 30, 40]
  2. High Environmental Gaussian Noise: sigma_n in [2, 5, 8, 12, 16, 20, 25]
  3. Defective / Dead / Saturated Columns: multi-column blindness simulation
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common.metrics import compute_psnr, compute_roughness_rho, compute_ssim
from mod01_priors_and_noise_simulator.src.noise_simulator import simulate_case1_conventional
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank


def stress_test_stripe_amplitudes(
    clean_img: np.ndarray,
    sigmas: List[float] = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    env_sigma: float = 5.0,
    stride: int = 4,
) -> pd.DataFrame:
    """Stress test pipeline against increasing stripe standard deviations."""
    clean_64 = clean_img.astype(np.float64)
    rows = []

    for s_sigma in sigmas:
        sim = simulate_case1_conventional(
            clean_64, stripe_sigma=s_sigma, env_sigma=env_sigma, seed=42
        )
        deg = sim.degraded
        t0 = time.perf_counter()
        c_s1, _, _ = solve_stage1_admm(deg, lambda1=0.5, lambda2=0.05, lambda3=2.0, maxiter=35)
        c_full = denoise_stage2_patch_lowrank(
            c_s1, patch_size=11, stride=stride, search_radius=12, num_similar_patches=50, n_jobs=-1
        )
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "Stress_Type": "Stripe_Sigma",
                "Stress_Level": s_sigma,
                "Degraded_PSNR": compute_psnr(clean_64, deg),
                "Restored_PSNR": compute_psnr(clean_64, c_full),
                "PSNR_Gain_dB": compute_psnr(clean_64, c_full) - compute_psnr(clean_64, deg),
                "Restored_SSIM": compute_ssim(clean_64, c_full),
                "Restored_Rho": compute_roughness_rho(c_full),
                "Time_sec": elapsed,
            }
        )
    return pd.DataFrame(rows)


def stress_test_environmental_noise(
    clean_img: np.ndarray,
    env_sigmas: List[float] = [2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0],
    stripe_sigma: float = 12.0,
    stride: int = 4,
) -> pd.DataFrame:
    """Stress test pipeline against increasing environmental noise."""
    clean_64 = clean_img.astype(np.float64)
    rows = []

    for e_sigma in env_sigmas:
        sim = simulate_case1_conventional(
            clean_64, stripe_sigma=stripe_sigma, env_sigma=e_sigma, seed=42
        )
        deg = sim.degraded
        t0 = time.perf_counter()
        c_s1, _, _ = solve_stage1_admm(deg, lambda1=0.5, lambda2=0.05, lambda3=2.0, maxiter=35)
        c_full = denoise_stage2_patch_lowrank(
            c_s1, patch_size=11, stride=stride, search_radius=12, num_similar_patches=50, n_jobs=-1
        )
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "Stress_Type": "Env_Noise_Sigma",
                "Stress_Level": e_sigma,
                "Degraded_PSNR": compute_psnr(clean_64, deg),
                "Restored_PSNR": compute_psnr(clean_64, c_full),
                "PSNR_Gain_dB": compute_psnr(clean_64, c_full) - compute_psnr(clean_64, deg),
                "Restored_SSIM": compute_ssim(clean_64, c_full),
                "Restored_Rho": compute_roughness_rho(c_full),
                "Time_sec": elapsed,
            }
        )
    return pd.DataFrame(rows)


def stress_test_defective_columns(
    clean_img: np.ndarray,
    defect_ratios: List[float] = [0.01, 0.03, 0.05, 0.08, 0.10],
    stride: int = 4,
) -> pd.DataFrame:
    """Stress test pipeline when detector columns are completely dead or saturated."""
    clean_64 = clean_img.astype(np.float64)
    H, W = clean_64.shape
    rows = []

    for ratio in defect_ratios:
        # Base degraded image
        sim = simulate_case1_conventional(clean_64, stripe_sigma=12.0, env_sigma=5.0, seed=42)
        deg = np.copy(sim.degraded)

        # Inject dead and saturated columns
        np.random.seed(int(ratio * 1000))
        num_defects = int(W * ratio)
        defect_cols = np.random.choice(W, size=num_defects, replace=False)

        half = num_defects // 2
        deg[:, defect_cols[:half]] = 0.0  # Dead columns
        deg[:, defect_cols[half:]] = 255.0  # Saturated columns

        t0 = time.perf_counter()
        c_s1, _, _ = solve_stage1_admm(deg, lambda1=0.5, lambda2=0.05, lambda3=2.0, maxiter=40)
        c_full = denoise_stage2_patch_lowrank(
            c_s1, patch_size=11, stride=stride, search_radius=12, num_similar_patches=50, n_jobs=-1
        )
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "Stress_Type": "Defective_Column_Ratio",
                "Stress_Level": ratio,
                "Degraded_PSNR": compute_psnr(clean_64, deg),
                "Restored_PSNR": compute_psnr(clean_64, c_full),
                "PSNR_Gain_dB": compute_psnr(clean_64, c_full) - compute_psnr(clean_64, deg),
                "Restored_SSIM": compute_ssim(clean_64, c_full),
                "Restored_Rho": compute_roughness_rho(c_full),
                "Time_sec": elapsed,
            }
        )
    return pd.DataFrame(rows)
