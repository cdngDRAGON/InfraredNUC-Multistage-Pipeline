"""mod05_ablation_and_stress_tests/src/ablation_noise_stress.py

Ablation Study 3: Nonuniform Noise Intensity Stress Tests (Section IV-C3, Figs. 19-20).

Stress tests the proposed pipeline under severe operational extremes:
  1. Stripe Intensity Stress (Fig. 19):
     Keeps environmental noise constant (sigma_n = 5.0) and increases vertical
     stripe intensity across 5 levels (Case 1 to Case 5: sigma_s in {8, 16, 24, 32, 40}).
     Displays 4 rows: Degraded Image, Stripes GT, Restore Image, Estimated Stripes.
  2. Environmental Noise Stress (Fig. 20):
     Keeps stripe intensity constant (sigma_s = 15.0) and escalates ambient
     thermal noise across 4 levels (Case 1 to Case 4: sigma_n in {4, 8, 14, 20}).
     Displays 2 rows: Degraded Image vs. Restore Image.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common.metrics import evaluate_all_metrics
from common.plot_style import set_ieee_style
from mod01_priors_and_noise_simulator.src.noise_simulator import (
    generate_column_stripe,
    simulate_case1_conventional,
)
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank


def run_pipeline(
    degraded: np.ndarray,
    stride: int = 4,
    maxiter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute complete two-stage NUC pipeline (Stage 1 ADMM + Stage 2 SVD patch denoiser).

    Parameters
    ----------
    degraded : np.ndarray
        Degraded input image in [0, 255].
    stride : int
        Patch stride for Stage 2 patch grouping (stride=4 provides excellent speed & quality).
    maxiter : int
        ADMM iterations for Stage 1.

    Returns
    -------
    c_final : np.ndarray
        Restored clean image in [0, 255].
    s_est : np.ndarray
        Estimated vertical stripe noise matrix.
    """
    # Stage 1: ADMM Destriping
    c_s1, s_est, _ = solve_stage1_admm(
        degraded,
        lambda1=0.5,
        lambda2=0.05,
        lambda3=2.0,
        maxiter=maxiter,
        tol=1e-4,
    )

    # Stage 2: SVD Low-Rank Patch Approximation
    c_final = denoise_stage2_patch_lowrank(
        c_s1,
        patch_size=11,
        stride=stride,
        search_radius=15,
        num_similar_patches=50,
        gamma=0.95,
        n_jobs=-1,
    )
    return c_final, s_est


def run_stripe_intensity_stress(
    clean: np.ndarray,
    stripe_sigmas: Sequence[float] = (8.0, 16.0, 24.0, 32.0, 40.0),
    env_sigma: float = 5.0,
    stride: int = 4,
) -> List[Dict[str, Any]]:
    """Evaluate pipeline under 5 escalating levels of vertical stripe noise."""
    clean_arr = np.asarray(clean, dtype=np.float64)
    H, W = clean_arr.shape[:2]
    cases = []

    for idx, sigma_s in enumerate(stripe_sigmas, start=1):
        rng = np.random.RandomState(42 + idx)
        s_col = rng.normal(0.0, sigma_s, W)
        stripe_gt = generate_column_stripe((H, W), s_col, vertical_drift_ratio=0.015, rng=rng)
        noise_env = rng.normal(0.0, env_sigma, (H, W))
        degraded = np.clip(clean_arr + stripe_gt + noise_env, 0.0, 255.0)

        c_restored, s_estimated = run_pipeline(degraded, stride=stride)
        m = evaluate_all_metrics(clean_arr, c_restored, degraded=degraded)

        cases.append(
            {
                "case_name": f"Case {idx}",
                "sigma_s": sigma_s,
                "sigma_n": env_sigma,
                "degraded": degraded,
                "stripe_gt": stripe_gt,
                "restored": c_restored,
                "stripe_est": s_estimated,
                "metrics": m,
            }
        )
    return cases


def run_env_noise_stress(
    clean: np.ndarray,
    env_sigmas: Sequence[float] = (4.0, 8.0, 14.0, 20.0),
    stripe_sigma: float = 15.0,
    stride: int = 4,
) -> List[Dict[str, Any]]:
    """Evaluate pipeline under 4 escalating levels of environmental ambient noise."""
    clean_arr = np.asarray(clean, dtype=np.float64)
    H, W = clean_arr.shape[:2]
    cases = []

    for idx, sigma_n in enumerate(env_sigmas, start=1):
        rng = np.random.RandomState(100 + idx)
        s_col = rng.normal(0.0, stripe_sigma, W)
        stripe_gt = generate_column_stripe((H, W), s_col, vertical_drift_ratio=0.015, rng=rng)
        noise_env = rng.normal(0.0, sigma_n, (H, W))
        degraded = np.clip(clean_arr + stripe_gt + noise_env, 0.0, 255.0)

        c_restored, s_estimated = run_pipeline(degraded, stride=stride)
        m = evaluate_all_metrics(clean_arr, c_restored, degraded=degraded)

        cases.append(
            {
                "case_name": f"Case {idx}",
                "sigma_s": stripe_sigma,
                "sigma_n": sigma_n,
                "degraded": degraded,
                "restored": c_restored,
                "metrics": m,
            }
        )
    return cases


def plot_fig19_stripe_intensity_stress(
    cases: List[Dict[str, Any]],
    save_path: Union[str, Path],
) -> None:
    """Generate Fig. 19: 4 rows x 5 columns displaying Degraded, GT Stripes, Restored, Estimated Stripes."""
    set_ieee_style()
    n_cases = len(cases)
    fig, axes = plt.subplots(4, n_cases, figsize=(14.5, 9.5), dpi=300)

    # Determine stripe dynamic range for visual consistency
    max_stripe_val = max(float(np.max(np.abs(c["stripe_gt"]))) for c in cases)

    for col, c in enumerate(cases):
        sigma_s = c["sigma_s"]
        m = c["metrics"]

        # Row 0: Degraded Image
        axes[0, col].imshow(c["degraded"], cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(
            f"{c['case_name']} ($\\sigma_s={sigma_s:.0f}$)\nDeg. PSNR: {m['Degraded_PSNR']:.2f} dB",
            fontsize=8.5,
            fontweight="bold",
        )
        axes[0, col].axis("off")

        # Row 1: Ground Truth Stripes
        axes[1, col].imshow(c["stripe_gt"], cmap="gray", vmin=-max_stripe_val, vmax=max_stripe_val)
        axes[1, col].set_title("Stripes GT", fontsize=8.0)
        axes[1, col].axis("off")

        # Row 2: Restored Clean Image
        axes[2, col].imshow(c["restored"], cmap="gray", vmin=0, vmax=255)
        axes[2, col].set_title(
            f"Restore Image\nPSNR: {m['PSNR']:.2f} dB, SSIM: {m['SSIM']:.4f}",
            fontsize=8.0,
            fontweight="bold",
        )
        axes[2, col].axis("off")

        # Row 3: Estimated Stripes
        axes[3, col].imshow(c["stripe_est"], cmap="gray", vmin=-max_stripe_val, vmax=max_stripe_val)
        axes[3, col].set_title("Estimated Stripes", fontsize=8.0)
        axes[3, col].axis("off")

    # Add row labels on the leftmost side
    row_labels = ["Degraded Image", "Stripes GT", "Restore Image", "Estimated Stripes"]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=9.0,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=90,
        )

    plt.suptitle(
        r"Figure 19: Pipeline Stress Testing Across Increasing Stripe Intensities ($\sigma_s \in [8, 40]$, $\sigma_n = 5$)",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure 19 Exported] -> {save_path}")


def plot_fig20_env_noise_intensity_stress(
    cases: List[Dict[str, Any]],
    save_path: Union[str, Path],
) -> None:
    """Generate Fig. 20: 2 rows x 4 columns displaying Degraded Image vs. Restore Image."""
    set_ieee_style()
    n_cases = len(cases)
    fig, axes = plt.subplots(2, n_cases, figsize=(12.5, 5.5), dpi=300)

    for col, c in enumerate(cases):
        sigma_n = c["sigma_n"]
        m = c["metrics"]

        # Row 0: Degraded Image
        axes[0, col].imshow(c["degraded"], cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(
            f"{c['case_name']} ($\\sigma_n={sigma_n:.0f}$)\nDeg. PSNR: {m['Degraded_PSNR']:.2f} dB",
            fontsize=8.5,
            fontweight="bold",
        )
        axes[0, col].axis("off")

        # Row 1: Restore Image
        axes[1, col].imshow(c["restored"], cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_title(
            f"Restore Image\nPSNR: {m['PSNR']:.2f} dB, SSIM: {m['SSIM']:.4f}",
            fontsize=8.5,
            fontweight="bold",
        )
        axes[1, col].axis("off")

    row_labels = ["Degraded Image", "Restore Image"]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=9.0,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=90,
        )

    plt.suptitle(
        r"Figure 20: Robustness Under Escalating Ambient Thermal Noise ($\sigma_n \in [4, 20]$, $\sigma_s = 15$)",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure 20 Exported] -> {save_path}")


def evaluate_noise_stress_ablation(
    clean_path: Union[str, Path],
    results_dir: Union[str, Path],
    stride: int = 4,
) -> pd.DataFrame:
    """Master routine executing Ablation 3 and generating Figs. 19-20 and CSV."""
    clean = np.array(Image.open(clean_path), dtype=np.float64)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation 3] Evaluating noise stress tests on {Path(clean_path).name}...")

    # 1. Stripe intensity stress test (Fig. 19)
    stripe_cases = run_stripe_intensity_stress(clean, stride=stride)
    fig19_path = results_dir / "fig19_stripe_intensity_stress.png"
    plot_fig19_stripe_intensity_stress(stripe_cases, fig19_path)

    # 2. Environmental noise stress test (Fig. 20)
    env_cases = run_env_noise_stress(clean, stride=stride)
    fig20_path = results_dir / "fig20_env_noise_intensity_stress.png"
    plot_fig20_env_noise_intensity_stress(env_cases, fig20_path)

    # 3. Export combined CSV
    rows = []
    for c in stripe_cases:
        m = c["metrics"]
        rows.append(
            {
                "Stress_Type": "Stripe_Intensity",
                "Case_Name": c["case_name"],
                "Sigma_Stripe": c["sigma_s"],
                "Sigma_Env": c["sigma_n"],
                "Degraded_PSNR": m["Degraded_PSNR"],
                "Degraded_SSIM": m["Degraded_SSIM"],
                "Restored_PSNR": m["PSNR"],
                "Restored_SSIM": m["SSIM"],
                "PSNR_Gain_dB": m["PSNR"] - m["Degraded_PSNR"],
                "SSIM_Gain": m["SSIM"] - m["Degraded_SSIM"],
                "AVGE": m["AVGE"],
                "Roughness_rho": m["Roughness_rho"],
                "ICV": m["ICV"],
                "AAD": m["AAD"],
                "MRD": m["MRD"],
            }
        )
    for c in env_cases:
        m = c["metrics"]
        rows.append(
            {
                "Stress_Type": "Environmental_Noise",
                "Case_Name": c["case_name"],
                "Sigma_Stripe": c["sigma_s"],
                "Sigma_Env": c["sigma_n"],
                "Degraded_PSNR": m["Degraded_PSNR"],
                "Degraded_SSIM": m["Degraded_SSIM"],
                "Restored_PSNR": m["PSNR"],
                "Restored_SSIM": m["SSIM"],
                "PSNR_Gain_dB": m["PSNR"] - m["Degraded_PSNR"],
                "SSIM_Gain": m["SSIM"] - m["Degraded_SSIM"],
                "AVGE": m["AVGE"],
                "Roughness_rho": m["Roughness_rho"],
                "ICV": m["ICV"],
                "AAD": m["AAD"],
                "MRD": m["MRD"],
            }
        )
    df_stress = pd.DataFrame(rows)
    csv_path = results_dir / "ablation_noise_stress.csv"
    df_stress.to_csv(csv_path, index=False)
    print(f"  [CSV Exported] -> {csv_path}")

    return df_stress
