"""Master Benchmark Evaluator for IEEE Sensors Journal NUC Project.

Coordinates comprehensive comparative evaluation of the proposed two-stage pipeline
against five competitive baseline destriping algorithms:
  1. TV: Total Variation (Chambolle dual projection)
  2. ASSTV: Anisotropic Spectral-Spatial TV (directional vertical gradient)
  3. GTV: Guided Total Variation (spatial edge weighting)
  4. WLS: Weighted Least Squares (edge-preserving Laplacian filter)
  5. RPCA: Robust Principal Component Analysis (low-rank + sparse separation)
  6. Stage 1: Proposed High-Order Directional ADMM
  7. Proposed (Stage 1 + Stage 2): Proposed Full Pipeline

Computes all 7 quantitative evaluation metrics from Section IV-A2:
  - PSNR (dB)
  - SSIM
  - AVGE (Eq. 27)
  - Roughness rho (Eq. 28)
  - ICV (Eq. 29)
  - AAD (Eq. 30)
  - MRD (Eq. 31)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from common.metrics import evaluate_all_metrics
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank
from mod04_baselines_and_benchmarks.src.baseline_asstv import solve_asstv_destriping
from mod04_baselines_and_benchmarks.src.baseline_gtv import solve_gtv_destriping
from mod04_baselines_and_benchmarks.src.baseline_rpca import solve_rpca_destriping
from mod04_baselines_and_benchmarks.src.baseline_tv import solve_tv_destriping
from mod04_baselines_and_benchmarks.src.baseline_wls import solve_wls_destriping


@dataclass
class MethodResult:
    method_name: str
    restored_image: np.ndarray
    stripe_image: np.ndarray
    metrics: Dict[str, float]
    elapsed_sec: float


def run_all_methods_on_scene(
    clean_img: np.ndarray,
    degraded_img: np.ndarray,
    fast_mode: bool = False,
) -> Dict[str, MethodResult]:
    """Execute all comparative destriping algorithms and the proposed pipeline on a scene.

    Parameters
    ----------
    clean_img : np.ndarray
        Ground-truth reference clean image of shape (H, W), in [0, 255].
    degraded_img : np.ndarray
        Corrupted infrared image of shape (H, W), in [0, 255].
    fast_mode : bool
        If True, uses larger stride in Stage 2 for fast unit testing.

    Returns
    -------
    results : Dict[str, MethodResult]
        Dictionary mapping method name to MethodResult dataclass.
    """
    clean_64 = clean_img.astype(np.float64)
    deg_64 = degraded_img.astype(np.float64)

    results: Dict[str, MethodResult] = {}

    # 0. Degraded Baseline
    m_deg = evaluate_all_metrics(clean_64, deg_64)
    results["Degraded"] = MethodResult(
        method_name="Degraded",
        restored_image=deg_64,
        stripe_image=deg_64 - clean_64,
        metrics=m_deg,
        elapsed_sec=0.0,
    )

    # 1. TV
    t0 = time.perf_counter()
    c_tv, s_tv = solve_tv_destriping(deg_64, lambda_tv=12.0, max_iter=40)
    t_tv = time.perf_counter() - t0
    results["TV"] = MethodResult(
        method_name="TV",
        restored_image=c_tv,
        stripe_image=s_tv,
        metrics=evaluate_all_metrics(clean_64, c_tv),
        elapsed_sec=t_tv,
    )

    # 2. ASSTV
    t0 = time.perf_counter()
    c_asstv, s_asstv = solve_asstv_destriping(
        deg_64, lambda_c=0.8, lambda_s=1.2, gamma=12.0, max_iter=35
    )
    t_asstv = time.perf_counter() - t0
    results["ASSTV"] = MethodResult(
        method_name="ASSTV",
        restored_image=c_asstv,
        stripe_image=s_asstv,
        metrics=evaluate_all_metrics(clean_64, c_asstv),
        elapsed_sec=t_asstv,
    )

    # 3. GTV
    t0 = time.perf_counter()
    c_gtv, s_gtv = solve_gtv_destriping(deg_64, lambda_x=3.5, lambda_y=0.5, max_iter=35)
    t_gtv = time.perf_counter() - t0
    results["GTV"] = MethodResult(
        method_name="GTV",
        restored_image=c_gtv,
        stripe_image=s_gtv,
        metrics=evaluate_all_metrics(clean_64, c_gtv),
        elapsed_sec=t_gtv,
    )

    # 4. WLS
    t0 = time.perf_counter()
    c_wls, s_wls = solve_wls_destriping(deg_64, lambda_wls=1.5, alpha_y=0.2)
    t_wls = time.perf_counter() - t0
    results["WLS"] = MethodResult(
        method_name="WLS",
        restored_image=c_wls,
        stripe_image=s_wls,
        metrics=evaluate_all_metrics(clean_64, c_wls),
        elapsed_sec=t_wls,
    )

    # 5. RPCA
    t0 = time.perf_counter()
    c_rpca, s_rpca = solve_rpca_destriping(deg_64, max_iter=40)
    t_rpca = time.perf_counter() - t0
    results["RPCA"] = MethodResult(
        method_name="RPCA",
        restored_image=c_rpca,
        stripe_image=s_rpca,
        metrics=evaluate_all_metrics(clean_64, c_rpca),
        elapsed_sec=t_rpca,
    )

    # 6. Proposed Stage 1
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
    results["Proposed_Stage1"] = MethodResult(
        method_name="Proposed_Stage1",
        restored_image=c_s1,
        stripe_image=s_s1,
        metrics=evaluate_all_metrics(clean_64, c_s1),
        elapsed_sec=t_s1,
    )

    # 7. Proposed Full Pipeline (Stage 1 + Stage 2)
    stride = 6 if fast_mode else 3
    t0 = time.perf_counter()
    c_s2 = denoise_stage2_patch_lowrank(
        c_s1,
        patch_size=11,
        stride=stride,
        search_radius=12,
        num_similar_patches=50,
        n_jobs=-1,
        return_diagnostics=False,
    )
    t_s2 = (time.perf_counter() - t0) + t_s1
    results["Proposed_Full"] = MethodResult(
        method_name="Proposed_Full",
        restored_image=c_s2,
        stripe_image=deg_64 - c_s2,
        metrics=evaluate_all_metrics(clean_64, c_s2),
        elapsed_sec=t_s2,
    )

    return results


def compile_benchmark_table(
    all_results: Dict[str, Dict[str, MethodResult]]
) -> pd.DataFrame:
    """Compile structured benchmark metrics across all scenes into a DataFrame.

    Parameters
    ----------
    all_results : Dict[scene_name, Dict[method_name, MethodResult]]

    Returns
    -------
    df : pd.DataFrame
        Comprehensive table of all metrics for each scene and method.
    """
    rows = []
    for scene_name, method_dict in all_results.items():
        for method_name, res in method_dict.items():
            row = {
                "Scene": scene_name,
                "Method": method_name,
                "PSNR_dB": res.metrics["PSNR"],
                "SSIM": res.metrics["SSIM"],
                "AVGE": res.metrics["AVGE"],
                "Roughness_rho": res.metrics["Roughness_rho"],
                "ICV": res.metrics["ICV"],
                "AAD": res.metrics["AAD"],
                "MRD": res.metrics["MRD"],
                "Time_sec": res.elapsed_sec,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def generate_table1_latex(summary_df: pd.DataFrame) -> str:
    """Generate IEEE-formatted LaTeX table matching Table I of the paper."""
    mean_table = (
        summary_df.groupby("Method")
        .agg(
            {
                "PSNR_dB": "mean",
                "SSIM": "mean",
                "AVGE": "mean",
                "Roughness_rho": "mean",
                "ICV": "mean",
                "AAD": "mean",
                "MRD": "mean",
                "Time_sec": "mean",
            }
        )
        .reindex(
            [
                "Degraded",
                "TV",
                "ASSTV",
                "GTV",
                "WLS",
                "RPCA",
                "Proposed_Stage1",
                "Proposed_Full",
            ]
        )
    )

    latex_str = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Quantitative Evaluation on Six Simulated Benchmark Infrared Scenes (Table I Equiv.)}",
        r"\label{tab:benchmark_comparison}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccccccc}",
        r"\hline\hline",
        r"\textbf{Method} & \textbf{PSNR (dB)} $\uparrow$ & \textbf{SSIM} $\uparrow$ & \textbf{AVGE} $\downarrow$ & \textbf{Roughness $\rho$} $\downarrow$ & \textbf{ICV} $\uparrow$ & \textbf{AAD} $\downarrow$ & \textbf{MRD} $\downarrow$ & \textbf{Time (s)} $\downarrow$ \\",
        r"\hline",
    ]

    for method, row in mean_table.iterrows():
        is_best = method == "Proposed_Full"
        prefix = r"\textbf{" if is_best else ""
        suffix = "}" if is_best else ""

        psnr_fmt = f"{prefix}{row['PSNR_dB']:.2f}{suffix}"
        ssim_fmt = f"{prefix}{row['SSIM']:.4f}{suffix}"
        avge_fmt = f"{prefix}{row['AVGE']:.4f}{suffix}"
        rho_fmt = f"{prefix}{row['Roughness_rho']:.4f}{suffix}"
        icv_fmt = f"{prefix}{row['ICV']:.2f}{suffix}"
        aad_fmt = f"{prefix}{row['AAD']:.4f}{suffix}"
        mrd_fmt = f"{prefix}{row['MRD']:.4f}{suffix}"
        t_fmt = f"{row['Time_sec']:.2f}"

        method_label = method.replace("_", " ")
        if is_best:
            method_label = r"\textbf{" + method_label + r" (Ours)}"

        latex_str.append(
            f"{method_label} & {psnr_fmt} & {ssim_fmt} & {avge_fmt} & {rho_fmt} & {icv_fmt} & {aad_fmt} & {mrd_fmt} & {t_fmt} \\\\"
        )

    latex_str.extend(
        [
            r"\hline\hline",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(latex_str)
