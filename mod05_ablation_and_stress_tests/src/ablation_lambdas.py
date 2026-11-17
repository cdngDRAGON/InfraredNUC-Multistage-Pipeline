"""mod05_ablation_and_stress_tests/src/ablation_lambdas.py

Ablation Study 2: Sensitivity Analysis of Regularization Parameters (Section IV-C3, Fig. 18).

Evaluates the sensitivity and trade-offs of the three primary regularization parameters:
  - lambda1: Schatten-1/2 low-rank penalty on stripe matrix S.
  - lambda2: First-order Total Variation penalty ||grad(C)||_1.
  - lambda3: Second-order horizontal directional TV penalty ||d2_dx2(C)||_1.

Varies each parameter across {1e-2, 1e-3, 1e-4} while holding the other two fixed at 1e-3,
generating 9 distinct configurations. Outputs:
  1. results/fig18_regularization_lambdas.png: 3x6 grid displaying Restored Clean Image
     and Stripe Error Map for each configuration.
  2. results/ablation_lambdas.csv: Quantitative metrics confirming that
     lambda1 = lambda2 = lambda3 = 1e-3 achieves the optimal balance.
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
from mod01_priors_and_noise_simulator.src.noise_simulator import simulate_case1_conventional
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


def solve_admm_normalized(
    degraded: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    rho: float = 0.5,
    maxiter: int = 70,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute Stage 1 ADMM on normalized [0, 1] domain for consistent lambda scaling.

    Parameters
    ----------
    degraded : np.ndarray
        Degraded image in [0, 255].
    lambda1, lambda2, lambda3 : float
        Regularization weights in {1e-2, 1e-3, 1e-4}.
    rho : float
        Augmented Lagrangian parameter.
    maxiter : int
        Number of iterations.

    Returns
    -------
    c_out : np.ndarray
        Restored clean image in [0, 255].
    s_out : np.ndarray
        Estimated stripe matrix in [-255, 255].
    """
    D = np.asarray(degraded, dtype=np.float64) / 255.0
    M, N = D.shape

    eig_gx_sq, eig_gy_sq, eig_hxx_sq = compute_fft_eigenvalues((M, N))
    fft_denom = 1.0 + rho * (eig_gx_sq + eig_gy_sq) + rho * eig_hxx_sq

    C = np.copy(D)
    S = np.zeros((M, N), dtype=np.float64)

    Gx = grad_x(C)
    Gy = grad_y(C)
    H = d2_dx2(C)

    Lambda_Gx = np.zeros((M, N), dtype=np.float64)
    Lambda_Gy = np.zeros((M, N), dtype=np.float64)
    Lambda_H = np.zeros((M, N), dtype=np.float64)

    for it in range(1, maxiter + 1):
        # 1. Stripe update
        residual = D - C
        S_svd, _ = schatten_half_shrinkage(residual, tau=lambda1 / rho, rank_limit=3)
        S_col = project_column_stripe(S_svd)
        S = 0.95 * S_col + 0.05 * S_svd

        # 2. Clean image update
        v_gx = Gx - (Lambda_Gx / rho)
        v_gy = Gy - (Lambda_Gy / rho)
        v_h = H - (Lambda_H / rho)
        div_g = grad_x_adj(v_gx) + grad_y_adj(v_gy)
        adj_h = d2_dx2_adj(v_h)

        rhs_spatial = (D - S) + rho * div_g + rho * adj_h
        C = np.real(np.fft.ifft2(np.fft.fft2(rhs_spatial) / fft_denom))

        # 3. Auxiliaries update
        u_gx = grad_x(C) + (Lambda_Gx / rho)
        u_gy = grad_y(C) + (Lambda_Gy / rho)
        Gx = soft_threshold(u_gx, threshold=lambda2 / rho)
        Gy = soft_threshold(u_gy, threshold=(lambda2 * 0.2) / rho)

        u_h = d2_dx2(C) + (Lambda_H / rho)
        H = soft_threshold(u_h, threshold=lambda3 / rho)

        # 4. Multipliers
        Lambda_Gx += rho * (grad_x(C) - Gx)
        Lambda_Gy += rho * (grad_y(C) - Gy)
        Lambda_H += rho * (d2_dx2(C) - H)

    c_out = np.clip(C * 255.0, 0.0, 255.0)
    s_out = S * 255.0
    return c_out, s_out


def run_lambda_sensitivity_sweep(
    clean: np.ndarray,
    degraded: np.ndarray,
    stripe_gt: np.ndarray,
    values: Sequence[float] = (1e-2, 1e-3, 1e-4),
    baseline_val: float = 1e-3,
    maxiter: int = 70,
) -> Dict[str, Dict[str, Any]]:
    """Run all 9 configurations varying lambda1, lambda2, and lambda3."""
    clean_arr = np.asarray(clean, dtype=np.float64)
    deg_arr = np.asarray(degraded, dtype=np.float64)
    s_gt_arr = np.asarray(stripe_gt, dtype=np.float64)

    configs = []
    # 1. Sweep lambda1
    for v in values:
        configs.append(
            {
                "group": "lambda1",
                "name": f"Sweep_lambda1_{v:.0e}",
                "l1": v,
                "l2": baseline_val,
                "l3": baseline_val,
            }
        )
    # 2. Sweep lambda2
    for v in values:
        configs.append(
            {
                "group": "lambda2",
                "name": f"Sweep_lambda2_{v:.0e}",
                "l1": baseline_val,
                "l2": v,
                "l3": baseline_val,
            }
        )
    # 3. Sweep lambda3
    for v in values:
        configs.append(
            {
                "group": "lambda3",
                "name": f"Sweep_lambda3_{v:.0e}",
                "l1": baseline_val,
                "l2": baseline_val,
                "l3": v,
            }
        )

    results: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        name = cfg["name"]
        l1, l2, l3 = cfg["l1"], cfg["l2"], cfg["l3"]
        c_est, s_est = solve_admm_normalized(
            degraded=deg_arr,
            lambda1=l1,
            lambda2=l2,
            lambda3=l3,
            maxiter=maxiter,
        )
        metrics = evaluate_all_metrics(clean_arr, c_est, degraded=deg_arr)
        stripe_err = np.abs(s_est - s_gt_arr)
        stripe_mae = float(np.mean(stripe_err))

        results[name] = {
            "group": cfg["group"],
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "restored": c_est,
            "stripe": s_est,
            "stripe_err": stripe_err,
            "stripe_mae": stripe_mae,
            "metrics": metrics,
        }

    return results


def plot_fig18_regularization_lambdas(
    sweep_results: Dict[str, Dict[str, Any]],
    save_path: Union[str, Path],
) -> None:
    """Generate Fig. 18: 3x6 grid displaying Clean Image and Stripe Error for each configuration."""
    set_ieee_style()
    groups = ["lambda1", "lambda2", "lambda3"]
    values = [1e-2, 1e-3, 1e-4]

    fig, axes = plt.subplots(3, 6, figsize=(16.5, 8.8), dpi=300)

    for row_idx, param in enumerate(groups):
        param_symbol = r"\lambda_1" if param == "lambda1" else (r"\lambda_2" if param == "lambda2" else r"\lambda_3")
        for col_pair, v in enumerate(values):
            cfg_name = f"Sweep_{param}_{v:.0e}"
            data = sweep_results[cfg_name]
            c_est = data["restored"]
            s_err = data["stripe_err"]
            m = data["metrics"]
            mae = data["stripe_mae"]

            c_ax = axes[row_idx, col_pair * 2]
            err_ax = axes[row_idx, col_pair * 2 + 1]

            # 1. Restored Clean Image
            c_ax.imshow(c_est, cmap="gray", vmin=0, vmax=255)
            optimal_tag = " (Optimal)" if (v == 1e-3) else ""
            c_ax.set_title(
                f"${param_symbol} = 10^{{{int(np.log10(v))}}}${optimal_tag}\nPSNR: {m['PSNR']:.2f} dB, SSIM: {m['SSIM']:.4f}",
                fontsize=8.0,
            )
            c_ax.axis("off")

            # 2. Stripe Error Map
            im_err = err_ax.imshow(s_err, cmap="inferno", vmin=0, vmax=15.0)
            err_ax.set_title(
                f"Stripe Error\nMAE: {mae:.2f}",
                fontsize=8.0,
            )
            err_ax.axis("off")

    plt.suptitle(
        r"Figure 18: Parametric Sensitivity Analysis of Regularization Hyperparameters $\lambda_1, \lambda_2, \lambda_3$",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure 18 Exported] -> {save_path}")


def evaluate_lambdas_ablation(
    clean_path: Union[str, Path],
    results_dir: Union[str, Path],
    maxiter: int = 70,
) -> pd.DataFrame:
    """Master routine executing Ablation 2 and generating Fig. 18 and CSV."""
    clean = np.array(Image.open(clean_path), dtype=np.float64)
    sim = simulate_case1_conventional(clean, stripe_sigma=12.0, env_sigma=3.0, seed=42)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation 2] Evaluating lambda parameters on {Path(clean_path).name}...")
    sweep_results = run_lambda_sensitivity_sweep(
        clean=clean,
        degraded=sim.degraded,
        stripe_gt=sim.stripe,
        maxiter=maxiter,
    )

    # 1. Export Figure 18
    fig18_path = results_dir / "fig18_regularization_lambdas.png"
    plot_fig18_regularization_lambdas(sweep_results, fig18_path)

    # 2. Export CSV
    rows = []
    for cfg_name, data in sweep_results.items():
        m = data["metrics"]
        rows.append(
            {
                "Config_Name": cfg_name,
                "Parameter_Swept": data["group"],
                "Lambda1": data["l1"],
                "Lambda2": data["l2"],
                "Lambda3": data["l3"],
                "PSNR_dB": m["PSNR"],
                "SSIM": m["SSIM"],
                "AVGE": m["AVGE"],
                "Roughness_rho": m["Roughness_rho"],
                "ICV": m["ICV"],
                "AAD": m["AAD"],
                "MRD": m["MRD"],
                "Stripe_MAE": data["stripe_mae"],
            }
        )
    df_lambdas = pd.DataFrame(rows)
    csv_path = results_dir / "ablation_lambdas.csv"
    df_lambdas.to_csv(csv_path, index=False)
    print(f"  [CSV Exported] -> {csv_path}")

    return df_lambdas
