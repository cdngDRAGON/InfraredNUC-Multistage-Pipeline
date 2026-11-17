"""mod01_priors_and_noise_simulator/src/lowrank_svd_analysis.py

SVD Low-Rank Prior Verification & Eckart-Young-Mirsky Theorem (Eqs. 1-4).

Mathematical Formulation:
- Eq. (1): Singular Value Decomposition:
    J = U * Sigma * V^T = sum_{i=1}^r sigma_i * u_i * v_i^T
- Eq. (2): Best rank-k approximation (Eckart-Young-Mirsky Theorem):
    J_hat_k = argmin_{rank(X) <= k} ||J - X||_F = U_k * Sigma_k * V_k^T
- Eq. (3): Optimal Frobenius norm error:
    ||J - J_hat_k||_F = sqrt(sum_{i=k+1}^r sigma_i^2)
- Eq. (4): Relative Frobenius energy captured by top-k singular values:
    eta_k = sum_{i=1}^k sigma_i^2 / sum_{i=1}^r sigma_i^2

Reproduces:
- Fig. 4: SVD singular value distribution and rapid L-curve decay across benchmark scenes.
- Fig. 9: 3D global energy surface comparison for Clean vs. Simulation Cases 1, 2, and 3.
- CSV table: results/svd_energy_distribution.csv recording Frobenius energy fractions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.plot_style import plot_2d_with_zoom_and_3d_energy, set_ieee_style


def compute_svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full SVD: J = U * diag(s) * Vt (Eq. 1).

    Parameters
    ----------
    matrix : np.ndarray
        2D matrix of shape (M, N).

    Returns
    -------
    U : np.ndarray
        Orthogonal left singular vectors (M, M).
    s : np.ndarray
        Singular values in non-increasing order (min(M, N),).
    Vt : np.ndarray
        Orthogonal right singular vectors transposed (N, N).
    """
    arr = np.asarray(matrix, dtype=np.float64)
    U, s, Vt = np.linalg.svd(arr, full_matrices=False)
    return U, s, Vt


def compute_rank_k_reconstruction(
    matrix: np.ndarray, k: int
) -> Tuple[np.ndarray, float, float]:
    """Compute optimal rank-k Eckart-Young-Mirsky approximation (Eq. 2-4).

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix J.
    k : int
        Target approximation rank (1 <= k <= min(M, N)).

    Returns
    -------
    J_hat_k : np.ndarray
        Optimal rank-k reconstruction.
    frobenius_error : float
        ||J - J_hat_k||_F (Eq. 3).
    energy_ratio : float
        eta_k = sum_{i=1}^k sigma_i^2 / sum_{i=1}^r sigma_i^2 (Eq. 4).
    """
    U, s, Vt = compute_svd(matrix)
    k = max(1, min(k, len(s)))

    # Eq. (2)
    J_hat_k = (U[:, :k] * s[:k]) @ Vt[:k, :]

    # Eq. (3)
    error_theoretical = float(np.sqrt(np.sum(s[k:] ** 2)))

    # Eq. (4)
    total_energy = float(np.sum(s**2))
    energy_k = float(np.sum(s[:k] ** 2))
    eta_k = energy_k / (total_energy + 1e-12)

    return J_hat_k, error_theoretical, eta_k


def analyze_singular_value_decay(
    matrix: np.ndarray, top_k_list: Sequence[int] = (1, 2, 5, 10, 20, 50)
) -> Dict[str, Union[float, Dict[int, float]]]:
    """Compute SVD spectrum telemetry and cumulative energy capture for top-k ranks."""
    _, s, _ = compute_svd(matrix)
    total_energy = float(np.sum(s**2))
    cum_energy = np.cumsum(s**2)

    k_ratios: Dict[int, float] = {}
    for k in top_k_list:
        k_clamped = min(k, len(s))
        ratio = float(cum_energy[k_clamped - 1] / (total_energy + 1e-12))
        k_ratios[k] = ratio

    return {
        "singular_values": s,
        "total_frobenius_norm": float(np.sqrt(total_energy)),
        "energy_fractions": k_ratios,
        "rank_estimate": int(np.sum(s > 1e-6 * s[0])),
    }


def generate_svd_energy_table(
    scenes_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    k_values: Sequence[int] = (1, 2, 5, 10, 20),
    save_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate comprehensive SVD Frobenius energy capture table.

    Parameters
    ----------
    scenes_dict : Dict[str, Tuple[C, D, S]]
        Dictionary mapping scene name to (clean, degraded, stripe).
    k_values : Sequence[int]
        Target rank values k in [1, 2, 5, 10, 20].
    save_csv_path : Optional[str]
        Path to save CSV output.

    Returns
    -------
    df : pd.DataFrame
        Energy capture table.
    """
    rows = []

    for scene_name, (clean, degraded, stripe) in scenes_dict.items():
        components = [
            ("Clean Scene (C)", clean),
            ("Degraded Scene (D)", degraded),
            ("Stripe Component (S)", stripe),
        ]
        for comp_name, mat in components:
            stats = analyze_singular_value_decay(mat, top_k_list=k_values)
            fractions = stats["energy_fractions"]

            row = {
                "Scene": scene_name,
                "Component": comp_name,
                "Frobenius_Norm": stats["total_frobenius_norm"],
            }
            for k in k_values:
                pct = fractions[k] * 100.0
                row[f"Energy_Top_{k}(%)"] = round(pct, 3)

            rows.append(row)

    df = pd.DataFrame(rows)
    if save_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv_path)), exist_ok=True)
        df.to_csv(save_csv_path, index=False)

    return df


def plot_fig4_svd_singular_value_distribution(
    scenes_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
    max_singular_values: int = 50,
) -> plt.Figure:
    """Reproduce Fig. 4: SVD singular value distribution across benchmark scenes.

    Displays a 2x2 grid of publication subplots (one per benchmark scene)
    plotting the singular value spectrum (normalized sigma_i / sigma_1)
    on semilog-y scale for:
    - Clean Image C
    - Degraded Image D
    - Stripe Noise Matrix S

    Demonstrates the extreme low-rank property of S (rank <= 3) and high
    energy concentration of thermal scene content.
    """
    set_ieee_style()
    scene_items = list(scenes_data.items())[:4]
    if len(scene_items) < 4:
        raise ValueError("scenes_data must contain at least 4 scenes for Fig. 4")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), facecolor="#FFFFFF")
    axes_flat = axes.flatten()

    colors = {
        "Clean": "#000000",       # Clean: Solid black
        "Degraded": "#D32F2F",    # Degraded: Crimson
        "Stripe": "#1976D2",      # Stripe: Deep blue
    }

    sub_labels = ["(a)", "(b)", "(c)", "(d)"]

    for idx, (scene_name, (clean, deg, stripe)) in enumerate(scene_items):
        ax = axes_flat[idx]

        _, s_clean, _ = compute_svd(clean)
        _, s_deg, _ = compute_svd(deg)
        _, s_stripe, _ = compute_svd(stripe)

        n_pts = min(max_singular_values, len(s_clean))
        x_idx = np.arange(1, n_pts + 1)

        # Plot normalized singular values on semilog-y scale
        ax.semilogy(
            x_idx,
            s_clean[:n_pts] / (s_clean[0] + 1e-12),
            color=colors["Clean"],
            linestyle="-",
            linewidth=1.2,
            label="Clean Scene ($C$)",
        )
        ax.semilogy(
            x_idx,
            s_deg[:n_pts] / (s_deg[0] + 1e-12),
            color=colors["Degraded"],
            linestyle="--",
            linewidth=1.1,
            label="Degraded ($D$)",
        )
        ax.semilogy(
            x_idx,
            s_stripe[:n_pts] / (s_stripe[0] + 1e-12),
            color=colors["Stripe"],
            linestyle="-.",
            linewidth=1.3,
            label="Stripe Matrix ($S$)",
        )

        display_name = scene_name.replace(".png", "").replace("Img", "Image ")
        ax.set_title(f"{sub_labels[idx]} {display_name}", fontsize=8.0, fontweight="bold", pad=4)
        ax.set_xlabel("Singular Value Index $i$", fontsize=7.5)
        ax.set_ylabel(r"Normalized $\sigma_i / \sigma_1$", fontsize=7.5)
        ax.set_xlim(1, n_pts)
        ax.set_ylim(1e-5, 1.5)
        ax.grid(True, linestyle=":", color="#CCCCCC", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9, edgecolor="#000000", fontsize=7.0)

        for spine in ax.spines.values():
            spine.set_edgecolor("#000000")
            spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor="#FFFFFF", edgecolor="#FFFFFF", bbox_inches="tight")

    return fig


def plot_fig9_simulation_cases_3d_energy(
    clean: np.ndarray,
    case1_deg: np.ndarray,
    case2_deg: np.ndarray,
    case3_deg: np.ndarray,
    roi_box: Tuple[int, int, int, int] = (64, 64, 48, 48),
    save_path: Optional[str] = None,
    title: str = "Figure 9: Comparison of 2D Infrared Imagery and 3D Energy Surfaces Across Degradation Cases",
) -> Tuple[plt.Figure, np.ndarray]:
    """Reproduce Fig. 9: 2D zoomed and 3D global energy comparison across simulation cases.

    Columns:
    1. Clean Reference
    2. Case 1: Conventional FPN + Env Noise
    3. Case 2: Periodic Stripe + Env Noise
    4. Case 3: Enhanced Severe Stripe + Strong Env Noise

    Top Row: 2D grayscale image with red ROI box and magnified corner inset.
    Bottom Row: 3D surface energy distribution Z = Image(X, Y).
    """
    images_dict = {
        "Clean Reference": clean,
        "Case 1: Conventional": case1_deg,
        "Case 2: Periodic": case2_deg,
        "Case 3: Enhanced": case3_deg,
    }

    fig, axes = plot_2d_with_zoom_and_3d_energy(
        images_dict=images_dict,
        roi_box=roi_box,
        save_path=save_path,
        title=title,
        figsize_per_col=(2.8, 5.2),
        zoom_scale=2.0,
    )
    return fig, axes
