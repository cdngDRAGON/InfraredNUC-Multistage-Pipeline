"""mod01_priors_and_noise_simulator/src/gradient_priors.py

Directional Gradient Energy Distribution & High-Order Derivative Analysis.

Reproduces:
- Fig. 2: Horizontal gradient energy distribution (severe stripe oscillations).
- Fig. 3: Vertical gradient energy distribution (preservation of true structural gradient).
- Fig. 5: 1st-order (x, y) and 2nd-order (xx, xy, yx, yy) directional derivative maps.

Demonstrates mathematical anisotropy:
    ||grad_x(S)||_F^2 >> ||grad_y(S)||_F^2
and validates the hybrid high-order directional total variation (HODTV) prior foundation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.plot_style import set_ieee_style


def compute_first_order_gradients(
    img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1st-order horizontal (grad_x) and vertical (grad_y) finite difference gradients.

    Forward differences with replicate boundary padding to preserve exact (H, W) shape.

    Parameters
    ----------
    img : np.ndarray
        2D float64 image array (H, W).

    Returns
    -------
    grad_x : np.ndarray
        Horizontal gradient df/dx (differences along columns, axis=1).
    grad_y : np.ndarray
        Vertical gradient df/dy (differences along rows, axis=0).
    """
    arr = np.asarray(img, dtype=np.float64)
    H, W = arr.shape

    # grad_x: diff along columns
    grad_x = np.zeros_like(arr)
    grad_x[:, :-1] = np.diff(arr, axis=1)
    grad_x[:, -1] = grad_x[:, -2]

    # grad_y: diff along rows
    grad_y = np.zeros_like(arr)
    grad_y[:-1, :] = np.diff(arr, axis=0)
    grad_y[-1, :] = grad_y[-2, :]

    return grad_x, grad_y


def compute_second_order_derivatives(
    img: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute 2nd-order directional derivatives: xx, xy, yx, yy.

    d2/dx2: Second difference along horizontal axis.
    d2/dy2: Second difference along vertical axis.
    d2/dxdy: Mixed derivative (horizontal then vertical).
    d2/dydx: Mixed derivative (vertical then horizontal).

    Parameters
    ----------
    img : np.ndarray
        2D float64 image array (H, W).

    Returns
    -------
    derivatives : Dict[str, np.ndarray]
        Dictionary with keys 'xx', 'yy', 'xy', 'yx'.
    """
    arr = np.asarray(img, dtype=np.float64)
    H, W = arr.shape

    # First derivatives
    gx, gy = compute_first_order_gradients(arr)

    # Second pure derivatives
    # d2/dx2: central-like second difference [1, -2, 1] along x
    d2_xx = np.zeros_like(arr)
    d2_xx[:, 1:-1] = arr[:, 2:] - 2.0 * arr[:, 1:-1] + arr[:, :-2]
    d2_xx[:, 0] = d2_xx[:, 1]
    d2_xx[:, -1] = d2_xx[:, -2]

    # d2/dy2: central-like second difference [1, -2, 1] along y
    d2_yy = np.zeros_like(arr)
    d2_yy[1:-1, :] = arr[2:, :] - 2.0 * arr[1:-1, :] + arr[:-2, :]
    d2_yy[0, :] = d2_yy[1, :]
    d2_yy[-1, :] = d2_yy[-2, :]

    # Mixed derivatives
    d2_xy, _ = compute_first_order_gradients(gy)  # d(gy)/dx
    _, d2_yx = compute_first_order_gradients(gx)  # d(gx)/dy

    return {
        "xx": d2_xx,
        "yy": d2_yy,
        "xy": d2_xy,
        "yx": d2_yx,
        "gx": gx,
        "gy": gy,
    }


def compute_gradient_anisotropy_ratio(
    stripe_matrix: np.ndarray,
) -> Dict[str, float]:
    """Quantify the directional gradient energy ratio of the stripe noise.

    Returns:
    - Energy_grad_x: ||grad_x(S)||_F^2
    - Energy_grad_y: ||grad_y(S)||_F^2
    - Anisotropy_Ratio: Energy_grad_x / (Energy_grad_y + 1e-8)
    """
    gx, gy = compute_first_order_gradients(stripe_matrix)
    e_x = float(np.sum(gx**2))
    e_y = float(np.sum(gy**2))
    ratio = e_x / (e_y + 1e-8)
    return {
        "Energy_grad_x": e_x,
        "Energy_grad_y": e_y,
        "Anisotropy_Ratio": ratio,
    }


def plot_fig2_horizontal_gradient_energy(
    clean: np.ndarray,
    degraded: np.ndarray,
    save_path: Optional[str] = None,
    positions: Optional[List[float]] = None,
) -> plt.Figure:
    """Reproduce Fig. 2: Horizontal gradient energy distribution.

    Plots horizontal gradient scanlines (dC/dx vs dD/dx) at three positions:
    - Position 1: top 25% height (y = 0.25 * H)
    - Position 2: middle 50% height (y = 0.50 * H)
    - Position 3: bottom 75% height (y = 0.75 * H)

    Shows the severe high-frequency fluctuations induced by vertical stripes.
    """
    set_ieee_style()
    H, W = clean.shape
    if positions is None:
        positions = [0.25, 0.50, 0.75]

    row_indices = [int(p * H) for p in positions]
    clean_gx, _ = compute_first_order_gradients(clean)
    deg_gx, _ = compute_first_order_gradients(degraded)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.0), sharex=True, facecolor="#FFFFFF")

    colors = {
        "clean": "#000000",       # Clean black line
        "degraded": "#D32F2F",    # Highlighting severe stripe fluctuation
    }

    labels = ["(a) Position 1 (Top 25% Row)", "(b) Position 2 (Middle 50% Row)", "(c) Position 3 (Bottom 75% Row)"]

    x_vals = np.arange(W)
    for idx, (r, ax, lbl) in enumerate(zip(row_indices, axes, labels)):
        ax.plot(x_vals, deg_gx[r, :], color=colors["degraded"], linewidth=0.9, alpha=0.85, label=r"Degraded $\nabla_x D$")
        ax.plot(x_vals, clean_gx[r, :], color=colors["clean"], linewidth=1.1, linestyle="--", label=r"Clean $\nabla_x C$")

        ax.set_title(f"{lbl} - Row {r}", fontsize=8.5, fontweight="bold", pad=4)
        ax.set_ylabel(r"Gradient $\nabla_x$", fontsize=8.0)
        ax.grid(True, linestyle=":", color="#CCCCCC", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9, edgecolor="#000000", fontsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#000000")
            spine.set_linewidth(0.8)

    axes[-1].set_xlabel("Horizontal Pixel Coordinate (Column Index $x$)", fontsize=8.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor="#FFFFFF", edgecolor="#FFFFFF", bbox_inches="tight")

    return fig


def plot_fig3_vertical_gradient_energy(
    clean: np.ndarray,
    degraded: np.ndarray,
    save_path: Optional[str] = None,
    positions: Optional[List[float]] = None,
) -> plt.Figure:
    """Reproduce Fig. 3: Vertical gradient energy distribution.

    Plots vertical gradient scanlines (dC/dy vs dD/dy) at three positions:
    - Position 1: top 25% height
    - Position 2: middle 50% height
    - Position 3: bottom 75% height

    Demonstrates that the vertical gradient of the degraded image closely tracks
    the clean reference image, preserving genuine vertical structural contours.
    """
    set_ieee_style()
    H, W = clean.shape
    if positions is None:
        positions = [0.25, 0.50, 0.75]

    row_indices = [int(p * H) for p in positions]
    _, clean_gy = compute_first_order_gradients(clean)
    _, deg_gy = compute_first_order_gradients(degraded)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.0), sharex=True, facecolor="#FFFFFF")

    colors = {
        "clean": "#000000",       # Clean reference
        "degraded": "#1976D2",    # Vertical gradient tracking
    }

    labels = ["(a) Position 1 (Top 25% Row)", "(b) Position 2 (Middle 50% Row)", "(c) Position 3 (Bottom 75% Row)"]

    x_vals = np.arange(W)
    for idx, (r, ax, lbl) in enumerate(zip(row_indices, axes, labels)):
        ax.plot(x_vals, deg_gy[r, :], color=colors["degraded"], linewidth=0.9, alpha=0.85, label=r"Degraded $\nabla_y D$")
        ax.plot(x_vals, clean_gy[r, :], color=colors["clean"], linewidth=1.1, linestyle="--", label=r"Clean $\nabla_y C$")

        ax.set_title(f"{lbl} - Row {r}", fontsize=8.5, fontweight="bold", pad=4)
        ax.set_ylabel(r"Gradient $\nabla_y$", fontsize=8.0)
        ax.grid(True, linestyle=":", color="#CCCCCC", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9, edgecolor="#000000", fontsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#000000")
            spine.set_linewidth(0.8)

    axes[-1].set_xlabel("Horizontal Pixel Coordinate (Column Index $x$)", fontsize=8.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor="#FFFFFF", edgecolor="#FFFFFF", bbox_inches="tight")

    return fig


def plot_fig5_first_and_second_order_gradients(
    clean: np.ndarray,
    degraded: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduce Fig. 5: 1st- and 2nd-order directional derivative maps.

    Displays:
    - Top row (Clean Scene):
      1. First derivative gx (df/dx)
      2. First derivative gy (df/dy)
      3. Second derivative xx (d2f/dx2)
      4. Second derivative yy (d2f/dy2)
    - Bottom row (Degraded Scene):
      1. First derivative gx (df/dx) - heavily striped
      2. First derivative gy (df/dy) - preserved
      3. Second derivative xx (d2f/dx2) - catastrophic stripe distortion
      4. Second derivative yy (d2f/dy2) - pristine high-order structural details

    Proves that high-order vertical derivative d2f/dy2 provides an uncorrupted
    regularization channel for destriping without staircasing.
    """
    set_ieee_style()
    clean_derivs = compute_second_order_derivatives(clean)
    deg_derivs = compute_second_order_derivatives(degraded)

    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.8), facecolor="#FFFFFF")

    cols_clean = [
        (clean_derivs["gx"], r"Clean $\nabla_x C$"),
        (clean_derivs["gy"], r"Clean $\nabla_y C$"),
        (clean_derivs["xx"], r"Clean $\partial^2 C / \partial x^2$"),
        (clean_derivs["yy"], r"Clean $\partial^2 C / \partial y^2$"),
    ]

    cols_deg = [
        (deg_derivs["gx"], r"Degraded $\nabla_x D$"),
        (deg_derivs["gy"], r"Degraded $\nabla_y D$"),
        (deg_derivs["xx"], r"Degraded $\partial^2 D / \partial x^2$"),
        (deg_derivs["yy"], r"Degraded $\partial^2 D / \partial y^2$"),
    ]

    for col_idx, ((c_img, c_title), (d_img, d_title)) in enumerate(zip(cols_clean, cols_deg)):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # Use symmetric intensity range around 0 for derivative maps
        v_max = max(float(np.percentile(np.abs(c_img), 99.0)), 15.0)

        im0 = ax_top.imshow(c_img, cmap="gray", vmin=-v_max, vmax=v_max)
        ax_top.set_title(c_title, fontsize=8.0, fontweight="bold", pad=3)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for sp in ax_top.spines.values():
            sp.set_edgecolor("#000000")
            sp.set_linewidth(0.8)

        im1 = ax_bot.imshow(d_img, cmap="gray", vmin=-v_max, vmax=v_max)
        ax_bot.set_title(d_title, fontsize=8.0, fontweight="bold", pad=3)
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])
        for sp in ax_bot.spines.values():
            sp.set_edgecolor("#000000")
            sp.set_linewidth(0.8)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08, wspace=0.06)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor="#FFFFFF", edgecolor="#FFFFFF", bbox_inches="tight")

    return fig
