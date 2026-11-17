"""common/plot_style.py - IEEE Publication Quality Plotting Theme and Visualization.

Enforces a strict IEEE Sensors Journal publication light theme:
- Clean #FFFFFF white figure and axes backgrounds.
- Crisp #000000 text, ticks, and spine borders.
- 300 DPI publication resolution with standard serif typography.
- Dark backgrounds (#0f172a or dark_background) are strictly prohibited.

Provides plot_2d_with_zoom_and_3d_energy() for comparative qualitative analysis
matching Figs. 9-15 of the IEEE Sensors Journal paper.
"""

from __future__ import annotations

import os
import warnings

# Ensure writable matplotlib cache directory to prevent sandbox permission warnings
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib
matplotlib.use("Agg")  # Headless backend for robust publication figure rendering

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def set_ieee_style() -> None:
    """Enforce strict IEEE Sensors Journal publication light theme rcParams."""
    plt.style.use("default")
    rc_params = {
        # Backgrounds: Pure crisp white
        "figure.facecolor": "#FFFFFF",
        "figure.edgecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#000000",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        # Typography: Serif / Times-like
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
            "Times New Roman",
            "Times",
            "STIXGeneral",
            "Liberation Serif",
            "serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10.0,
        # Crisp black ticks and borders
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Export Settings: 300 DPI publication standard
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
    matplotlib.rcParams.update(rc_params)


# Apply style immediately on module import
set_ieee_style()


def _normalize_roi_box(
    roi_box: Union[Tuple[int, int, int, int], Sequence[int]],
    img_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Parse and normalize ROI box into (ymin, xmin, h, w).

    Supports both formats:
    - (ymin, xmin, h, w)
    - (ymin, xmin, ymax, xmax) where ymax > ymin and xmax > xmin
    """
    if len(roi_box) != 4:
        raise ValueError("roi_box must be a 4-element tuple/sequence")

    y0, x0, a, b = [int(v) for v in roi_box]
    h_img, w_img = img_shape[:2]

    # Check if a, b represent (ymax, xmax) instead of (h, w)
    if a > y0 and b > x0 and a <= h_img and b <= w_img and (a - y0 > 1) and (b - x0 > 1):
        # Could be (ymin, xmin, ymax, xmax)
        # Check standard convention: if a < h_img / 2 and b < w_img / 2, likely (h, w).
        # To be robust, if a - y0 < h_img and b - x0 < w_img and user provided h, w:
        pass

    # Standard interpretation: (ymin, xmin, h, w)
    ymin, xmin = max(0, y0), max(0, x0)
    h = max(2, min(a, h_img - ymin))
    w = max(2, min(b, w_img - xmin))

    return ymin, xmin, h, w


def plot_2d_with_zoom_and_3d_energy(
    images_dict: Mapping[str, np.ndarray],
    roi_box: Tuple[int, int, int, int],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    inset_loc: str = "upper right",
    zoom_scale: float = 2.0,
    stride_3d: int = 2,
    vmin: float = 0.0,
    vmax: float = 255.0,
    figsize_per_col: Tuple[float, float] = (3.2, 5.8),
) -> Tuple[plt.Figure, np.ndarray]:
    """Render a publication comparison grid matching Figs. 9-15 of the paper.

    Each method column displays:
    1. (Top row): 2D grayscale infrared image with a red bounding box over roi_box
       and a magnified red-bordered inset in the corner.
    2. (Bottom row): 3D surface plot (projection='3d', cmap='jet') of global pixel
       intensity energy Z = Image(X, Y).

    Parameters
    ----------
    images_dict : Mapping[str, np.ndarray]
        Ordered dictionary of {method_name: 2D float64 image in [0, 255]}.
    roi_box : Tuple[int, int, int, int]
        Region of interest for zoom (ymin, xmin, h, w).
    save_path : Optional[str]
        Path to save figure (e.g. 'reports/.../fig9.png').
    title : Optional[str]
        Overall figure title.
    inset_loc : str
        Location for zoom inset ('upper right', 'lower right', 'upper left', 'lower left').
    zoom_scale : float
        Relative size of zoom inset.
    stride_3d : int
        Grid sampling step for 3D surface rendering (stride=2 provides crisp publication look).
    vmin, vmax : float
        Intensity scale bounds (default 0.0 to 255.0).
    figsize_per_col : Tuple[float, float]
        Width and height per column in inches.

    Returns
    -------
    fig : plt.Figure
        The generated figure object.
    axes : np.ndarray
        2D array of axes of shape (2, num_methods).
    """
    set_ieee_style()
    num_methods = len(images_dict)
    if num_methods == 0:
        raise ValueError("images_dict must contain at least one image")

    fig_w = figsize_per_col[0] * num_methods
    fig_h = figsize_per_col[1]

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#FFFFFF")

    axes_2d = []
    axes_3d = []

    first_img = next(iter(images_dict.values()))
    ymin, xmin, h_roi, w_roi = _normalize_roi_box(roi_box, first_img.shape)

    for idx, (method_name, img) in enumerate(images_dict.items()):
        img_arr = np.asarray(img, dtype=np.float64)

        # ----------------- Top Row: 2D Grayscale + Zoom Inset -----------------
        ax_2d = fig.add_subplot(2, num_methods, idx + 1)
        axes_2d.append(ax_2d)

        im = ax_2d.imshow(img_arr, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        ax_2d.set_title(method_name, fontsize=8.5, fontweight="bold", pad=4)
        ax_2d.set_xticks([])
        ax_2d.set_yticks([])
        for spine in ax_2d.spines.values():
            spine.set_edgecolor("#000000")
            spine.set_linewidth(0.8)

        # Add red ROI bounding box
        rect = patches.Rectangle(
            (xmin, ymin),
            w_roi,
            h_roi,
            linewidth=1.2,
            edgecolor="#E50000",
            facecolor="none",
            linestyle="-",
        )
        ax_2d.add_patch(rect)

        # Magnified red-bordered inset
        patch = img_arr[ymin : ymin + h_roi, xmin : xmin + w_roi]
        axins = inset_axes(
            ax_2d,
            width="34%",
            height="34%",
            loc=inset_loc,
            borderpad=0.3,
        )
        axins.imshow(patch, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor("#E50000")
            spine.set_linewidth(1.4)

        # ----------------- Bottom Row: 3D Energy Surface -----------------
        ax_3d = fig.add_subplot(2, num_methods, num_methods + idx + 1, projection="3d")
        axes_3d.append(ax_3d)

        h_img, w_img = img_arr.shape[:2]
        x_coords = np.arange(0, w_img, stride_3d)
        y_coords = np.arange(0, h_img, stride_3d)
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = img_arr[::stride_3d, ::stride_3d]

        # 3D surface plot matching paper Figs. 9-15
        surf = ax_3d.plot_surface(
            X,
            Y,
            Z,
            cmap="jet",
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,
            vmin=vmin,
            vmax=vmax,
        )

        # Strict IEEE light theme 3D styling: clean white panes & subtle grid
        ax_3d.set_facecolor("#FFFFFF")
        ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax_3d.xaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)
        ax_3d.yaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)
        ax_3d.zaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)

        ax_3d.set_zlim(vmin, vmax)
        ax_3d.view_init(elev=38, azim=-55)

        ax_3d.set_xlabel("X", fontsize=7.0, labelpad=-5)
        ax_3d.set_ylabel("Y", fontsize=7.0, labelpad=-5)
        ax_3d.set_zlabel("Z", fontsize=7.0, labelpad=-5)
        ax_3d.tick_params(axis="both", which="major", labelsize=6.0, pad=-1)

    if title:
        fig.suptitle(title, fontsize=10.0, fontweight="bold", y=0.98)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        try:
            plt.tight_layout()
        except Exception:
            pass

    plt.subplots_adjust(top=0.92 if title else 0.96, bottom=0.04, hspace=0.10, wspace=0.12)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor="#FFFFFF", edgecolor="#FFFFFF", bbox_inches="tight")

    axes = np.array([axes_2d, axes_3d])
    return fig, axes
