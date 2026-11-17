"""Master Benchmark Runner for IEEE Sensors Journal Module 4.

Evaluates all 11 comparative destriping methods:
  1. 1D-GF (Cao et al., 2016)
  2. 1D-WLS (Dong et al., 2018)
  3. ADOM (Kim et al., 2023)
  4. BM3D (Dabov et al., 2007)
  5. DLS-NUC (He et al., 2019)
  6. SNRCNN (Kuai et al., 2017)
  7. LRSID (Chang et al., 2016)
  8. GSRSR (Chen et al., 2017)
  9. SNRWDNN (Guan et al., 2019)
  10. DestripeCycleGAN (Xiao et al., 2020)
  11. Ours (Unified Two-Stage Pipeline)
across all 6 benchmark infrared scenes (Img1 through Img6).

Computes all 7 quantitative metrics:
  - AVGE (Eq. 27) [lower is better]
  - MRD (Eq. 31) [lower is better]
  - AAD (Eq. 30) [lower is better]
  - ICV (Eq. 29) [higher is better]
  - rho (Roughness, Eq. 28) [lower is better]
  - PSNR (dB) [higher is better]
  - SSIM [higher is better]

Generates:
  - results/table1_quantitative_evaluation.csv
  - docs/table1_quantitative.tex (booktabs with bold best and underlined second-best)
  - results/fig10_yohann_scene1.png through fig15_llvip_scene6.png (2x6 panel 2D ROI + 3D energy)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

from common.benchmark import BenchmarkSuite, profile_stage
from common.metrics import evaluate_all_metrics
from common.plot_style import set_ieee_style
from mod04_baselines_and_benchmarks.src.filter_baselines import (
    solve_1d_gf,
    solve_1d_wls,
    solve_bm3d,
)
from mod04_baselines_and_benchmarks.src.learning_baselines import (
    solve_destripe_cyclegan,
    solve_dls_nuc,
    solve_snrcnn,
    solve_snrwdnn,
)
from mod04_baselines_and_benchmarks.src.optim_baselines import (
    solve_adom,
    solve_gsrsr,
    solve_lrsid,
)
from mod04_baselines_and_benchmarks.src.proposed_pipeline import solve_proposed_ours

# Ordered comparison methods
BENCHMARK_METHODS: List[str] = [
    "1D-GF",
    "1D-WLS",
    "ADOM",
    "BM3D",
    "DLS-NUC",
    "SNRCNN",
    "LRSID",
    "GSRSR",
    "SNRWDNN",
    "DestripeCycleGAN",
    "Ours",
]

# Scene definitions with corresponding Figure filenames and prominent ROIs (ymin, xmin, h, w)
SCENE_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "Img1_Checkerboard_Yohann.png",
        "fig_name": "fig10_yohann_scene1.png",
        "title": "Scene 1: Checkerboard (Yohann)",
        "roi_box": (40, 40, 50, 50),
    },
    {
        "name": "Img2_IndoorMonitor_Yohann.png",
        "fig_name": "fig11_yohann_scene2.png",
        "title": "Scene 2: Indoor Monitor (Yohann)",
        "roi_box": (80, 100, 50, 50),
    },
    {
        "name": "Img3_BuildingSkyline_He.png",
        "fig_name": "fig12_he_scene3.png",
        "title": "Scene 3: Building Skyline (He)",
        "roi_box": (70, 90, 50, 50),
    },
    {
        "name": "Img4_HousePedestrian_He.png",
        "fig_name": "fig13_he_scene4.png",
        "title": "Scene 4: House Pedestrian (He)",
        "roi_box": (90, 110, 50, 50),
    },
    {
        "name": "Img5_FacadeWindows_TNO.png",
        "fig_name": "fig14_tno_scene5.png",
        "title": "Scene 5: Facade Windows (TNO)",
        "roi_box": (60, 70, 50, 50),
    },
    {
        "name": "Img6_StreetCars_LLVIP.png",
        "fig_name": "fig15_llvip_scene6.png",
        "title": "Scene 6: Street Cars (LLVIP)",
        "roi_box": (120, 80, 50, 50),
    },
]

# Metric properties (True if higher is better, False if lower is better)
METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "AVGE": False,
    "MRD": False,
    "AAD": False,
    "ICV": True,
    "rho": False,
    "PSNR": True,
    "SSIM": True,
}


def run_single_method(
    method_name: str,
    degraded_img: np.ndarray,
    fast_mode: bool = False,
) -> Tuple[np.ndarray, float]:
    """Execute a single destriping algorithm and record its execution time."""
    t0 = time.perf_counter()
    if method_name == "1D-GF":
        c_est, _ = solve_1d_gf(degraded_img)
    elif method_name == "1D-WLS":
        c_est, _ = solve_1d_wls(degraded_img)
    elif method_name == "ADOM":
        c_est, _ = solve_adom(degraded_img, max_iter=30)
    elif method_name == "BM3D":
        stride_bm = 6 if fast_mode else 4
        c_est, _ = solve_bm3d(degraded_img, stride=stride_bm)
    elif method_name == "DLS-NUC":
        c_est, _ = solve_dls_nuc(degraded_img)
    elif method_name == "SNRCNN":
        c_est, _ = solve_snrcnn(degraded_img)
    elif method_name == "LRSID":
        c_est, _ = solve_lrsid(degraded_img, max_iter=30)
    elif method_name == "GSRSR":
        c_est, _ = solve_gsrsr(degraded_img, max_iter=30)
    elif method_name == "SNRWDNN":
        c_est, _ = solve_snrwdnn(degraded_img)
    elif method_name == "DestripeCycleGAN":
        c_est, _ = solve_destripe_cyclegan(degraded_img)
    elif method_name == "Ours":
        c_est, _, _ = solve_proposed_ours(degraded_img, fast_mode=fast_mode)
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    elapsed = time.perf_counter() - t0
    return c_est, elapsed


def evaluate_scene_all_methods(
    clean_img: np.ndarray,
    degraded_img: np.ndarray,
    fast_mode: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate Degraded + all 11 methods on a single scene."""
    clean_64 = clean_img.astype(np.float64)
    deg_64 = degraded_img.astype(np.float64)

    results: Dict[str, Dict[str, Any]] = {}

    # 0. Degraded reference
    m_deg = evaluate_all_metrics(clean_64, deg_64)
    results["Degraded"] = {
        "restored": deg_64,
        "metrics": {
            "AVGE": m_deg["AVGE"],
            "MRD": m_deg["MRD"],
            "AAD": m_deg["AAD"],
            "ICV": m_deg["ICV"],
            "rho": m_deg["Roughness_rho"],
            "PSNR": m_deg["PSNR"],
            "SSIM": m_deg["SSIM"],
        },
        "time_sec": 0.0,
    }

    # 1..11 Methods
    for method in BENCHMARK_METHODS:
        c_est, elapsed = run_single_method(method, deg_64, fast_mode=fast_mode)
        m = evaluate_all_metrics(clean_64, c_est)
        results[method] = {
            "restored": c_est,
            "metrics": {
                "AVGE": m["AVGE"],
                "MRD": m["MRD"],
                "AAD": m["AAD"],
                "ICV": m["ICV"],
                "rho": m["Roughness_rho"],
                "PSNR": m["PSNR"],
                "SSIM": m["SSIM"],
            },
            "time_sec": elapsed,
        }

    return results


def format_table1_latex(
    scene_evals: Dict[str, Dict[str, Dict[str, Any]]],
    output_tex_path: Path,
) -> str:
    """Format and save IEEE booktabs Table I with bold best and underlined second-best."""
    metrics_list = ["AVGE", "MRD", "AAD", "ICV", "rho", "PSNR", "SSIM"]
    methods = BENCHMARK_METHODS

    # Build header
    col_spec = "ll" + "c" * len(methods)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Quantitative Evaluation of 11 Destriping Methods Across Six Benchmark Infrared Scenes (Table I)}",
        r"\label{tab:table1_quantitative_evaluation}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"\textbf{Scene} & \textbf{Metric} & "
        + " & ".join([rf"\textbf{{{m}}}" for m in methods])
        + r" \\",
        r"\midrule",
    ]

    for s_idx, cfg in enumerate(SCENE_CONFIGS):
        scene_name = cfg["name"]
        short_name = cfg["title"].split(":")[1].strip()
        eval_dict = scene_evals[scene_name]

        for m_idx, metric in enumerate(metrics_list):
            higher_is_better = METRIC_HIGHER_IS_BETTER[metric]
            arrow = r"$\uparrow$" if higher_is_better else r"$\downarrow$"

            # Extract values for all 11 methods
            vals = [eval_dict[m]["metrics"][metric] for m in methods]

            # Determine best and second-best
            if higher_is_better:
                sorted_indices = np.argsort(vals)[::-1]
            else:
                sorted_indices = np.argsort(vals)

            best_idx = sorted_indices[0]
            second_idx = sorted_indices[1]

            row_cells = []
            for idx, val in enumerate(vals):
                if metric in ["PSNR", "ICV"]:
                    val_str = f"{val:.2f}"
                elif metric in ["SSIM"]:
                    val_str = f"{val:.4f}"
                elif metric in ["AVGE", "MRD", "AAD", "rho"]:
                    val_str = f"{val:.4f}"
                else:
                    val_str = f"{val:.3f}"

                if idx == best_idx:
                    val_str = rf"\textbf{{{val_str}}}"
                elif idx == second_idx:
                    val_str = rf"\underline{{{val_str}}}"

                row_cells.append(val_str)

            scene_col = rf"\multirow{{{len(metrics_list)}}}{{*}}{{{short_name}}}" if m_idx == 0 else ""
            lines.append(f"{scene_col} & {metric} {arrow} & " + " & ".join(row_cells) + r" \\")

        if s_idx < len(SCENE_CONFIGS) - 1:
            lines.append(r"\midrule")

    # Add Average Row across all 6 scenes
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textbf{Average Performance}} & " + " & " * (len(methods) - 1) + r" \\")

    for metric in metrics_list:
        higher_is_better = METRIC_HIGHER_IS_BETTER[metric]
        arrow = r"$\uparrow$" if higher_is_better else r"$\downarrow$"

        mean_vals = []
        for m in methods:
            scene_vals = [scene_evals[cfg["name"]][m]["metrics"][metric] for cfg in SCENE_CONFIGS]
            mean_vals.append(float(np.mean(scene_vals)))

        if higher_is_better:
            sorted_indices = np.argsort(mean_vals)[::-1]
        else:
            sorted_indices = np.argsort(mean_vals)

        best_idx = sorted_indices[0]
        second_idx = sorted_indices[1]

        row_cells = []
        for idx, val in enumerate(mean_vals):
            if metric in ["PSNR", "ICV"]:
                val_str = f"{val:.2f}"
            elif metric in ["SSIM", "AVGE", "MRD", "AAD", "rho"]:
                val_str = f"{val:.4f}"
            else:
                val_str = f"{val:.3f}"

            if idx == best_idx:
                val_str = rf"\textbf{{{val_str}}}"
            elif idx == second_idx:
                val_str = rf"\underline{{{val_str}}}"

            row_cells.append(val_str)

        lines.append(f" & {metric} {arrow} & " + " & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
        ]
    )

    tex_content = "\n".join(lines)
    output_tex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    return tex_content


def plot_qualitative_panel(
    scene_title: str,
    all_methods_dict: Dict[str, np.ndarray],
    roi_box: Tuple[int, int, int, int],
    save_path: Path,
) -> None:
    """Generate 2x6 method panel figure (Figs. 10-15).

    Layout:
      - 12 methods total: Degraded + 11 comparative methods.
      - Arranged in 2 rows of 6 methods.
      - Each method shows:
          Top: 2D restored image with red bounding box + magnified inset.
          Bottom: 3D Global Energy surface map (cmap='jet').
    """
    set_ieee_style()
    method_keys = ["Degraded"] + BENCHMARK_METHODS
    H, W = next(iter(all_methods_dict.values())).shape
    ymin, xmin, h_roi, w_roi = roi_box

    # Create figure: 4 rows x 6 columns
    # Row 0: 2D images for methods 0..5
    # Row 1: 3D energy for methods 0..5
    # Row 2: 2D images for methods 6..11
    # Row 3: 3D energy for methods 6..11
    fig = plt.figure(figsize=(19.0, 11.5), dpi=300, facecolor="#FFFFFF")

    # Grid for 3D energy surfaces (subsample with stride 3 for crisp 3D plot)
    stride_3d = 3
    x_grid = np.arange(0, W, stride_3d)
    y_grid = np.arange(0, H, stride_3d)
    X, Y = np.meshgrid(x_grid, y_grid)

    for idx, m_name in enumerate(method_keys):
        img_arr = all_methods_dict[m_name]
        block_row = idx // 6  # 0 or 1
        block_col = idx % 6   # 0 to 5

        # ----------------- 2D Subplot -----------------
        row_2d = block_row * 2
        ax_2d = fig.add_subplot(4, 6, row_2d * 6 + block_col + 1)
        ax_2d.imshow(img_arr, cmap="gray", vmin=0, vmax=255, origin="upper")

        # Title
        display_name = m_name if m_name != "Ours" else "Ours (Proposed)"
        is_ours = m_name == "Ours"
        title_color = "#000000" if not is_ours else "#b30000"
        ax_2d.set_title(display_name, fontsize=8.5, fontweight="bold", color=title_color, pad=3)
        ax_2d.set_xticks([])
        ax_2d.set_yticks([])
        for spine in ax_2d.spines.values():
            spine.set_edgecolor("#000000" if not is_ours else "#b30000")
            spine.set_linewidth(1.0 if not is_ours else 1.5)

        # Red ROI bounding box
        rect = patches.Rectangle(
            (xmin, ymin),
            w_roi,
            h_roi,
            linewidth=1.2,
            edgecolor="#FF0000",
            facecolor="none",
            linestyle="-",
        )
        ax_2d.add_patch(rect)

        # Magnified zoom inset in lower right corner
        ax_ins = inset_axes(
            ax_2d,
            width="38%",
            height="38%",
            loc="lower right",
            borderpad=0.4,
        )
        roi_patch = img_arr[ymin : ymin + h_roi, xmin : xmin + w_roi]
        ax_ins.imshow(roi_patch, cmap="gray", vmin=0, vmax=255, origin="upper")
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        for spine in ax_ins.spines.values():
            spine.set_edgecolor("#FF0000")
            spine.set_linewidth(1.4)

        # ----------------- 3D Surface Subplot -----------------
        row_3d = block_row * 2 + 1
        ax_3d = fig.add_subplot(4, 6, row_3d * 6 + block_col + 1, projection="3d")
        Z = img_arr[::stride_3d, ::stride_3d]

        surf = ax_3d.plot_surface(
            X,
            Y,
            Z,
            cmap="jet",
            vmin=0.0,
            vmax=255.0,
            linewidth=0.0,
            antialiased=True,
            rstride=1,
            cstride=1,
            shade=True,
        )
        ax_3d.set_zlim(0.0, 255.0)
        ax_3d.view_init(elev=50, azim=-60)
        ax_3d.set_xticks([])
        ax_3d.set_yticks([])
        ax_3d.set_zticks([])
        ax_3d.set_facecolor("#FFFFFF")
        ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_3d.grid(False)

    plt.suptitle(
        f"Comparative Destriping and 3D Energy Surfaces - {scene_title}",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved Figure] {save_path.name}")


def main() -> None:
    set_ieee_style()
    suite = BenchmarkSuite("Module 4: 11-Method Benchmark Suite")

    clean_dir = Path("data/clean_scenes")
    deg_dir = Path("data/degraded_scenes")
    results_dir = Path("mod04_baselines_and_benchmarks/results")
    docs_dir = Path("mod04_baselines_and_benchmarks/docs")
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  IEEE Sensors Journal - Module 4: 11-Method Master Benchmark Suite")
    print("=" * 75)

    all_scene_evaluations: Dict[str, Dict[str, Dict[str, Any]]] = {}
    csv_rows: List[Dict[str, Any]] = []

    for cfg in SCENE_CONFIGS:
        scene_name = cfg["name"]
        clean_path = clean_dir / scene_name
        deg_path = deg_dir / f"{clean_path.stem}_case1.png"

        if not clean_path.exists() or not deg_path.exists():
            print(f"[WARN] Skipping {scene_name}: file missing.")
            continue

        print(f"\n[Evaluating Scene] {scene_name}...")
        clean_img = np.array(Image.open(clean_path), dtype=np.float64)
        deg_img = np.array(Image.open(deg_path), dtype=np.float64)

        scene_results = evaluate_scene_all_methods(clean_img, deg_img, fast_mode=False)
        all_scene_evaluations[scene_name] = scene_results

        # Compile rows for CSV
        for method_name, res in scene_results.items():
            row = {
                "Scene": scene_name,
                "Method": method_name,
                "AVGE": res["metrics"]["AVGE"],
                "MRD": res["metrics"]["MRD"],
                "AAD": res["metrics"]["AAD"],
                "ICV": res["metrics"]["ICV"],
                "rho": res["metrics"]["rho"],
                "PSNR_dB": res["metrics"]["PSNR"],
                "SSIM": res["metrics"]["SSIM"],
                "Time_sec": res["time_sec"],
            }
            csv_rows.append(row)

        # Generate qualitative 2x6 panel figure for this scene
        fig_path = results_dir / cfg["fig_name"]
        all_methods_images = {k: v["restored"] for k, v in scene_results.items()}
        plot_qualitative_panel(
            scene_title=cfg["title"],
            all_methods_dict=all_methods_images,
            roi_box=cfg["roi_box"],
            save_path=fig_path,
        )

    # 1. Export CSV
    df = pd.DataFrame(csv_rows)
    csv_path = results_dir / "table1_quantitative_evaluation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Exported CSV] Table I metrics saved to: {csv_path}")

    # 2. Export LaTeX Table I
    tex_path = docs_dir / "table1_quantitative.tex"
    format_table1_latex(all_scene_evaluations, tex_path)
    print(f"[Exported LaTeX] Booktabs Table I saved to: {tex_path}")

    # 3. Print mean quantitative summary
    summary_mean = (
        df.groupby("Method")
        .agg(
            {
                "AVGE": "mean",
                "MRD": "mean",
                "AAD": "mean",
                "ICV": "mean",
                "rho": "mean",
                "PSNR_dB": "mean",
                "SSIM": "mean",
                "Time_sec": "mean",
            }
        )
        .reindex(["Degraded"] + BENCHMARK_METHODS)
    )

    print("\n" + "=" * 75)
    print("  Table I Equivalent: Mean Quantitative Performance Across 6 Scenes")
    print("=" * 75)
    print(summary_mean.round(4).to_string())


if __name__ == "__main__":
    main()
