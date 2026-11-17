"""mod03_stage2_patch_lowrank/src/run_mod03.py

Master CLI Runner for Stage 2 Patch Low-Rank Prior Recovery.

Executes:
1. Stage 2 denoising across all 6 benchmark scenes on Stage 1 outputs.
2. Generates results/stage1_vs_stage2_metrics.csv comparing Degraded vs Stage 1 vs Stage 2.
3. Produces results/fig7_8_patch_observation_pipeline.png (Figs. 7 & 8 schematic).
4. Produces results/fig_stage1_vs_stage2_comparison.png (2D zoom + 3D energy comparison).
5. Produces results/rank_distribution_histogram.png (adaptive rank k histogram).
6. Profiles multi-threaded execution (Parfor equivalent) and exports telemetry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common.benchmark import BenchmarkSuite, profile_stage
from common.dataset_generator import BENCHMARK_SCENES, load_scene
from common.metrics import evaluate_all_metrics
from common.plot_style import plot_2d_with_zoom_and_3d_energy, set_ieee_style

from mod01_priors_and_noise_simulator.src.noise_simulator import simulate_case2_periodic
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.patch_matching import (
    extract_sliding_patches_view,
    find_similar_patches_for_ref,
)
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank


@profile_stage(name="Stage2_Batch_Execution")
def run_stage2_batch(
    scenes: List[str] = BENCHMARK_SCENES,
    stage1_dir: str = "mod02_stage1_admm_destriping/results",
    results_dir: str = "mod03_stage2_patch_lowrank/results",
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]]:
    """Execute Stage 2 patch recovery across all benchmark scenes."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_path = Path(stage1_dir)

    metrics_rows: List[Dict[str, Any]] = []
    sol_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]] = {}

    print("\n[Stage 2] Processing 6 benchmark infrared scenes via Patch Low-Rank SVD...")
    for scene_name in scenes:
        clean = load_scene(scene_name)
        stem = Path(scene_name).stem

        # Load degraded observation (Case 2)
        sim = simulate_case2_periodic(clean, seed=42)
        deg = sim.degraded

        # Load or compute Stage 1 output
        stage1_file = s1_path / f"{stem}_stage1.png"
        if stage1_file.is_file():
            C_stage1 = np.array(Image.open(stage1_file), dtype=np.float64)
        else:
            print(f"  Generating Stage 1 baseline for {scene_name}...")
            C_stage1, _, _ = solve_stage1_admm(deg, clean_ref=clean, maxiter=75, verbose=False)

        print(f"  --> Stage 2 Denoising: {scene_name:30s}")
        C_stage2, diag = denoise_stage2_patch_lowrank(
            img=C_stage1,
            patch_size=11,
            stride=3,
            search_radius=15,
            num_similar_patches=50,
            n_jobs=n_jobs,
            return_diagnostics=True,
        )

        # Evaluate metrics across Degraded, Stage 1, Stage 2
        m_deg = evaluate_all_metrics(clean, deg)
        m_s1 = evaluate_all_metrics(clean, C_stage1)
        m_s2 = evaluate_all_metrics(clean, C_stage2)

        psnr_gain_total = m_s2["PSNR"] - m_deg["PSNR"]
        psnr_gain_stage2 = m_s2["PSNR"] - m_s1["PSNR"]
        ssim_gain_total = m_s2["SSIM"] - m_deg["SSIM"]
        ssim_gain_stage2 = m_s2["SSIM"] - m_s1["SSIM"]

        print(
            f"      [Degraded] PSNR: {m_deg['PSNR']:.2f} dB, SSIM: {m_deg['SSIM']:.4f}\n"
            f"      [Stage 1 ] PSNR: {m_s1['PSNR']:.2f} dB, SSIM: {m_s1['SSIM']:.4f} (ΔPSNR: +{m_s1['PSNR'] - m_deg['PSNR']:.2f} dB)\n"
            f"      [Stage 2 ] PSNR: {m_s2['PSNR']:.2f} dB, SSIM: {m_s2['SSIM']:.4f} "
            f"(ΔStage2: +{psnr_gain_stage2:.2f} dB, Total Gain: +{psnr_gain_total:.2f} dB) | "
            f"Mean Rank: {diag['mean_rank']:.2f} | Time: {diag['elapsed_sec']:.2f}s"
        )

        row = {
            "Scene": scene_name,
            "Degraded_PSNR": round(m_deg["PSNR"], 2),
            "Stage1_PSNR": round(m_s1["PSNR"], 2),
            "Stage2_PSNR": round(m_s2["PSNR"], 2),
            "Stage2_Gain_over_S1_dB": round(psnr_gain_stage2, 2),
            "Total_PSNR_Gain_dB": round(psnr_gain_total, 2),
            "Degraded_SSIM": round(m_deg["SSIM"], 4),
            "Stage1_SSIM": round(m_s1["SSIM"], 4),
            "Stage2_SSIM": round(m_s2["SSIM"], 4),
            "Stage2_SSIM_Gain": round(ssim_gain_stage2, 4),
            "Total_SSIM_Gain": round(ssim_gain_total, 4),
            "Stage2_AVGE": round(m_s2["AVGE"], 2),
            "Stage2_Roughness": round(m_s2["Roughness_rho"], 4),
            "Stage2_ICV": round(m_s2["ICV"], 2),
            "Mean_Rank_k": round(diag["mean_rank"], 2),
            "Elapsed_Sec": round(diag["elapsed_sec"], 3),
        }
        metrics_rows.append(row)
        sol_dict[scene_name] = (deg, C_stage1, C_stage2, diag)

        # Save restored image
        save_img_path = out_dir / f"{stem}_stage2.png"
        uint8_arr = np.clip(np.round(C_stage2), 0, 255).astype(np.uint8)
        Image.fromarray(uint8_arr, mode="L").save(save_img_path)

    metrics_df = pd.DataFrame(metrics_rows)
    csv_path = out_dir / "stage1_vs_stage2_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Exported Stage 1 vs Stage 2 metrics to: {csv_path}")

    return metrics_df, sol_dict


@profile_stage(name="Stage2_Visualizations")
def generate_stage2_figures(
    sol_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]],
    results_dir: str = "mod03_stage2_patch_lowrank/results",
) -> None:
    """Generate Figs. 7-8 schematic, Stage 1 vs Stage 2 comparison, and rank histogram."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ieee_style()

    # --------------------------------------------------------------------------
    # 1. Figure 7 & 8 Schematic: Patch Similarity Matching & Observation Matrix
    # --------------------------------------------------------------------------
    print("  Generating fig7_8_patch_observation_pipeline.png...")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), facecolor="#FFFFFF")

    primary_scene = "Img3_BuildingSkyline_He.png"
    clean = load_scene(primary_scene)
    deg, C_stage1, C_stage2, diag = sol_dict[primary_scene]

    # Panel (a): Fig. 7 Patch matching demonstration
    ax0 = axes[0]
    ax0.imshow(C_stage1, cmap="gray", vmin=0, vmax=255)
    r_j, c_j = 80, 100
    R = 15
    B = 11

    # Draw local search window (blue)
    rect_win = patches.Rectangle(
        (c_j - R, r_j - R),
        2 * R + B,
        2 * R + B,
        linewidth=1.2,
        edgecolor="#1976D2",
        facecolor="none",
        linestyle="--",
        label=r"Search Window $W_s \times W_s$",
    )
    ax0.add_patch(rect_win)

    # Draw reference patch y_j (red)
    rect_ref = patches.Rectangle(
        (c_j, r_j),
        B,
        B,
        linewidth=1.6,
        edgecolor="#D32F2F",
        facecolor="none",
        label=r"Reference Patch $y_j$ ($11 \times 11$)",
    )
    ax0.add_patch(rect_ref)

    # Find and plot a few similar patches (green)
    patches_view = extract_sliding_patches_view(C_stage1, patch_size=B)
    group = find_similar_patches_for_ref((r_j, c_j), patches_view, patch_size=B, search_radius=R, num_similar_patches=50)

    for r_c, c_c in group.patch_coords[1:6]:
        rect_cand = patches.Rectangle(
            (c_c, r_c),
            B,
            B,
            linewidth=1.0,
            edgecolor="#2E7D32",
            facecolor="none",
        )
        ax0.add_patch(rect_cand)

    ax0.set_title("(a) Fig. 7: Patch Similarity Search & Grouping", fontsize=8.0, fontweight="bold", pad=4)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.legend(loc="lower right", fontsize=6.5, edgecolor="#000000", framealpha=0.9)

    # Panel (b): Fig. 8 Observation matrix & SVD singular values
    ax1 = axes[1]
    G = group.observation_matrix  # Shape (121, 50)
    col_means = np.mean(G, axis=0, keepdims=True)
    _, s_vals, _ = np.linalg.svd(G - col_means, full_matrices=False)

    tau_sq = diag["tau_squared"]
    tail_sums = np.array([np.sum(s_vals[j:] ** 2) for j in range(len(s_vals) + 1)])
    sel_k = 1
    for k in range(1, len(s_vals)):
        if tail_sums[k] <= tau_sq:
            sel_k = k
            break

    x_s = np.arange(1, len(s_vals) + 1)
    ax1.plot(x_s, s_vals, "o-", color="#000000", markersize=3.0, linewidth=1.1, label=r"Singular Values $\sigma_i$")
    ax1.axvline(sel_k, color="#D32F2F", linestyle="--", linewidth=1.2, label=f"Adaptive Rank $k = {sel_k}$ (Eq. 26)")
    ax1.set_title(r"(b) Fig. 8: Observation Matrix $\mathbf{G} \in \mathbb{R}^{121 \times 50}$ SVD Spectrum", fontsize=8.0, fontweight="bold", pad=4)
    ax1.set_xlabel("Singular Value Index $i$", fontsize=7.5)
    ax1.set_ylabel("Singular Value Magnitude", fontsize=7.5)
    ax1.grid(True, linestyle=":", color="#CCCCCC", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=6.5, edgecolor="#000000")

    for ax in axes:
        for sp in ax.spines.values():
            sp.set_edgecolor("#000000")
            sp.set_linewidth(0.8)

    plt.tight_layout()
    fig_sched_path = str(out_dir / "fig7_8_patch_observation_pipeline.png")
    fig.savefig(fig_sched_path, dpi=300, facecolor="#FFFFFF", edgecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)
    print(f"  Exported Fig. 7-8 schematic to: {fig_sched_path}")

    # --------------------------------------------------------------------------
    # 2. Stage 1 vs Stage 2 Comparative Publication Figure
    # --------------------------------------------------------------------------
    print("  Generating fig_stage1_vs_stage2_comparison.png...")
    images_dict = {
        "Degraded Input (D)": deg,
        "Stage 1: ADMM (C1)": C_stage1,
        "Stage 2: Low-Rank (C2)": C_stage2,
        "Clean Reference (C)": clean,
    }
    plot_2d_with_zoom_and_3d_energy(
        images_dict=images_dict,
        roi_box=(70, 75, 45, 45),
        save_path=str(out_dir / "fig_stage1_vs_stage2_comparison.png"),
        title="Comparison of Infrared Restoration: Degraded vs. Stage 1 vs. Stage 2 vs. Clean",
        figsize_per_col=(2.8, 5.2),
    )

    # --------------------------------------------------------------------------
    # 3. Adaptive Rank k Histogram Distribution
    # --------------------------------------------------------------------------
    print("  Generating rank_distribution_histogram.png...")
    fig_hist, ax_h = plt.subplots(figsize=(4.5, 3.0), facecolor="#FFFFFF")
    ranks = diag["selected_ranks"]
    bins = np.arange(0.5, 20.5, 1.0)
    ax_h.hist(ranks, bins=bins, color="#1976D2", edgecolor="#000000", linewidth=0.8, rwidth=0.85, alpha=0.85)
    ax_h.axvline(np.mean(ranks), color="#D32F2F", linestyle="--", linewidth=1.2, label=f"Mean Rank: {np.mean(ranks):.2f}")

    ax_h.set_title("Distribution of Selected Adaptive Ranks $k$ (Eq. 26)", fontsize=8.0, fontweight="bold", pad=4)
    ax_h.set_xlabel("Adaptive Truncation Rank $k$", fontsize=7.5)
    ax_h.set_ylabel("Frequency (Number of Patches)", fontsize=7.5)
    ax_h.set_xlim(0, 20)
    ax_h.grid(True, linestyle=":", color="#CCCCCC", alpha=0.6)
    ax_h.legend(loc="upper right", fontsize=7.0, edgecolor="#000000")

    for sp in ax_h.spines.values():
        sp.set_edgecolor("#000000")
        sp.set_linewidth(0.8)

    plt.tight_layout()
    hist_path = str(out_dir / "rank_distribution_histogram.png")
    fig_hist.savefig(hist_path, dpi=300, facecolor="#FFFFFF", edgecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig_hist)
    print(f"  Exported rank distribution histogram to: {hist_path}")


def main() -> None:
    print("======================================================================")
    print("  IEEE Sensors Journal - Module 3: Stage 2 Patch Low-Rank Runner")
    print("======================================================================")

    # 1. Execute Stage 2 batch processing
    metrics_df, sol_dict = run_stage2_batch(n_jobs=-1)

    # 2. Generate publication figures
    print("\n[Step 2/2] Generating Stage 2 publication figures and schematics...")
    generate_stage2_figures(sol_dict)

    # 3. Telemetry summary
    print("\n======================================================================")
    print("  Stage 2 Performance Summary: Degraded vs Stage 1 vs Stage 2")
    print("======================================================================")
    cols = ["Scene", "Degraded_PSNR", "Stage1_PSNR", "Stage2_PSNR", "Stage2_Gain_over_S1_dB", "Total_PSNR_Gain_dB", "Stage2_SSIM", "Mean_Rank_k", "Elapsed_Sec"]
    print(metrics_df[cols].to_string(index=False))

    suite = BenchmarkSuite.get_global_suite()
    print("\nComputational Profiling:")
    print(suite.summary_table())
    print("\nModule 3 Stage 2 execution completed successfully.")


if __name__ == "__main__":
    main()
