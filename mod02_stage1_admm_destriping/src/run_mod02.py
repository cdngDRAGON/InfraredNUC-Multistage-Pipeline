"""mod02_stage1_admm_destriping/src/run_mod02.py

Master CLI Runner for Stage 1 ADMM Destriping.

Executes Stage 1 across all 6 benchmark infrared scenes:
1. Solves the constrained optimization problem (Eqs. 11-21).
2. Generates results/stage1_metrics.csv with full 7-metric telemetry and gains.
3. Produces results/fig_stage1_decomposition.png (Clean, Degraded, Stripe, Restored 2D+3D).
4. Produces results/fig_stage1_admm_convergence.png (Objective, RelChange, PSNR/SSIM curves).
5. Profiles execution wall-clock time and peak memory footprint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
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


@profile_stage(name="Stage1_ADMM_Batch_Restoration")
def run_stage1_batch(
    scenes: List[str] = BENCHMARK_SCENES,
    results_dir: str = "mod02_stage1_admm_destriping/results",
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]]]:
    """Execute Stage 1 ADMM destriping across benchmark scenes."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    sol_dict: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}

    print("\n[Stage 1] Processing 6 benchmark infrared scenes via ADMM...")
    for scene_name in scenes:
        clean = load_scene(scene_name)
        sim = simulate_case2_periodic(clean, seed=42)
        deg = sim.degraded

        deg_metrics = evaluate_all_metrics(clean, deg)

        print(f"  --> Solving Scene: {scene_name:30s} (Degraded PSNR: {deg_metrics['PSNR']:.2f} dB)")
        C_est, S_est, hist_df = solve_stage1_admm(
            D=deg,
            clean_ref=clean,
            lambda1=0.5,
            lambda2=0.05,
            lambda3=2.0,
            rho=0.5,
            maxiter=75,
            tol=1e-4,
            verbose=False,
        )

        stage1_metrics = evaluate_all_metrics(clean, C_est)
        psnr_gain = stage1_metrics["PSNR"] - deg_metrics["PSNR"]
        ssim_gain = stage1_metrics["SSIM"] - deg_metrics["SSIM"]
        last_iter = int(hist_df["iteration"].iloc[-1])
        elapsed_sec = float(hist_df["elapsed_sec"].iloc[-1])

        print(
            f"      [Solved] Stage 1 PSNR: {stage1_metrics['PSNR']:.2f} dB (Gain: +{psnr_gain:.2f} dB) | "
            f"SSIM: {stage1_metrics['SSIM']:.4f} (Gain: +{ssim_gain:.4f}) | "
            f"Iter: {last_iter} | Time: {elapsed_sec:.2f}s"
        )

        row = {
            "Scene": scene_name,
            "Degraded_PSNR": round(deg_metrics["PSNR"], 2),
            "Degraded_SSIM": round(deg_metrics["SSIM"], 4),
            "Degraded_AVGE": round(deg_metrics["AVGE"], 2),
            "Degraded_Roughness": round(deg_metrics["Roughness_rho"], 4),
            "Stage1_PSNR": round(stage1_metrics["PSNR"], 2),
            "Stage1_SSIM": round(stage1_metrics["SSIM"], 4),
            "Stage1_AVGE": round(stage1_metrics["AVGE"], 2),
            "Stage1_Roughness": round(stage1_metrics["Roughness_rho"], 4),
            "PSNR_Gain_dB": round(psnr_gain, 2),
            "SSIM_Gain": round(ssim_gain, 4),
            "Iterations": last_iter,
            "Elapsed_Sec": round(elapsed_sec, 3),
        }
        metrics_rows.append(row)
        sol_dict[scene_name] = (C_est, S_est, hist_df)

        # Save restored image
        stem = Path(scene_name).stem
        save_img_path = out_dir / f"{stem}_stage1.png"
        uint8_arr = np.clip(np.round(C_est), 0, 255).astype(np.uint8)
        Image.fromarray(uint8_arr, mode="L").save(save_img_path)

    metrics_df = pd.DataFrame(metrics_rows)
    csv_path = out_dir / "stage1_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Exported Stage 1 benchmark metrics to: {csv_path}")

    return metrics_df, sol_dict


@profile_stage(name="Stage1_Visualization_Generation")
def generate_stage1_figures(
    sol_dict: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    results_dir: str = "mod02_stage1_admm_destriping/results",
) -> None:
    """Generate publication figures for Stage 1 decomposition and convergence."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Figure 1: Decomposition (Clean, Degraded, Estimated Stripe, Stage 1 Restored)
    primary_scene = "Img3_BuildingSkyline_He.png"
    clean = load_scene(primary_scene)
    sim = simulate_case2_periodic(clean, seed=42)
    deg = sim.degraded
    C_est, S_est, hist_df = sol_dict[primary_scene]

    # Map stripe to [0, 255] visualization range
    S_vis = np.clip(S_est + 128.0, 0.0, 255.0)

    images_dict = {
        "Clean Reference (C)": clean,
        "Degraded Input (D)": deg,
        "Estimated Stripe (S)": S_vis,
        "Stage 1 Restored (C1)": C_est,
    }

    print("  Generating fig_stage1_decomposition.png (2D Zoom + 3D Energy Surface)...")
    plot_2d_with_zoom_and_3d_energy(
        images_dict=images_dict,
        roi_box=(70, 75, 45, 45),
        save_path=str(out_dir / "fig_stage1_decomposition.png"),
        title="Stage 1 ADMM High-Order Directional Destriping Decomposition",
        figsize_per_col=(2.8, 5.2),
    )

    # 2. Figure 2: ADMM Convergence Trajectory (Objective, RelChange, PSNR/SSIM)
    print("  Generating fig_stage1_admm_convergence.png...")
    set_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.7), facecolor="#FFFFFF")

    iters = hist_df["iteration"].values

    # Subplot (a): Objective function decay
    ax0 = axes[0]
    ax0.semilogy(iters, hist_df["objective"].values, color="#D32F2F", linewidth=1.2)
    ax0.set_title("(a) Objective Value $J(C^k, S^k)$", fontsize=8.0, fontweight="bold", pad=4)
    ax0.set_xlabel("ADMM Iteration $k$", fontsize=7.5)
    ax0.set_ylabel("Total Objective", fontsize=7.5)
    ax0.grid(True, linestyle=":", color="#CCCCCC", alpha=0.6)

    # Subplot (b): Relative change
    ax1 = axes[1]
    ax1.semilogy(iters, hist_df["rel_change"].values, color="#1976D2", linewidth=1.2)
    ax1.axhline(1e-4, color="#000000", linestyle="--", linewidth=0.8, label="Tolerance (1e-4)")
    ax1.set_title(r"(b) Relative Change $\|C^{k+1}-C^k\|_F / \|C^k\|_F$", fontsize=8.0, fontweight="bold", pad=4)
    ax1.set_xlabel("ADMM Iteration $k$", fontsize=7.5)
    ax1.set_ylabel("Relative Change", fontsize=7.5)
    ax1.legend(loc="upper right", fontsize=6.5, edgecolor="#000000")
    ax1.grid(True, linestyle=":", color="#CCCCCC", alpha=0.6)

    # Subplot (c): PSNR and SSIM trajectory
    ax2 = axes[2]
    ax2_twin = ax2.twinx()

    l1 = ax2.plot(iters, hist_df["psnr"].values, color="#2E7D32", linewidth=1.2, label="PSNR (dB)")
    l2 = ax2_twin.plot(iters, hist_df["ssim"].values, color="#E65100", linewidth=1.2, linestyle="-.", label="SSIM")

    ax2.set_title("(c) Restoration Trajectory", fontsize=8.0, fontweight="bold", pad=4)
    ax2.set_xlabel("ADMM Iteration $k$", fontsize=7.5)
    ax2.set_ylabel("PSNR (dB)", color="#2E7D32", fontsize=7.5)
    ax2_twin.set_ylabel("SSIM", color="#E65100", fontsize=7.5)
    ax2.tick_params(axis="y", labelcolor="#2E7D32")
    ax2_twin.tick_params(axis="y", labelcolor="#E65100")
    ax2.grid(True, linestyle=":", color="#CCCCCC", alpha=0.6)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="lower right", fontsize=6.5, edgecolor="#000000")

    for ax in [ax0, ax1, ax2, ax2_twin]:
        for sp in ax.spines.values():
            sp.set_edgecolor("#000000")
            sp.set_linewidth(0.8)

    plt.tight_layout()
    fig_path = str(out_dir / "fig_stage1_admm_convergence.png")
    fig.savefig(fig_path, dpi=300, facecolor="#FFFFFF", edgecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)
    print(f"  Exported convergence plot to: {fig_path}")


def main() -> None:
    print("======================================================================")
    print("  IEEE Sensors Journal - Module 2: Stage 1 ADMM Destriping Runner")
    print("======================================================================")

    # 1. Run batch ADMM solver across all scenes
    metrics_df, sol_dict = run_stage1_batch()

    # 2. Generate publication figures
    print("\n[Step 2/2] Generating Stage 1 publication figures...")
    generate_stage1_figures(sol_dict)

    # 3. Telemetry summary
    print("\n======================================================================")
    print("  Stage 1 Performance Summary across 6 Benchmark Scenes")
    print("======================================================================")
    print(metrics_df[["Scene", "Degraded_PSNR", "Stage1_PSNR", "PSNR_Gain_dB", "SSIM_Gain", "Iterations", "Elapsed_Sec"]].to_string(index=False))

    suite = BenchmarkSuite.get_global_suite()
    print("\nComputational Profiling:")
    print(suite.summary_table())
    print("\nModule 2 Stage 1 execution completed successfully.")


if __name__ == "__main__":
    main()
