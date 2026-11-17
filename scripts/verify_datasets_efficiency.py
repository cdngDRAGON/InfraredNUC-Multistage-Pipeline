#!/usr/bin/env python3
"""
scripts/verify_datasets_efficiency.py

Comprehensive cross-dataset evaluation across all 5 harvested real infrared groups:
  1. yohann_mire (Real uncooled FPN/stripe)
  2. dls_nuc_he (Real LWIR microbolometer stripes)
  3. tno (Real clean thermal surveillance/building)
  4. msrs_roadscene (Real clean thermal road/vehicle/pedestrian)
  5. m3fd_llvip (Real clean thermal high-detail street)

Generates:
  - results_verification/real_datasets_noref_metrics.csv
  - results_verification/simulated_datasets_fullref_metrics.csv
  - results_verification/native_resolution_runtime_table2.csv
  - results_verification/fig_real_datasets_3d_energy.png
  - results_verification/fig_simulated_datasets_3d_energy.png
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common.metrics import (
    evaluate_all_metrics,
    compute_roughness_rho,
    compute_icv,
    compute_avge,
    compute_aad,
    compute_mrd,
    compute_psnr,
    compute_ssim,
)
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank
from mod04_baselines_and_benchmarks.src.filter_baselines import solve_1d_gf, solve_1d_wls, solve_bm3d
from mod04_baselines_and_benchmarks.src.optim_baselines import solve_adom

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results_verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline(img: np.ndarray, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray, float]:
    """Execute Stage 1 + Stage 2 proposed pipeline."""
    t0 = time.perf_counter()
    c1, s_est, _ = solve_stage1_admm(img, maxiter=30)
    h, w = img.shape
    stride = 4 if max(h, w) <= 300 else 6
    c2 = denoise_stage2_patch_lowrank(
        c1,
        stride=stride,
        search_radius=12,
        num_similar_patches=50,
        n_jobs=n_jobs,
    )
    t_total = time.perf_counter() - t0
    return c1, c2, t_total


def evaluate_real_noref_datasets():
    print("\n[*] Evaluating Real Uncooled Degraded Datasets (No-Reference)...")
    manifest_p = PROJECT_ROOT / "data" / "dataset_manifest.csv"
    df = pd.read_csv(manifest_p)

    real_sub = df[df["dataset_group"].isin(["yohann_mire", "dls_nuc_he"])].copy()
    # Pick top 6 distinct real images
    sample_rows = real_sub.iloc[[0, 1, 6, 10, 12, 16]]

    results = []

    for _, row in sample_rows.iterrows():
        p = PROJECT_ROOT / row["saved_path"]
        img = np.array(Image.open(p).convert("L")).astype(np.float64)

        # Proposed pipeline
        c1, c2, t_ours = run_pipeline(img, n_jobs=-1)

        # Baseline: 1D-GF
        c_gf, _ = solve_1d_gf(img)
        # Baseline: 1D-WLS
        c_wls, _ = solve_1d_wls(img)

        # Measure roughness and ICV
        rho_in = compute_roughness_rho(img)
        rho_gf = compute_roughness_rho(c_gf)
        rho_wls = compute_roughness_rho(c_wls)
        rho_ours = compute_roughness_rho(c2)

        icv_in = compute_icv(img)
        icv_gf = compute_icv(c_gf)
        icv_wls = compute_icv(c_wls)
        icv_ours = compute_icv(c2)

        avge_ours = compute_avge(img, c2)
        aad_ours = compute_aad(img, c2)
        mrd_ours = compute_mrd(img, c2)

        results.append({
            "Dataset_Group": row["dataset_group"],
            "Image": p.name,
            "Resolution": f"{img.shape[1]}x{img.shape[0]}",
            "Input_Rho": round(rho_in, 4),
            "GF_Rho": round(rho_gf, 4),
            "WLS_Rho": round(rho_wls, 4),
            "Ours_Rho": round(rho_ours, 4),
            "Input_ICV": round(icv_in, 2),
            "GF_ICV": round(icv_gf, 2),
            "WLS_ICV": round(icv_wls, 2),
            "Ours_ICV": round(icv_ours, 2),
            "Ours_AVGE": round(avge_ours, 4),
            "Ours_AAD": round(aad_ours, 4),
            "Ours_MRD": round(mrd_ours, 4),
            "Runtime_s": round(t_ours, 3),
        })
        print(f"  - {p.name:20s}: Input Rho={rho_in:.4f} -> Ours Rho={rho_ours:.4f}, ICV={icv_ours:.1f}")

    res_df = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "real_datasets_noref_metrics.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"[✓] Saved real dataset no-ref metrics -> {out_csv}")


def evaluate_simulated_fullref_datasets():
    print("\n[*] Evaluating Real Clean Reference Datasets with Calibrated Noise (Full-Reference)...")
    manifest_p = PROJECT_ROOT / "data" / "dataset_manifest.csv"
    df = pd.read_csv(manifest_p)

    clean_sub = df[df["dataset_group"].isin(["tno", "msrs_roadscene", "m3fd_llvip"])].copy()
    sample_rows = clean_sub.iloc[[0, 2, 8, 12, 18, 22]]

    results = []

    for _, row in sample_rows.iterrows():
        p = PROJECT_ROOT / row["saved_path"]
        clean = np.array(Image.open(p).convert("L")).astype(np.float64)
        h, w = clean.shape

        # Simulate Case 2 periodic stripe + env noise
        np.random.seed(42)
        stripe_col = 14.0 * np.sin(2.0 * np.pi * 0.18 * np.arange(w)) + np.random.normal(0, 5, (w,))
        stripe = np.tile(stripe_col, (h, 1))
        noise = np.random.normal(0, 5, (h, w))
        degraded = np.clip(clean + stripe + noise, 0.0, 255.0)

        # Process
        c_gf, _ = solve_1d_gf(degraded)
        c_wls, _ = solve_1d_wls(degraded)
        c_adom, _ = solve_adom(degraded, max_iter=25)
        c1, c_ours, t_ours = run_pipeline(degraded, n_jobs=-1)

        m_deg = evaluate_all_metrics(clean, degraded, degraded=degraded)
        m_gf = evaluate_all_metrics(clean, c_gf, degraded=degraded)
        m_wls = evaluate_all_metrics(clean, c_wls, degraded=degraded)
        m_adom = evaluate_all_metrics(clean, c_adom, degraded=degraded)
        m_ours = evaluate_all_metrics(clean, c_ours, degraded=degraded)

        results.append({
            "Dataset_Group": row["dataset_group"],
            "Image": p.name,
            "Resolution": f"{w}x{h}",
            "Degraded_PSNR": round(m_deg["PSNR"], 2),
            "GF_PSNR": round(m_gf["PSNR"], 2),
            "WLS_PSNR": round(m_wls["PSNR"], 2),
            "ADOM_PSNR": round(m_adom["PSNR"], 2),
            "Ours_PSNR": round(m_ours["PSNR"], 2),
            "Degraded_SSIM": round(m_deg["SSIM"], 4),
            "GF_SSIM": round(m_gf["SSIM"], 4),
            "WLS_SSIM": round(m_wls["SSIM"], 4),
            "ADOM_SSIM": round(m_adom["SSIM"], 4),
            "Ours_SSIM": round(m_ours["SSIM"], 4),
            "Ours_AVGE": round(m_ours["AVGE"], 4),
            "Ours_AAD": round(m_ours["AAD"], 4),
            "Ours_MRD": round(m_ours["MRD"], 4),
            "Ours_Rho": round(m_ours["Roughness_rho"], 4),
            "Ours_ICV": round(m_ours["ICV"], 2),
        })
        print(f"  - {p.name:20s}: Degraded PSNR={m_deg['PSNR']:.2f} -> Ours={m_ours['PSNR']:.2f} dB, SSIM={m_ours['SSIM']:.4f}")

    res_df = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "simulated_datasets_fullref_metrics.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"[✓] Saved full-ref metrics -> {out_csv}")


def evaluate_native_resolution_scaling():
    print("\n[*] Evaluating Native-Resolution Computational Efficiency (Table II Scaling)...")
    manifest_p = PROJECT_ROOT / "data" / "dataset_manifest.csv"
    df = pd.read_csv(manifest_p)

    test_targets = [
        ("Yohann (320x220)", "yohann_mire", "test1_8bits.png"),
        ("1D-GF (384x269)", "yohann_mire", "test2_8bits.png"),
        ("DLS-NUC (384x288)", "dls_nuc_he", "input_01.png"),
        ("TNO (280x280)", "tno", "05.png"),
        ("TNO (620x450)", "tno", "17.png"),
        ("MSRS (640x480)", "msrs_roadscene", "00537D.png"),
        ("M3FD (1024x768)", "m3fd_llvip", "M3FD_00471.png"),
    ]

    records = []

    for name, group, fname in test_targets:
        match = df[(df["dataset_group"] == group) & (df["saved_path"].str.endswith(fname))]
        if match.empty:
            continue
        p = PROJECT_ROOT / match.iloc[0]["saved_path"]
        img = np.array(Image.open(p).convert("L")).astype(np.float64)
        h, w = img.shape
        stride = 4 if max(h, w) <= 300 else 6

        # 1. Single-Thread (n_jobs=1)
        t_s1_start = time.perf_counter()
        c1, _, _ = solve_stage1_admm(img, maxiter=25)
        t_s1_single = time.perf_counter() - t_s1_start

        t_s2_start = time.perf_counter()
        c2_single = denoise_stage2_patch_lowrank(c1, stride=stride, search_radius=12, num_similar_patches=50, n_jobs=1)
        t_s2_single = time.perf_counter() - t_s2_start
        t_total_single = t_s1_single + t_s2_single

        # 2. Parfor Multi-Core (n_jobs=-1)
        t_s2_par_start = time.perf_counter()
        c2_par = denoise_stage2_patch_lowrank(c1, stride=stride, search_radius=12, num_similar_patches=50, n_jobs=-1)
        t_s2_par = time.perf_counter() - t_s2_par_start
        t_total_par = t_s1_single + t_s2_par

        speedup = t_total_single / max(1e-6, t_total_par)
        s2_speedup = t_s2_single / max(1e-6, t_s2_par)

        records.append({
            "Dataset_Scenario": name,
            "Resolution": f"{w}x{h}",
            "Megapixels": round((w * h) / 1e6, 3),
            "Stage1_Time_s": round(t_s1_single, 3),
            "Stage2_Single_s": round(t_s2_single, 3),
            "Total_Single_s": round(t_total_single, 3),
            "Stage2_Parfor_s": round(t_s2_par, 3),
            "Total_Parfor_s": round(t_total_par, 3),
            "Stage2_Speedup": round(s2_speedup, 2),
            "Total_Speedup": round(speedup, 2),
        })
        print(f"  - {name:22s} [{w}x{h}]: Single={t_total_single:.2f}s | Parfor={t_total_par:.2f}s (Speedup={speedup:.2f}x)")

    df_runtime = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "native_resolution_runtime_table2.csv"
    df_runtime.to_csv(out_csv, index=False)
    print(f"[✓] Saved native resolution scaling -> {out_csv}")


def generate_visual_figures():
    print("\n[*] Generating 300-DPI Publication Figures with 3D Global Energy Profiles...")
    
    # 1. Real degraded visual figure
    real_path = PROJECT_ROOT / "data" / "real_degraded" / "yohann_mire" / "test1_8bits.png"
    img_real = np.array(Image.open(real_path).convert("L")).astype(np.float64)
    c1, c2, _ = run_pipeline(img_real, n_jobs=-1)

    fig = plt.figure(figsize=(12, 6), dpi=300, facecolor="#FFFFFF")
    
    # Row 1: 2D Images
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_real, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("(a) Real Degraded (Yohann)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(c1, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("(b) Stage 1 (ADMM)", fontsize=10, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(c2, cmap="gray", vmin=0, vmax=255)
    ax3.set_title("(c) Stage 2 (Ours Restored)", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # Row 2: 3D Energy Surfaces
    step = 4
    Y, X = np.mgrid[0:img_real.shape[0]:step, 0:img_real.shape[1]:step]

    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    ax4.plot_surface(X, Y, img_real[::step, ::step], cmap="jet", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax4.set_title("3D Energy: Raw Stripes", fontsize=9)
    ax4.view_init(elev=55, azim=-60)
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    ax5.plot_surface(X, Y, c1[::step, ::step], cmap="jet", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax5.set_title("3D Energy: Stage 1 Destriped", fontsize=9)
    ax5.view_init(elev=55, azim=-60)
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    ax6.plot_surface(X, Y, c2[::step, ::step], cmap="jet", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax6.set_title("3D Energy: Clean Manifold", fontsize=9)
    ax6.view_init(elev=55, azim=-60)
    ax6.axis("off")

    plt.tight_layout()
    out_real_fig = OUTPUT_DIR / "fig_real_datasets_3d_energy.png"
    plt.savefig(out_real_fig, dpi=300, facecolor="#FFFFFF")
    plt.close()
    print(f"[✓] Saved real dataset 3D energy figure -> {out_real_fig}")

    # 2. Simulated full-ref visual figure
    clean_path = PROJECT_ROOT / "data" / "clean_reference" / "msrs_roadscene" / "00537D.png"
    clean_img = np.array(Image.open(clean_path).convert("L")).astype(np.float64)
    h, w = clean_img.shape
    np.random.seed(42)
    stripe = np.tile(16.0 * np.sin(2 * np.pi * 0.18 * np.arange(w)), (h, 1))
    noise = np.random.normal(0, 6, (h, w))
    deg_img = np.clip(clean_img + stripe + noise, 0.0, 255.0)
    c1_s, c2_s, _ = run_pipeline(deg_img, n_jobs=-1)

    fig2 = plt.figure(figsize=(12, 6), dpi=300, facecolor="#FFFFFF")

    ax1 = fig2.add_subplot(2, 3, 1)
    ax1.imshow(deg_img, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("(a) Degraded Input (MSRS)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig2.add_subplot(2, 3, 2)
    ax2.imshow(c1_s, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("(b) Stage 1 Destriped", fontsize=10, fontweight="bold")
    ax2.axis("off")

    ax3 = fig2.add_subplot(2, 3, 3)
    ax3.imshow(c2_s, cmap="gray", vmin=0, vmax=255)
    ax3.set_title("(c) Stage 2 Restored (Ours)", fontsize=10, fontweight="bold")
    ax3.axis("off")

    step = 8
    Y, X = np.mgrid[0:h:step, 0:w:step]

    ax4 = fig2.add_subplot(2, 3, 4, projection="3d")
    ax4.plot_surface(X, Y, deg_img[::step, ::step], cmap="jet", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax4.set_title("3D Energy: Periodic Stripes", fontsize=9)
    ax4.view_init(elev=55, azim=-60)
    ax4.axis("off")

    ax5 = fig2.add_subplot(2, 3, 5, projection="3d")
    ax5.plot_surface(X, Y, c1_s[::step, ::step], cmap="jet", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax5.set_title("3D Energy: Stage 1", fontsize=9)
    ax5.view_init(elev=55, azim=-60)
    ax5.axis("off")

    ax6 = fig2.add_subplot(2, 3, 6, projection="3d")
    ax6.plot_surface(X, Y, c2_s[::step, ::step], cmap="jet", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax6.set_title("3D Energy: Final Clean Surface", fontsize=9)
    ax6.view_init(elev=55, azim=-60)
    ax6.axis("off")

    plt.tight_layout()
    out_sim_fig = OUTPUT_DIR / "fig_simulated_datasets_3d_energy.png"
    plt.savefig(out_sim_fig, dpi=300, facecolor="#FFFFFF")
    plt.close()
    print(f"[✓] Saved simulated dataset 3D energy figure -> {out_sim_fig}")


def main():
    print("================================================================================")
    print("  CROSS-DATASET VERIFICATION & NATIVE RESOLUTION BENCHMARKING")
    print("================================================================================")
    evaluate_real_noref_datasets()
    evaluate_simulated_fullref_datasets()
    evaluate_native_resolution_scaling()
    generate_visual_figures()
    print("\n================================================================================")
    print("  VERIFICATION COMPLETE: ALL DATASET GROUPS TESTED & BENCHMARKED")
    print("================================================================================")


if __name__ == "__main__":
    main()
