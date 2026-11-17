"""mod03_stage2_patch_lowrank/src/interactive_mod03.py

Interactive Stage 2 Patch Low-Rank SVD & Group Size Explorer.
Secondary CLI layer for interactive exploration and viva defense demonstrations.

Allows users to:
  1. Select an infrared scene (or custom degraded image).
  2. Vary patch group count L in {10..80} and noise energy parameter gamma / tau.
  3. Observe live adaptive SVD rank truncation (k) and Kaiser-windowed back-projection.
  4. Compare Stage 1 vs. Stage 2 metrics (PSNR, SSIM, Roughness rho, latency) in real time.

Supports both interactive input() mode and automated non-blocking verification via `--demo`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tabulate import tabulate

from common.metrics import compute_avge, compute_psnr, compute_roughness_rho, compute_ssim
from common.plot_style import set_ieee_style
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank

SCENE_CHOICES = [
    ("1", "Img1_Checkerboard_Yohann_case1.png", "Img1_Checkerboard_Yohann.png", "Checkerboard Yohann (Case 1)"),
    ("2", "Img2_IndoorMonitor_Yohann_case1.png", "Img2_IndoorMonitor_Yohann.png", "Indoor Monitor Yohann (Case 1)"),
    ("3", "Img3_BuildingSkyline_He_case1.png", "Img3_BuildingSkyline_He.png", "Building Skyline He (Case 1)"),
    ("4", "Img4_HousePedestrian_He_case1.png", "Img4_HousePedestrian_He.png", "House & Pedestrian He (Case 1)"),
    ("5", "Img5_FacadeWindows_TNO_case1.png", "Img5_FacadeWindows_TNO.png", "Facade Windows TNO (Case 1)"),
    ("6", "Img6_StreetCars_LLVIP_case1.png", "Img6_StreetCars_LLVIP.png", "Street Cars LLVIP (Case 1)"),
]


def load_scene_pair(
    workspace_root: Path,
    scene_idx: str = "1",
    custom_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Load degraded and clean reference scenes."""
    if custom_path:
        p = Path(custom_path)
        if not p.exists():
            raise FileNotFoundError(f"Custom image not found: {custom_path}")
        return np.array(Image.open(p).convert("L"), dtype=np.float64), None, p.stem

    deg_dir = workspace_root / "data" / "degraded_scenes"
    clean_dir = workspace_root / "data" / "clean_scenes"

    for key, deg_fname, clean_fname, desc in SCENE_CHOICES:
        if scene_idx.strip() == key or scene_idx.strip().lower() in deg_fname.lower():
            p_deg = deg_dir / deg_fname
            p_clean = clean_dir / clean_fname
            if p_deg.exists():
                deg = np.array(Image.open(p_deg).convert("L"), dtype=np.float64)
                clean = np.array(Image.open(p_clean).convert("L"), dtype=np.float64) if p_clean.exists() else None
                return deg, clean, desc

    # Fallback
    p_deg = deg_dir / SCENE_CHOICES[0][1]
    p_clean = clean_dir / SCENE_CHOICES[0][2]
    if p_deg.exists():
        deg = np.array(Image.open(p_deg).convert("L"), dtype=np.float64)
        clean = np.array(Image.open(p_clean).convert("L"), dtype=np.float64) if p_clean.exists() else None
        return deg, clean, SCENE_CHOICES[0][3]

    clean = np.random.uniform(50.0, 200.0, (256, 256))
    deg = np.clip(clean + np.random.normal(0, 15, (256, 256)), 0, 255)
    return deg, clean, "Synthetic Scene"


def run_mod03_interactive(
    scene_choice: str = "1",
    custom_path: Optional[str] = None,
    num_similar_patches_L: int = 50,
    gamma_tau: float = 0.95,
    stride: int = 4,
    save_fig: bool = True,
    interactive: bool = True,
) -> int:
    """Execute interactive Module 3 Stage 2 Low-Rank Patch exploration."""
    set_ieee_style()
    workspace_root = Path(__file__).resolve().parent.parent.parent
    results_dir = workspace_root / "mod03_stage2_patch_lowrank" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  MODULE 3: STAGE 2 PATCH LOW-RANK SVD & GROUP SIZE EXPLORER")
    print("=" * 80)

    if interactive:
        print("\nAvailable Benchmark Scenes:")
        for key, _, _, desc in SCENE_CHOICES:
            print(f"  [{key}] {desc}")
        print("  [C] Custom Image Path")

        u_sc = input(f"\nSelect scene [1-6, or C, default {scene_choice}]: ").strip()
        if u_sc.upper() == "C":
            cp = input("Enter path to custom image: ").strip()
            if cp:
                custom_path = cp
        elif u_sc in ["1", "2", "3", "4", "5", "6"]:
            scene_choice = u_sc

        l_in = input(f"Enter patch count L in [10..80] (Eq. 22 top similar patches) [default {num_similar_patches_L}]: ").strip()
        if l_in:
            try:
                num_similar_patches_L = max(10, min(80, int(l_in)))
            except ValueError:
                pass

        g_in = input(f"Enter noise threshold factor gamma in [0.5..2.0] (Eq. 26) [default {gamma_tau:.2f}]: ").strip()
        if g_in:
            try:
                gamma_tau = max(0.2, min(3.0, float(g_in)))
            except ValueError:
                pass

        s_in = input(f"Enter sliding patch stride s in [2..8] [default {stride}]: ").strip()
        if s_in:
            try:
                stride = max(2, min(8, int(s_in)))
            except ValueError:
                pass

    deg_img, clean_img, scene_desc = load_scene_pair(workspace_root, scene_choice, custom_path)
    print(f"\n[*] Loaded Scene: '{scene_desc}' ({deg_img.shape[0]}x{deg_img.shape[1]})")

    # Step 1: Run fast Stage 1 ADMM destriping
    print("[*] Executing Stage 1 ADMM Destriping (K=25 iterations)...")
    t0 = time.perf_counter()
    c_s1, s_est, _ = solve_stage1_admm(deg_img, maxiter=25, verbose=False)
    t_s1 = time.perf_counter() - t0
    print(f"    Stage 1 Complete in {t_s1:.2f} s.")

    # Step 2: Run Stage 2 Patch Low-Rank SVD
    print(f"[*] Executing Stage 2 Patch Low-Rank SVD (L={num_similar_patches_L}, gamma={gamma_tau:.2f}, stride={stride})...")
    t0 = time.perf_counter()
    c_s2, diag = denoise_stage2_patch_lowrank(
        c_s1,
        patch_size=11,
        stride=stride,
        search_radius=12,
        num_similar_patches=num_similar_patches_L,
        gamma=gamma_tau,
        n_jobs=-1,
        return_diagnostics=True,
    )
    t_s2 = time.perf_counter() - t0
    print(f"    Stage 2 Complete in {t_s2:.2f} s.")

    # Compute comparative metrics
    has_ref = clean_img is not None
    psnr_deg = compute_psnr(deg_img, clean_img) if has_ref else float("nan")
    psnr_s1 = compute_psnr(c_s1, clean_img) if has_ref else float("nan")
    psnr_s2 = compute_psnr(c_s2, clean_img) if has_ref else float("nan")

    ssim_deg = compute_ssim(deg_img, clean_img) if has_ref else float("nan")
    ssim_s1 = compute_ssim(c_s1, clean_img) if has_ref else float("nan")
    ssim_s2 = compute_ssim(c_s2, clean_img) if has_ref else float("nan")

    rho_deg = compute_roughness_rho(deg_img)
    rho_s1 = compute_roughness_rho(c_s1)
    rho_s2 = compute_roughness_rho(c_s2)

    avge_deg = compute_avge(deg_img, clean_img) if has_ref else float("nan")
    avge_s1 = compute_avge(c_s1, clean_img) if has_ref else float("nan")
    avge_s2 = compute_avge(c_s2, clean_img) if has_ref else float("nan")

    avg_k = float(diag.get("mean_rank", 0.0))
    n_groups = int(diag.get("total_ref_patches", 0))

    comp_table = [
        ["Degraded Input", f"{psnr_deg:.2f} dB", f"{ssim_deg:.4f}", f"{rho_deg:.4f}", f"{avge_deg:.4f}", "-"],
        ["Stage 1 Alone (w/o Stage 2)", f"{psnr_s1:.2f} dB", f"{ssim_s1:.4f}", f"{rho_s1:.4f}", f"{avge_s1:.4f}", f"{t_s1:.2f} s"],
        [f"Stage 2 Restored (L={num_similar_patches_L})", f"{psnr_s2:.2f} dB", f"{ssim_s2:.4f}", f"{rho_s2:.4f}", f"{avge_s2:.4f}", f"{t_s2:.2f} s"],
        ["Stage 2 Incremental Gain", f"+{psnr_s2 - psnr_s1:.2f} dB", f"+{ssim_s2 - ssim_s1:.4f}", f"{rho_s2 - rho_s1:+.4f}", f"{avge_s2 - avge_s1:+.4f}", f"Avg rank k={avg_k:.1f}"],
    ]

    print("\n" + "-" * 75)
    print("  STAGE 1 VS. STAGE 2 MULTI-METRIC PERFORMANCE EVALUATION")
    print("-" * 75)
    print(tabulate(
        comp_table,
        headers=["Pipeline Stage", "PSNR", "SSIM", "Roughness ρ", "AVGE", "Latency / Complexity"],
        tablefmt="fancy_grid",
    ))

    diag_table = [
        ["Total Reference Patch Groups", f"{n_groups} clusters", f"Stride s={stride}"],
        ["Observation Matrix G Shape", f"121 x {num_similar_patches_L}", f"d=11x11, L={num_similar_patches_L}"],
        ["Average Truncation Rank k", f"{avg_k:.2f} / {min(121, num_similar_patches_L)}", "Governed by Eq. 26 noise energy bound"],
        ["Stage 2 Parallel Throughput", f"{n_groups / (t_s2 + 1e-6):.1f} clusters/s", "Multi-threaded worker pool"],
    ]

    print("\n" + "-" * 75)
    print("  STAGE 2 SVD LOW-RANK OBSERVATION MATRIX DIAGNOSTICS")
    print("-" * 75)
    print(tabulate(diag_table, headers=["Diagnostic Metric", "Value", "Technical Description"], tablefmt="fancy_grid"))

    if save_fig:
        out_fig = results_dir / f"interactive_mod03_{scene_desc.replace(' ', '_')}_L{num_similar_patches_L}.png"
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

        axes[0].imshow(deg_img, cmap="gray")
        axes[0].set_title(f"Degraded Input\nPSNR: {psnr_deg:.2f} dB", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(c_s1, cmap="gray")
        axes[1].set_title(f"Stage 1 (w/o Stage 2)\nPSNR: {psnr_s1:.2f} dB", fontsize=9, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(c_s2, cmap="gray")
        axes[2].set_title(f"Stage 2 (L={num_similar_patches_L})\nPSNR: {psnr_s2:.2f} dB (+{psnr_s2 - psnr_s1:.2f} dB)", fontsize=9, fontweight="bold")
        axes[2].axis("off")

        # Difference map showing environmental noise removed by Stage 2
        diff = np.abs(c_s1 - c_s2)
        im4 = axes[3].imshow(diff, cmap="magma")
        axes[3].set_title(f"Environmental Noise Removed\nAvg rank k={avg_k:.1f}", fontsize=9, fontweight="bold")
        axes[3].axis("off")
        plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(out_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[+] Interactive Stage 2 Figure Saved -> {out_fig}")

    print("\n[+] Module 3 Interactive Patch Low-Rank Exploration Complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Module 3 Patch Low-Rank Explorer")
    parser.add_argument("--demo", action="store_true", help="Run non-blocking automated demo verification")
    parser.add_argument("--scene", type=str, default="1", help="Scene number 1..6")
    parser.add_argument("--custom", type=str, default=None, help="Path to custom image")
    parser.add_argument("--L", type=int, default=40, help="Patch count L (default 40 for demo)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Noise energy factor gamma")
    parser.add_argument("--stride", type=int, default=6, help="Sliding stride (default 6 for fast demo)")
    args = parser.parse_args()

    return run_mod03_interactive(
        scene_choice=args.scene,
        custom_path=args.custom,
        num_similar_patches_L=args.L,
        gamma_tau=args.gamma,
        stride=args.stride,
        save_fig=True,
        interactive=not args.demo,
    )


if __name__ == "__main__":
    sys.exit(main())
