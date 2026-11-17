"""mod02_stage1_admm_destriping/src/interactive_mod02.py

Interactive Stage 1 ADMM Destriping Hyperparameter Tuner.
Secondary CLI layer for interactive exploration and viva defense demonstrations.

Allows users to:
  1. Pick a degraded scene (or load custom image).
  2. Live-tune ADMM hyperparameters: lambda1 (Schatten 1/2), lambda2 (TV_1), lambda3 (d2_dx2), maxiter K.
  3. Watch primal-dual ADMM convergence progress table (k, relative change, PSNR, SSIM, latency).
  4. Compare baseline degraded vs. Stage 1 restored fidelity.

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

from common.metrics import compute_psnr, compute_ssim
from common.plot_style import set_ieee_style
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm

SCENE_CHOICES = [
    ("1", "Img1_Checkerboard_Yohann_case1.png", "Img1_Checkerboard_Yohann.png", "Checkerboard Yohann (Case 1)"),
    ("2", "Img2_IndoorMonitor_Yohann_case1.png", "Img2_IndoorMonitor_Yohann.png", "Indoor Monitor Yohann (Case 1)"),
    ("3", "Img3_BuildingSkyline_He_case1.png", "Img3_BuildingSkyline_He.png", "Building Skyline He (Case 1)"),
    ("4", "Img4_HousePedestrian_He_case1.png", "Img4_HousePedestrian_He.png", "House & Pedestrian He (Case 1)"),
    ("5", "Img5_FacadeWindows_TNO_case1.png", "Img5_FacadeWindows_TNO.png", "Facade Windows TNO (Case 1)"),
    ("6", "Img6_StreetCars_LLVIP_case1.png", "Img6_StreetCars_LLVIP.png", "Street Cars LLVIP (Case 1)"),
]


def load_images(
    workspace_root: Path,
    scene_idx: str = "1",
    custom_deg: Optional[str] = None,
    custom_clean: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Load degraded and clean reference images."""
    if custom_deg:
        p_deg = Path(custom_deg)
        if not p_deg.exists():
            raise FileNotFoundError(f"Custom degraded image not found: {custom_deg}")
        deg = np.array(Image.open(p_deg).convert("L"), dtype=np.float64)
        clean = None
        if custom_clean and Path(custom_clean).exists():
            clean = np.array(Image.open(custom_clean).convert("L"), dtype=np.float64)
        return deg, clean, p_deg.stem

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

    # Fallback default
    p_deg = deg_dir / SCENE_CHOICES[0][1]
    p_clean = clean_dir / SCENE_CHOICES[0][2]
    if p_deg.exists():
        deg = np.array(Image.open(p_deg).convert("L"), dtype=np.float64)
        clean = np.array(Image.open(p_clean).convert("L"), dtype=np.float64) if p_clean.exists() else None
        return deg, clean, SCENE_CHOICES[0][3]

    # Synthetic fallback
    clean = np.random.uniform(50.0, 200.0, (256, 256))
    stripe = np.tile(np.random.normal(0, 20, 256), (256, 1))
    deg = np.clip(clean + stripe + np.random.normal(0, 5, (256, 256)), 0, 255)
    return deg, clean, "Synthetic Scene"


def run_mod02_interactive(
    scene_choice: str = "1",
    custom_deg: Optional[str] = None,
    custom_clean: Optional[str] = None,
    lambda1: float = 0.5,
    lambda2: float = 0.05,
    lambda3: float = 2.0,
    maxiter: int = 40,
    save_fig: bool = True,
    interactive: bool = True,
) -> int:
    """Execute interactive Stage 1 ADMM destriping."""
    set_ieee_style()
    workspace_root = Path(__file__).resolve().parent.parent.parent
    results_dir = workspace_root / "mod02_stage1_admm_destriping" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  MODULE 2: STAGE 1 ADMM DESTRIPING HYPERPARAMETER TUNER")
    print("=" * 80)

    if interactive:
        print("\nAvailable Benchmark Scenes:")
        for key, _, _, desc in SCENE_CHOICES:
            print(f"  [{key}] {desc}")
        print("  [C] Custom Image Path")

        u_sc = input(f"\nSelect scene [1-6, or C, default {scene_choice}]: ").strip()
        if u_sc.upper() == "C":
            cd = input("Enter path to degraded image: ").strip()
            if cd:
                custom_deg = cd
                cc = input("Enter path to clean reference image (optional, press Enter to skip): ").strip()
                if cc:
                    custom_clean = cc
        elif u_sc in ["1", "2", "3", "4", "5", "6"]:
            scene_choice = u_sc

        l1_in = input(f"Enter lambda1 (Schatten 1/2 low-rank on S) [default {lambda1}]: ").strip()
        if l1_in:
            try:
                lambda1 = float(l1_in)
            except ValueError:
                pass

        l2_in = input(f"Enter lambda2 (First-order TV ||grad(C)||_1) [default {lambda2}]: ").strip()
        if l2_in:
            try:
                lambda2 = float(l2_in)
            except ValueError:
                pass

        l3_in = input(f"Enter lambda3 (Second-order horiz TV ||d2_dx2(C)||_1) [default {lambda3}]: ").strip()
        if l3_in:
            try:
                lambda3 = float(l3_in)
            except ValueError:
                pass

        k_in = input(f"Enter maxiter K (ADMM iterations, 10..250) [default {maxiter}]: ").strip()
        if k_in:
            try:
                maxiter = int(k_in)
            except ValueError:
                pass

    deg_img, clean_img, scene_desc = load_images(workspace_root, scene_choice, custom_deg, custom_clean)
    H, W = deg_img.shape
    print(f"\n[*] Loaded Scene: '{scene_desc}' ({H}x{W})")

    # Initial metrics
    init_psnr = compute_psnr(deg_img, clean_img) if clean_img is not None else float("nan")
    init_ssim = compute_ssim(deg_img, clean_img) if clean_img is not None else float("nan")

    print("\n" + "-" * 75)
    print(f"  RUNNING ADMM SOLVER: K={maxiter}, λ1={lambda1}, λ2={lambda2}, λ3={lambda3}")
    print("-" * 75)
    if clean_img is not None:
        print(f"  Initial Degraded Fidelity: PSNR = {init_psnr:.2f} dB, SSIM = {init_ssim:.4f}")

    t0 = time.perf_counter()
    c_s1, s_est, history_df = solve_stage1_admm(
        deg_img,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        rho=0.5,
        maxiter=maxiter,
        tol=1e-4,
        clean_ref=clean_img,
        verbose=False,
    )
    total_time = time.perf_counter() - t0

    # Display convergence trajectory table
    n_rows = len(history_df)
    if n_rows > 12:
        # Sample steps: first 5, middle 2, last 5
        sample_indices = sorted(list(set(
            list(range(min(5, n_rows))) +
            [n_rows // 2, n_rows // 2 + 1] +
            list(range(max(0, n_rows - 5), n_rows))
        )))
        sample_df = history_df.iloc[sample_indices]
    else:
        sample_df = history_df

    tbl_data = []
    for _, r in sample_df.iterrows():
        it = int(r.get("iteration", 0))
        rc = float(r.get("rel_change", 0.0))
        psnr_val = f"{float(r['psnr']):.2f} dB" if "psnr" in r and not np.isnan(r["psnr"]) else "N/A"
        ssim_val = f"{float(r['ssim']):.4f}" if "ssim" in r and not np.isnan(r["ssim"]) else "N/A"
        t_sec = f"{float(r.get('time_s', 0.0)):.2f} s"
        tbl_data.append([f"Iter {it:03d}", f"{rc:.3e}", psnr_val, ssim_val, t_sec])

    print("\n" + tabulate(
        tbl_data,
        headers=["Iteration", "Primal Rel Change ||C^{k+1} - C^k|| / ||C^k||", "PSNR", "SSIM", "Wall-Clock Time"],
        tablefmt="fancy_grid",
    ))

    # Final summary
    final_psnr = compute_psnr(c_s1, clean_img) if clean_img is not None else float("nan")
    final_ssim = compute_ssim(c_s1, clean_img) if clean_img is not None else float("nan")

    delta_psnr = final_psnr - init_psnr if clean_img is not None else float("nan")
    delta_ssim = final_ssim - init_ssim if clean_img is not None else float("nan")

    summary_tbl = [
        ["Initial Degraded", f"{init_psnr:.2f} dB", f"{init_ssim:.4f}", "-"],
        ["Stage 1 Restored C", f"{final_psnr:.2f} dB", f"{final_ssim:.4f}", f"{total_time:.2f} s"],
        ["Improvement (Gain)", f"+{delta_psnr:.2f} dB", f"+{delta_ssim:.4f}", f"{n_rows / total_time:.1f} iters/s"],
    ]

    print("\n" + "-" * 75)
    print("  STAGE 1 ADMM RESTORATION SUMMARY")
    print("-" * 75)
    print(tabulate(summary_tbl, headers=["State", "PSNR", "SSIM", "Performance"], tablefmt="fancy_grid"))

    if save_fig:
        out_path = results_dir / f"interactive_mod02_{scene_desc.replace(' ', '_')}.png"
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=300)
        axes[0].imshow(deg_img, cmap="gray")
        axes[0].set_title(f"Degraded Input\nPSNR: {init_psnr:.2f} dB", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(c_s1, cmap="gray")
        axes[1].set_title(f"Stage 1 Restored C (K={maxiter})\nPSNR: {final_psnr:.2f} dB (+{delta_psnr:.2f} dB)", fontsize=9, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(s_est, cmap="gray")
        axes[2].set_title(f"Separated Stripe S\nλ1={lambda1}, λ2={lambda2}, λ3={lambda3}", fontsize=9, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[+] Interactive Stage 1 Figure Saved -> {out_path}")

    print("\n[+] Module 2 Interactive Hyperparameter Tuning Complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Module 2 ADMM Destriping Tuner")
    parser.add_argument("--demo", action="store_true", help="Run non-blocking automated demo verification")
    parser.add_argument("--scene", type=str, default="1", help="Scene number 1..6")
    parser.add_argument("--custom_deg", type=str, default=None, help="Path to custom degraded image")
    parser.add_argument("--custom_clean", type=str, default=None, help="Path to clean reference image")
    parser.add_argument("--lambda1", type=float, default=0.5, help="Schatten 1/2 penalty weight")
    parser.add_argument("--lambda2", type=float, default=0.05, help="First-order TV penalty weight")
    parser.add_argument("--lambda3", type=float, default=2.0, help="Second-order directional TV penalty weight")
    parser.add_argument("--maxiter", type=int, default=15, help="ADMM iterations (default 15 for fast demo)")
    args = parser.parse_args()

    return run_mod02_interactive(
        scene_choice=args.scene,
        custom_deg=args.custom_deg,
        custom_clean=args.custom_clean,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        maxiter=args.maxiter,
        save_fig=True,
        interactive=not args.demo,
    )


if __name__ == "__main__":
    sys.exit(main())
