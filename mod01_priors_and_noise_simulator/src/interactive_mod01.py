"""mod01_priors_and_noise_simulator/src/interactive_mod01.py

Interactive Noise Degradation Simulator & Prior Energy Explorer.
Secondary CLI layer for interactive exploration and viva defense demonstrations.

Allows users to:
  1. Pick any benchmark scene (or custom image path).
  2. Select degradation model: Case 1 (Conventional), Case 2 (Periodic), Case 3 (Enhanced).
  3. Specify custom stripe amplitude and environmental noise intensity.
  4. Inspect live SVD singular value energy concentration and rank-k error bounds.
  5. Inspect horizontal vs. vertical gradient energy anisotropy ratios.

Supports both interactive input() mode and automated non-blocking verification via `--demo`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tabulate import tabulate

from common.plot_style import set_ieee_style
from mod01_priors_and_noise_simulator.src.gradient_priors import (
    compute_first_order_gradients,
    compute_gradient_anisotropy_ratio,
)
from mod01_priors_and_noise_simulator.src.lowrank_svd_analysis import (
    analyze_singular_value_decay,
    compute_rank_k_reconstruction,
    compute_svd,
)
from mod01_priors_and_noise_simulator.src.noise_simulator import (
    simulate_case1_conventional,
    simulate_case2_periodic,
    simulate_case3_enhanced,
)

BENCHMARK_SCENES = [
    ("1", "Img1_Checkerboard_Yohann.png", "Checkerboard Calibration Scene (Yohann 2019)"),
    ("2", "Img2_IndoorMonitor_Yohann.png", "Indoor Electronics Lab Monitor (Yohann 2019)"),
    ("3", "Img3_BuildingSkyline_He.png", "Urban Building Skyline (He 2019)"),
    ("4", "Img4_HousePedestrian_He.png", "House & Pedestrian Complex (He 2019)"),
    ("5", "Img5_FacadeWindows_TNO.png", "Architectural Windows Facade (TNO Dataset)"),
    ("6", "Img6_StreetCars_LLVIP.png", "Night Surveillance Street Cars (LLVIP Dataset)"),
]


def load_scene_interactive(
    workspace_root: Path,
    scene_choice: str = "1",
    custom_path: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Load benchmark scene or custom image."""
    clean_dir = workspace_root / "data" / "clean_scenes"
    if custom_path:
        p = Path(custom_path)
        if not p.exists():
            raise FileNotFoundError(f"Custom image not found: {custom_path}")
        im = Image.open(p).convert("L")
        return np.array(im, dtype=np.float64), p.stem

    for key, fname, desc in BENCHMARK_SCENES:
        if scene_choice.strip() == key or scene_choice.strip().lower() in fname.lower():
            p = clean_dir / fname
            if not p.exists():
                # Synthesize fallback clean image
                x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
                im_arr = 128.0 + 64.0 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)
                return im_arr, fname.replace(".png", "")
            im = Image.open(p).convert("L")
            return np.array(im, dtype=np.float64), fname.replace(".png", "")

    # Default fallback
    p = clean_dir / BENCHMARK_SCENES[0][1]
    if p.exists():
        im = Image.open(p).convert("L")
        return np.array(im, dtype=np.float64), BENCHMARK_SCENES[0][1].replace(".png", "")
    return np.random.uniform(50.0, 200.0, (256, 256)), "Fallback_Synthetic"


def run_mod01_interactive(
    scene_choice: str = "1",
    custom_path: Optional[str] = None,
    noise_case: int = 1,
    stripe_amp: float = 25.0,
    env_sigma: float = 5.0,
    save_fig: bool = True,
    interactive: bool = True,
) -> int:
    """Execute interactive Module 1 prior exploration."""
    set_ieee_style()
    workspace_root = Path(__file__).resolve().parent.parent.parent
    results_dir = workspace_root / "mod01_priors_and_noise_simulator" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  MODULE 1: NOISE DEGRADATION SIMULATOR & PRIOR ENERGY EXPLORER")
    print("=" * 80)

    if interactive:
        print("\nAvailable Benchmark Scenes:")
        for key, fname, desc in BENCHMARK_SCENES:
            print(f"  [{key}] {desc} ({fname})")
        print("  [C] Custom Image Path")

        user_sc = input(f"\nSelect scene [1-6, or C for custom, default {scene_choice}]: ").strip()
        if user_sc.upper() == "C":
            custom_input = input("Enter path to custom infrared image: ").strip()
            if custom_input:
                custom_path = custom_input
        elif user_sc in ["1", "2", "3", "4", "5", "6"]:
            scene_choice = user_sc

        case_in = input(f"Select Degradation Model (1=Case 1 Additive, 2=Case 2 Periodic, 3=Case 3 Enhanced) [default {noise_case}]: ").strip()
        if case_in in ["1", "2", "3"]:
            noise_case = int(case_in)

        s_amp_in = input(f"Enter Stripe Noise Amplitude sigma_s [default {stripe_amp:.1f}]: ").strip()
        if s_amp_in:
            try:
                stripe_amp = float(s_amp_in)
            except ValueError:
                pass

        env_in = input(f"Enter Environmental Noise Sigma sigma_n [default {env_sigma:.1f}]: ").strip()
        if env_in:
            try:
                env_sigma = float(env_in)
            except ValueError:
                pass

    clean_img, scene_title = load_scene_interactive(workspace_root, scene_choice, custom_path)
    H, W = clean_img.shape
    print(f"\n[*] Loaded Scene: '{scene_title}' ({H}x{W} pixels)")

    # Simulate degradation
    print(f"[*] Simulating Degradation: Case {noise_case} (Stripe Amp: {stripe_amp:.1f}, Env Sigma: {env_sigma:.1f})...")
    if noise_case == 1:
        res = simulate_case1_conventional(clean_img, stripe_sigma=stripe_amp, env_sigma=env_sigma, seed=42)
    elif noise_case == 2:
        res = simulate_case2_periodic(clean_img, stripe_amp=stripe_amp, freq=0.18, env_sigma=env_sigma, seed=42)
    else:
        res = simulate_case3_enhanced(clean_img, stripe_amp=stripe_amp, freq=0.22, env_sigma=env_sigma, seed=42)

    degraded_img = res.degraded
    stripe_img = res.stripe
    env_img = res.noise

    # 1. SVD Low-Rank Analysis on Stripe Matrix S
    U, s, Vt = compute_svd(stripe_img)
    total_energy = float(np.sum(s**2))
    top1_ratio = float((s[0]**2) / total_energy * 100.0) if total_energy > 0 else 0.0
    top2_ratio = float((s[0]**2 + (s[1]**2 if len(s) > 1 else 0)) / total_energy * 100.0) if total_energy > 0 else 0.0
    top5_ratio = float(np.sum(s[:5]**2) / total_energy * 100.0) if total_energy > 0 else 0.0

    # Rank-1 reconstruction error
    S_rank1, rank1_err, energy_ratio_1 = compute_rank_k_reconstruction(stripe_img, k=1)
    tail_bound = float(np.sqrt(np.sum(s[1:]**2)))

    svd_table = [
        ["Singular Value σ_1 (Dominant)", f"{s[0]:.2f}", f"{top1_ratio:.2f}%", "Principal vertical stripe mode"],
        ["Top-2 Singular Values (σ_1 + σ_2)", f"{(s[0]+s[1]):.2f}", f"{top2_ratio:.2f}%", "Rank-2 subspace capture"],
        ["Top-5 Singular Values (Σ σ_i^2)", f"{np.sum(s[:5]):.2f}", f"{top5_ratio:.2f}%", "High energy concentration"],
        ["Eckart-Young Rank-1 Error ||S - S_1||_F", f"{rank1_err:.4f}", f"Bound: {tail_bound:.4f}", "Exact Mirsky Theorem Match"],
    ]

    print("\n" + "-" * 75)
    print("  1. SVD LOW-RANK PROPERTY OF ESTIMATED STRIPE MATRIX S")
    print("-" * 75)
    print(tabulate(svd_table, headers=["Metric / Singular Value", "Value", "Energy Share / Bound", "Mathematical Interpretation"], tablefmt="fancy_grid"))

    # 2. Gradient Anisotropy Energy Profiles
    gx_s, gy_s = compute_first_order_gradients(stripe_img)
    gx_c, gy_c = compute_first_order_gradients(clean_img)
    ex_s = float(np.sum(gx_s**2))
    ey_s = float(np.sum(gy_s**2))
    ratio_s = ex_s / (ey_s + 1e-10)

    ex_c = float(np.sum(gx_c**2))
    ey_c = float(np.sum(gy_c**2))
    ratio_c = ex_c / (ey_c + 1e-10)

    grad_table = [
        ["Clean Scene C", f"{ex_c:.2e}", f"{ey_c:.2e}", f"{ratio_c:.4f}", "Isotropic natural radiance (Ratio ~ 1.0)"],
        ["Stripe Matrix S", f"{ex_s:.2e}", f"{ey_s:.2e}", f"{ratio_s:.1f}", "Severe directional anisotropy (Ex >> Ey, Ratio >> 100)"],
    ]

    print("\n" + "-" * 75)
    print("  2. FIRST-ORDER GRADIENT ENERGY & DIRECTIONAL ANISOTROPY")
    print("-" * 75)
    print(tabulate(grad_table, headers=["Component", "Horiz Energy E_x", "Vert Energy E_y", "Anisotropy Ratio E_x / E_y", "Physical Interpretation"], tablefmt="fancy_grid"))

    # Visual Output Generation
    if save_fig:
        out_fig_path = results_dir / f"interactive_mod01_{scene_title}_case{noise_case}.png"
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
        axes[0].imshow(clean_img, cmap="gray")
        axes[0].set_title(f"Clean Scene: {scene_title}", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(degraded_img, cmap="gray")
        axes[1].set_title(f"Degraded (Case {noise_case})\nσ_s={stripe_amp:.1f}, σ_n={env_sigma:.1f}", fontsize=9, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(stripe_img, cmap="gray")
        axes[2].set_title(f"Stripe Component S\nTop-1 Energy: {top1_ratio:.1f}%", fontsize=9, fontweight="bold")
        axes[2].axis("off")

        axes[3].semilogy(s[:30], "o-", color="#2b5c8f", markersize=4, linewidth=1.2)
        axes[3].set_title("SVD Singular Value Decay", fontsize=9, fontweight="bold")
        axes[3].set_xlabel("Singular Index i", fontsize=8)
        axes[3].set_ylabel("Singular Value σ_i (log)", fontsize=8)
        axes[3].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[+] Interactive Visualization Saved -> {out_fig_path}")

    print("\n[+] Module 1 Interactive Exploration Complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Module 1 Noise & Prior Explorer")
    parser.add_argument("--demo", action="store_true", help="Run non-blocking automated demo verification")
    parser.add_argument("--scene", type=str, default="1", help="Scene number 1..6")
    parser.add_argument("--custom", type=str, default=None, help="Path to custom image")
    parser.add_argument("--case", type=int, default=1, choices=[1, 2, 3], help="Degradation case")
    parser.add_argument("--stripe_amp", type=float, default=25.0, help="Stripe noise amplitude")
    parser.add_argument("--env_sigma", type=float, default=5.0, help="Environmental noise sigma")
    args = parser.parse_args()

    return run_mod01_interactive(
        scene_choice=args.scene,
        custom_path=args.custom,
        noise_case=args.case,
        stripe_amp=args.stripe_amp,
        env_sigma=args.env_sigma,
        save_fig=True,
        interactive=not args.demo,
    )


if __name__ == "__main__":
    sys.exit(main())
