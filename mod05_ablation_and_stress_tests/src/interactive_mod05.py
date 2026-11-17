"""mod05_ablation_and_stress_tests/src/interactive_mod05.py

Interactive "What-If" Extreme Noise Stress Tester.
Secondary CLI layer for interactive exploration and viva defense demonstrations.

Allows users to:
  1. Pick any clean infrared benchmark scene (or custom clean image).
  2. Synthesize custom "What-If" stress conditions with arbitrary stripe amplitude
     (sigma_s in [0..100]) and ambient noise intensity (sigma_n in [0..50]).
  3. Classify degradation stress regime (Mild, Moderate, Severe, Extreme, Catastrophic).
  4. Test pipeline resilience and verify stability bounds (Delta_PSNR >= +5.0 dB).
  5. Generate multi-panel stress diagnostics figures and metrics tables.

Supports both interactive input() mode and automated non-blocking verification via `--demo`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tabulate import tabulate

from common.metrics import compute_avge, compute_psnr, compute_roughness_rho, compute_ssim
from common.plot_style import set_ieee_style
from mod01_priors_and_noise_simulator.src.noise_simulator import generate_column_stripe
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank

SCENE_CHOICES = [
    ("1", "Img1_Checkerboard_Yohann.png", "Checkerboard Yohann"),
    ("2", "Img2_IndoorMonitor_Yohann.png", "Indoor Monitor Yohann"),
    ("3", "Img3_BuildingSkyline_He.png", "Building Skyline He"),
    ("4", "Img4_HousePedestrian_He.png", "House & Pedestrian He"),
    ("5", "Img5_FacadeWindows_TNO.png", "Facade Windows TNO"),
    ("6", "Img6_StreetCars_LLVIP.png", "Street Cars LLVIP"),
]


def load_clean_scene(
    workspace_root: Path,
    scene_idx: str = "1",
    custom_path: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Load clean infrared scene."""
    if custom_path:
        p = Path(custom_path)
        if not p.exists():
            raise FileNotFoundError(f"Custom clean image not found: {custom_path}")
        return np.array(Image.open(p).convert("L"), dtype=np.float64), p.stem

    clean_dir = workspace_root / "data" / "clean_scenes"
    for key, fname, desc in SCENE_CHOICES:
        if scene_idx.strip() == key or scene_idx.strip().lower() in fname.lower():
            p = clean_dir / fname
            if p.exists():
                return np.array(Image.open(p).convert("L"), dtype=np.float64), desc

    p = clean_dir / SCENE_CHOICES[0][1]
    if p.exists():
        return np.array(Image.open(p).convert("L"), dtype=np.float64), SCENE_CHOICES[0][2]

    # Synthetic clean fallback
    x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    clean = 128.0 + 60.0 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)
    return clean, "Synthetic Calibration"


def synthesize_stress_noise(
    clean_img: np.ndarray,
    stripe_sigma: float,
    env_sigma: float,
    include_defect_lines: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize custom extreme stress noise."""
    H, W = clean_img.shape
    rng = np.random.RandomState(seed)

    # 1. Column stripe vector
    s_col = rng.normal(0.0, stripe_sigma, W)
    if include_defect_lines:
        # Add high-intensity saturated defect columns
        defect_cols = rng.choice(W, size=max(1, W // 40), replace=False)
        s_col[defect_cols] += rng.choice([-1.0, 1.0], size=len(defect_cols)) * stripe_sigma * 2.5

    S = generate_column_stripe((H, W), s_col, vertical_drift_ratio=0.02, rng=rng)

    # 2. Environmental noise
    N = rng.normal(0.0, env_sigma, (H, W))

    # 3. Observation
    D = np.clip(clean_img + S + N, 0.0, 255.0)
    return D, S, N


def classify_stress_regime(stripe_sigma: float, env_sigma: float) -> str:
    """Categorize operational stress severity."""
    score = stripe_sigma * 0.6 + env_sigma * 1.4
    if score < 20.0:
        return "Mild (Nominal Sensor Drift)"
    elif score < 40.0:
        return "Moderate (Elevated FPN & Readout Jitter)"
    elif score < 65.0:
        return "Severe (Hostile High-Temp Uncooled Oper)"
    elif score < 95.0:
        return "Extreme (Near Saturated Multi-Defect Regime)"
    else:
        return "Catastrophic (Detector Breakdown Margin)"


def evaluate_stress_case(
    clean_img: np.ndarray,
    stripe_sigma: float,
    env_sigma: float,
    scene_name: str,
    fast_mode: bool = True,
) -> Dict[str, Any]:
    """Run pipeline against a specific stress condition and compute resilience."""
    deg_img, S_true, N_true = synthesize_stress_noise(
        clean_img,
        stripe_sigma=stripe_sigma,
        env_sigma=env_sigma,
        include_defect_lines=(stripe_sigma >= 40.0),
        seed=42,
    )

    t0 = time.perf_counter()
    k_iters = 30 if fast_mode else 80
    stride = 6 if fast_mode else 4
    c_s1, s_est, _ = solve_stage1_admm(deg_img, maxiter=k_iters, verbose=False)
    c_s2 = denoise_stage2_patch_lowrank(c_s1, patch_size=11, stride=stride, num_similar_patches=40 if fast_mode else 50, n_jobs=-1)
    latency = time.perf_counter() - t0

    psnr_in = compute_psnr(deg_img, clean_img)
    ssim_in = compute_ssim(deg_img, clean_img)
    psnr_out = compute_psnr(c_s2, clean_img)
    ssim_out = compute_ssim(c_s2, clean_img)
    delta_psnr = psnr_out - psnr_in
    delta_ssim = ssim_out - ssim_in

    # Stripe suppression metric: reduction in vertical stripe variance
    residual_stripe = s_est - S_true
    stripe_suppression_db = 10.0 * np.log10(np.var(S_true) / (np.var(residual_stripe) + 1e-8))

    regime = classify_stress_regime(stripe_sigma, env_sigma)
    status = "ROBUST (PASS)" if delta_psnr >= 5.0 else ("MARGINAL" if delta_psnr >= 2.5 else "CRITICAL")

    return {
        "Scene": scene_name,
        "Stripe_Sigma": stripe_sigma,
        "Env_Sigma": env_sigma,
        "Regime": regime,
        "PSNR_In": psnr_in,
        "SSIM_In": ssim_in,
        "PSNR_Out": psnr_out,
        "SSIM_Out": ssim_out,
        "Delta_PSNR": delta_psnr,
        "Delta_SSIM": delta_ssim,
        "Stripe_Suppression_dB": stripe_suppression_db,
        "Status": status,
        "Latency": latency,
        "deg_img": deg_img,
        "restored_img": c_s2,
        "stripe_est": s_est,
    }


def run_mod05_interactive(
    scene_choice: str = "1",
    custom_path: Optional[str] = None,
    stripe_sigma: float = 35.0,
    env_sigma: float = 12.0,
    save_fig: bool = True,
    interactive: bool = True,
) -> int:
    """Execute interactive Module 5 stress testing."""
    set_ieee_style()
    workspace_root = Path(__file__).resolve().parent.parent.parent
    results_dir = workspace_root / "mod05_ablation_and_stress_tests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  MODULE 5: 'WHAT-IF' EXTREME NOISE STRESS TESTER")
    print("=" * 80)

    clean_img, scene_name = load_clean_scene(workspace_root, scene_choice, custom_path)

    cases_to_run: List[Tuple[float, float, str]] = []

    if interactive:
        print("\nAvailable Benchmark Scenes:")
        for key, _, desc in SCENE_CHOICES:
            print(f"  [{key}] {desc}")
        print("  [C] Custom Image Path")

        u_sc = input(f"\nSelect scene [1-6, default {scene_choice}]: ").strip()
        if u_sc.upper() == "C":
            cp = input("Enter path to custom clean image: ").strip()
            if cp:
                clean_img, scene_name = load_clean_scene(workspace_root, custom_path=cp)
        elif u_sc in ["1", "2", "3", "4", "5", "6"]:
            clean_img, scene_name = load_clean_scene(workspace_root, scene_idx=u_sc)

        print("\nStress Test Scenarios:")
        print("  [1] Custom What-If Parameters (enter your own σ_s and σ_n)")
        print("  [2] Multi-Tier Escalation Suite (Moderate -> Severe -> Extreme -> Catastrophic)")

        mode_in = input("\nSelect stress test mode [1 or 2, default 1]: ").strip()
        if mode_in == "2":
            cases_to_run = [
                (15.0, 5.0, "Tier 1: Nominal Drift"),
                (30.0, 12.0, "Tier 2: Moderate Stress"),
                (50.0, 20.0, "Tier 3: Severe Hostile"),
                (75.0, 30.0, "Tier 4: Extreme Boundary"),
            ]
        else:
            s_in = input(f"Enter custom Stripe Amplitude sigma_s [0..100, default {stripe_sigma:.1f}]: ").strip()
            if s_in:
                try:
                    stripe_sigma = float(s_in)
                except ValueError:
                    pass

            e_in = input(f"Enter custom Environmental Noise sigma_n [0..50, default {env_sigma:.1f}]: ").strip()
            if e_in:
                try:
                    env_sigma = float(e_in)
                except ValueError:
                    pass

            cases_to_run = [(stripe_sigma, env_sigma, "Custom What-If Stress")]
    else:
        # Demo mode: run 2 representative stress tests
        cases_to_run = [
            (25.0, 10.0, "Moderate Stress Test"),
            (50.0, 20.0, "Severe Stress Test"),
        ]

    print(f"\n[*] Evaluating {len(cases_to_run)} Stress Test Scenario(s) on '{scene_name}'...")

    table_rows = []
    last_res = None
    for s_sig, e_sig, tag in cases_to_run:
        print(f"\n[*] Running: {tag} (σ_s={s_sig:.1f}, σ_n={e_sig:.1f})...")
        res = evaluate_stress_case(clean_img, s_sig, e_sig, scene_name, fast_mode=True)
        last_res = res

        table_rows.append([
            tag,
            f"σ_s={s_sig:.1f}, σ_n={e_sig:.1f}",
            res["Regime"],
            f"{res['PSNR_In']:.2f} dB",
            f"{res['PSNR_Out']:.2f} dB",
            f"+{res['Delta_PSNR']:.2f} dB",
            f"{res['SSIM_In']:.4f} → {res['SSIM_Out']:.4f}",
            f"{res['Stripe_Suppression_dB']:.1f} dB",
            res["Status"],
        ])

    headers = [
        "Scenario",
        "Parameters",
        "Stress Regime",
        "Degraded PSNR",
        "Restored PSNR",
        "Δ PSNR",
        "SSIM Transition",
        "Stripe Suppr",
        "Resilience",
    ]

    print("\n" + "=" * 80)
    print("  WHAT-IF NOISE STRESS TEST & RESILIENCE EVALUATION")
    print("=" * 80)
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))

    if save_fig and last_res is not None:
        out_fig = results_dir / f"interactive_mod05_stress_{scene_name.replace(' ', '_')}.png"
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

        axes[0].imshow(clean_img, cmap="gray")
        axes[0].set_title(f"Clean Scene: {scene_name}", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(last_res["deg_img"], cmap="gray")
        axes[1].set_title(f"Stressed (σ_s={last_res['Stripe_Sigma']:.0f}, σ_n={last_res['Env_Sigma']:.0f})\nPSNR: {last_res['PSNR_In']:.2f} dB", fontsize=9, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(last_res["restored_img"], cmap="gray")
        axes[2].set_title(f"Restored (Proposed)\nPSNR: {last_res['PSNR_Out']:.2f} dB (+{last_res['Delta_PSNR']:.2f} dB)", fontsize=9, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(last_res["stripe_est"], cmap="gray")
        axes[3].set_title(f"Estimated Stripe Component\nSuppression: {last_res['Stripe_Suppression_dB']:.1f} dB", fontsize=9, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(out_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[+] Interactive Stress Figure Saved -> {out_fig}")

    print("\n[+] Module 5 Interactive What-If Stress Testing Complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Module 5 Stress Tester")
    parser.add_argument("--demo", action="store_true", help="Run non-blocking automated demo verification")
    parser.add_argument("--scene", type=str, default="1", help="Scene number 1..6")
    parser.add_argument("--stripe_sigma", type=float, default=35.0, help="Custom stripe amplitude")
    parser.add_argument("--env_sigma", type=float, default=12.0, help="Custom environmental noise sigma")
    args = parser.parse_args()

    return run_mod05_interactive(
        scene_choice=args.scene,
        stripe_sigma=args.stripe_sigma,
        env_sigma=args.env_sigma,
        save_fig=True,
        interactive=not args.demo,
    )


if __name__ == "__main__":
    sys.exit(main())
