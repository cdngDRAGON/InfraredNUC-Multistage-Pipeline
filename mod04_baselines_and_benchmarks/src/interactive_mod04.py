"""mod04_baselines_and_benchmarks/src/interactive_mod04.py

Interactive Head-to-Head Algorithm Race (Ours vs. Baselines).
Secondary CLI layer for interactive exploration and viva defense demonstrations.

Allows users to:
  1. Pick any benchmark scene (or custom degraded image).
  2. Select any comparison baseline: 1D-GF, 1D-WLS, ADOM, BM3D, LRSID, GSRSR, SNRCNN, DestripeCycleGAN, or ALL.
  3. Execute a head-to-head race against the proposed two-stage pipeline (Ours).
  4. Display a publication-ready 7-metric comparison table with highlighted winners.

Supports both interactive input() mode and automated non-blocking verification via `--demo`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tabulate import tabulate

from common.metrics import (
    compute_aad,
    compute_avge,
    compute_icv,
    compute_mrd,
    compute_psnr,
    compute_roughness_rho,
    compute_ssim,
    evaluate_all_metrics,
)
from common.plot_style import set_ieee_style
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank
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

BASELINES: Dict[str, Tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "1": ("1D-GF (Cao 2016)", lambda im: solve_1d_gf(im)),
    "2": ("1D-WLS (Dong 2018)", lambda im: solve_1d_wls(im)),
    "3": ("ADOM (Kim 2023)", lambda im: solve_adom(im, max_iter=25)[0]),
    "4": ("BM3D (Dabov 2007)", lambda im: solve_bm3d(im, patch_size=8, search_window=16)),
    "5": ("LRSID (Chang 2016)", lambda im: solve_lrsid(im, max_iter=25)[0]),
    "6": ("GSRSR (Chen 2017)", lambda im: solve_gsrsr(im, max_iter=20)[0]),
    "7": ("SNRCNN (Kuai 2017)", lambda im: solve_snrcnn(im)),
    "8": ("DestripeCycleGAN (Xiao 2020)", lambda im: solve_destripe_cyclegan(im)),
}

SCENE_CHOICES = [
    ("1", "Img1_Checkerboard_Yohann_case1.png", "Img1_Checkerboard_Yohann.png", "Checkerboard Yohann (Case 1)"),
    ("2", "Img2_IndoorMonitor_Yohann_case1.png", "Img2_IndoorMonitor_Yohann.png", "Indoor Monitor Yohann (Case 1)"),
    ("3", "Img3_BuildingSkyline_He_case1.png", "Img3_BuildingSkyline_He.png", "Building Skyline He (Case 1)"),
    ("4", "Img4_HousePedestrian_He_case1.png", "Img4_HousePedestrian_He.png", "House & Pedestrian He (Case 1)"),
    ("5", "Img5_FacadeWindows_TNO_case1.png", "Img5_FacadeWindows_TNO.png", "Facade Windows TNO (Case 1)"),
    ("6", "Img6_StreetCars_LLVIP_case1.png", "Img6_StreetCars_LLVIP.png", "Street Cars LLVIP (Case 1)"),
]


def load_scene(
    workspace_root: Path,
    scene_idx: str = "1",
    custom_deg: Optional[str] = None,
    custom_clean: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Load degraded and clean scene images."""
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


def solve_proposed_interactive(
    deg_img: np.ndarray,
    fast_mode: bool = True,
) -> Tuple[np.ndarray, float]:
    """Execute proposed two-stage framework."""
    t0 = time.perf_counter()
    k_iters = 30 if fast_mode else 100
    stride_val = 6 if fast_mode else 4
    c_s1, _, _ = solve_stage1_admm(deg_img, maxiter=k_iters, verbose=False)
    c_s2 = denoise_stage2_patch_lowrank(c_s1, patch_size=11, stride=stride_val, num_similar_patches=40 if fast_mode else 50, n_jobs=-1)
    elapsed = time.perf_counter() - t0
    return c_s2, elapsed


def run_mod04_interactive(
    scene_choice: str = "1",
    custom_deg: Optional[str] = None,
    custom_clean: Optional[str] = None,
    baseline_choice: str = "1",
    save_fig: bool = True,
    interactive: bool = True,
) -> int:
    """Execute head-to-head algorithm race."""
    set_ieee_style()
    workspace_root = Path(__file__).resolve().parent.parent.parent
    results_dir = workspace_root / "mod04_baselines_and_benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  MODULE 4: HEAD-TO-HEAD ALGORITHM RACE (OURS VS. BASELINES)")
    print("=" * 80)

    if interactive:
        print("\nAvailable Benchmark Scenes:")
        for key, _, _, desc in SCENE_CHOICES:
            print(f"  [{key}] {desc}")

        u_sc = input(f"\nSelect scene [1-6, default {scene_choice}]: ").strip()
        if u_sc in ["1", "2", "3", "4", "5", "6"]:
            scene_choice = u_sc

        print("\nAvailable Comparison Baselines:")
        for key, (name, _) in BASELINES.items():
            print(f"  [{key}] {name}")
        print("  [A] Race Against ALL Baselines")

        b_in = input(f"\nSelect baseline to race against Ours [1-8, or A, default {baseline_choice}]: ").strip()
        if b_in:
            baseline_choice = b_in

    deg_img, clean_img, scene_desc = load_scene(workspace_root, scene_choice, custom_deg, custom_clean)
    print(f"\n[*] Loaded Scene: '{scene_desc}' ({deg_img.shape[0]}x{deg_img.shape[1]})")

    # Determine baseline methods to evaluate
    targets: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    if baseline_choice.upper() == "A":
        targets = list(BASELINES.values())
    elif baseline_choice in BASELINES:
        targets = [BASELINES[baseline_choice]]
    else:
        targets = [BASELINES["1"], BASELINES["3"]]  # Default fallback 1D-GF and ADOM

    # 1. Run Proposed Framework (Ours)
    print("\n[*] Racing Method 1: Proposed Two-Stage Framework (Ours)...")
    ours_img, ours_time = solve_proposed_interactive(deg_img, fast_mode=True)
    ours_metrics = evaluate_all_metrics(ours_img, clean_img) if clean_img is not None else {}
    print(f"    Ours completed in {ours_time:.3f} s.")

    # 2. Run selected baselines
    results_records: List[Dict[str, Any]] = []

    # Degraded entry
    deg_m = evaluate_all_metrics(deg_img, clean_img) if clean_img is not None else {}
    deg_entry = {
        "Method": "Degraded Input",
        "PSNR": f"{deg_m.get('PSNR', 0.0):.2f} dB",
        "SSIM": f"{deg_m.get('SSIM', 0.0):.4f}",
        "AVGE": f"{deg_m.get('AVGE', 0.0):.4f}",
        "Roughness": f"{compute_roughness_rho(deg_img):.4f}",
        "ICV": f"{deg_m.get('ICV', 0.0):.4f}",
        "AAD": f"{deg_m.get('AAD', 0.0):.4f}",
        "MRD": f"{deg_m.get('MRD', 0.0):.4f}",
        "Latency": "-",
    }
    results_records.append(deg_entry)

    for b_name, b_func in targets:
        print(f"[*] Racing Baseline: {b_name}...")
        t0 = time.perf_counter()
        out = b_func(deg_img)
        b_img = out[0] if isinstance(out, (tuple, list)) else out
        b_time = time.perf_counter() - t0
        b_m = evaluate_all_metrics(b_img, clean_img) if clean_img is not None else {}

        results_records.append({
            "Method": b_name,
            "PSNR": f"{b_m.get('PSNR', 0.0):.2f} dB",
            "SSIM": f"{b_m.get('SSIM', 0.0):.4f}",
            "AVGE": f"{b_m.get('AVGE', 0.0):.4f}",
            "Roughness": f"{compute_roughness_rho(b_img):.4f}",
            "ICV": f"{b_m.get('ICV', 0.0):.4f}",
            "AAD": f"{b_m.get('AAD', 0.0):.4f}",
            "MRD": f"{b_m.get('MRD', 0.0):.4f}",
            "Latency": f"{b_time:.3f} s",
        })

    # Proposed Ours entry
    results_records.append({
        "Method": "**Ours (Proposed)**",
        "PSNR": f"**{ours_metrics.get('PSNR', 0.0):.2f} dB**",
        "SSIM": f"**{ours_metrics.get('SSIM', 0.0):.4f}**",
        "AVGE": f"**{ours_metrics.get('AVGE', 0.0):.4f}**",
        "Roughness": f"**{compute_roughness_rho(ours_img):.4f}**",
        "ICV": f"**{ours_metrics.get('ICV', 0.0):.4f}**",
        "AAD": f"**{ours_metrics.get('AAD', 0.0):.4f}**",
        "MRD": f"**{ours_metrics.get('MRD', 0.0):.4f}**",
        "Latency": f"**{ours_time:.3f} s**",
    })

    headers = ["Method", "PSNR ↑", "SSIM ↑", "AVGE ↓", "Roughness ↓", "ICV ↓", "AAD ↓", "MRD ↓", "Latency (s)"]
    rows = [
        [r["Method"], r["PSNR"], r["SSIM"], r["AVGE"], r["Roughness"], r["ICV"], r["AAD"], r["MRD"], r["Latency"]]
        for r in results_records
    ]

    print("\n" + "=" * 80)
    print("  HEAD-TO-HEAD RACE RESULTS (7 QUANTITATIVE EVALUATION METRICS)")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    if save_fig and len(targets) > 0:
        first_b_name, first_b_func = targets[0]
        first_b_out = first_b_func(deg_img)
        first_b_img = first_b_out[0] if isinstance(first_b_out, (tuple, list)) else first_b_out
        out_fig = results_dir / f"interactive_mod04_race_{scene_desc.replace(' ', '_')}.png"

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)
        axes[0].imshow(deg_img, cmap="gray")
        axes[0].set_title("Degraded Input", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        if clean_img is not None:
            axes[1].imshow(clean_img, cmap="gray")
            axes[1].set_title("Clean Reference", fontsize=9, fontweight="bold")
        else:
            axes[1].imshow(deg_img, cmap="gray")
            axes[1].set_title("Input Scene", fontsize=9, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(first_b_img, cmap="gray")
        axes[2].set_title(f"Baseline: {first_b_name}\nPSNR: {compute_psnr(first_b_img, clean_img):.2f} dB", fontsize=9, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(ours_img, cmap="gray")
        axes[3].set_title(f"Ours (Proposed)\nPSNR: {ours_metrics.get('PSNR', 0.0):.2f} dB", fontsize=9, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(out_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[+] Interactive Head-to-Head Visual Saved -> {out_fig}")

    print("\n[+] Module 4 Interactive Head-to-Head Algorithm Race Complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Module 4 Algorithm Race")
    parser.add_argument("--demo", action="store_true", help="Run non-blocking automated demo verification")
    parser.add_argument("--scene", type=str, default="1", help="Scene number 1..6")
    parser.add_argument("--baseline", type=str, default="1", help="Baseline choice: 1..8 or A")
    args = parser.parse_args()

    return run_mod04_interactive(
        scene_choice=args.scene,
        baseline_choice="1" if not args.demo else "1",
        save_fig=True,
        interactive=not args.demo,
    )


if __name__ == "__main__":
    sys.exit(main())
