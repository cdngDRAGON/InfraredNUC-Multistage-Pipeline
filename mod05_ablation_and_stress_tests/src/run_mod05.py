"""mod05_ablation_and_stress_tests/src/run_mod05.py

Master CLI Runner for Module 5: Section IV-C3 Ablation Studies and Stress Tests.

Executes all 4 paper ablation studies producing Figures 16 to 22:
  1. Ablation 1 (Figs. 16-17): Iteration count K in {150, 200, 250, 300} visual stripe
     estimation and PSNR/SSIM bar/line plots.
  2. Ablation 2 (Fig. 18): Regularization parameters lambda1, lambda2, lambda3 sensitivity
     grid and stripe error maps.
  3. Ablation 3 (Figs. 19-20): Nonuniform noise intensity stress testing (5 stripe cases
     and 4 ambient environmental noise cases).
  4. Ablation 4 (Figs. 21-22): Stage 2 patch count L in {20, 25, 30, 35, 40, 45, 50, 55, 60}
     vs. w/o Stage 2 baseline (lime-green bar chart and SSIM plateau plot).

Exports all CSV data to results/ and reports/ieee_sensors_nuc/tables/
Exports all 300-DPI IEEE figures to results/ and reports/ieee_sensors_nuc/figures/
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import pandas as pd

from common.benchmark import BenchmarkSuite, profile_stage
from common.plot_style import set_ieee_style
from mod05_ablation_and_stress_tests.src.ablation_iterations import evaluate_iterations_ablation
from mod05_ablation_and_stress_tests.src.ablation_lambdas import evaluate_lambdas_ablation
from mod05_ablation_and_stress_tests.src.ablation_noise_stress import evaluate_noise_stress_ablation
from mod05_ablation_and_stress_tests.src.ablation_patch_count_L import evaluate_patch_count_ablation


def copy_artifacts_to_reports(results_dir: Path, workspace_root: Path) -> None:
    """Copy all generated ablation figures and CSVs into reports/ieee_sensors_nuc."""
    rep_figs = workspace_root / "reports" / "ieee_sensors_nuc" / "figures"
    rep_tbls = workspace_root / "reports" / "ieee_sensors_nuc" / "tables"
    rep_figs.mkdir(parents=True, exist_ok=True)
    rep_tbls.mkdir(parents=True, exist_ok=True)

    # Copy figures 16 to 22
    for fig_id in range(16, 23):
        for f in results_dir.glob(f"fig{fig_id}_*.png"):
            dest = rep_figs / f.name
            shutil.copy2(f, dest)
            print(f"  [Report Sync] Figure copied -> {dest}")

    # Copy CSV tables
    for csv_file in results_dir.glob("ablation_*.csv"):
        dest = rep_tbls / csv_file.name
        shutil.copy2(csv_file, dest)
        print(f"  [Report Sync] Table copied -> {dest}")


def main() -> int:
    """Master CLI execution pipeline."""
    set_ieee_style()
    suite = BenchmarkSuite(name="IEEE_Sensors_Mod05_Ablation_Studies")

    workspace_root = Path(__file__).resolve().parent.parent.parent
    mod05_dir = workspace_root / "mod05_ablation_and_stress_tests"
    results_dir = mod05_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark scene
    clean_scene = workspace_root / "data" / "clean_scenes" / "Img1_Checkerboard_Yohann.png"
    if not clean_scene.exists():
        clean_candidates = list((workspace_root / "data" / "clean_scenes").glob("*.png"))
        if not clean_candidates:
            print("ERROR: No clean scenes found in data/clean_scenes/")
            return 1
        clean_scene = clean_candidates[0]

    print("=" * 80)
    print("  IEEE SENSORS JOURNAL (FEB 2026) - MODULE 5: ABLATION & STRESS TESTS")
    print(f"  Representative Scene: {clean_scene.name}")
    print(f"  Output Directory:     {results_dir}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Ablation Study 1: Iterations K (Figs. 16 & 17)
    # --------------------------------------------------------------------------
    @profile_stage(name="Ablation_1_Iterations_K", suite=suite)
    def _run_abl1():
        print("\n>>> Executing Ablation 1: Iteration Count K in {150, 200, 250, 300}...")
        df_iter = evaluate_iterations_ablation(
            clean_path=clean_scene,
            results_dir=results_dir,
            k_values=(150, 200, 250, 300),
        )
        print(f"    Completed. {len(df_iter)} iterations evaluated.")

    _run_abl1()

    # --------------------------------------------------------------------------
    # 2. Ablation Study 2: Regularization Lambdas (Fig. 18)
    # --------------------------------------------------------------------------
    @profile_stage(name="Ablation_2_Regularization_Lambdas", suite=suite)
    def _run_abl2():
        print("\n>>> Executing Ablation 2: Regularization Parameters (lambda1, lambda2, lambda3)...")
        df_lambdas = evaluate_lambdas_ablation(
            clean_path=clean_scene,
            results_dir=results_dir,
            maxiter=60,
        )
        print(f"    Completed. {len(df_lambdas)} configurations evaluated.")

    _run_abl2()

    # --------------------------------------------------------------------------
    # 3. Ablation Study 3: Nonuniform Noise Stress Tests (Figs. 19 & 20)
    # --------------------------------------------------------------------------
    @profile_stage(name="Ablation_3_Noise_Stress_Tests", suite=suite)
    def _run_abl3():
        print("\n>>> Executing Ablation 3: Operational Noise Stress Tests...")
        df_stress = evaluate_noise_stress_ablation(
            clean_path=clean_scene,
            results_dir=results_dir,
            stride=4,
        )
        print(f"    Completed. {len(df_stress)} stress test conditions evaluated.")

    _run_abl3()

    # --------------------------------------------------------------------------
    # 4. Ablation Study 4: Optimal Patch Count L in Stage 2 (Figs. 21 & 22)
    # --------------------------------------------------------------------------
    @profile_stage(name="Ablation_4_Patch_Count_L", suite=suite)
    def _run_abl4():
        print("\n>>> Executing Ablation 4: Stage 2 Patch Group Size L in {20..60}...")
        df_l = evaluate_patch_count_ablation(
            clean_path=clean_scene,
            results_dir=results_dir,
            stride=4,
        )
        print(f"    Completed. {len(df_l)} configurations evaluated.")

    _run_abl4()

    # --------------------------------------------------------------------------
    # 5. Synchronize artifacts to reports/
    # --------------------------------------------------------------------------
    print("\n>>> Synchronizing generated figures and tables with reports/ieee_sensors_nuc/...")
    copy_artifacts_to_reports(results_dir, workspace_root)

    print("\n" + "=" * 80)
    print("  MODULE 5 BENCHMARK & EXECUTION SUMMARY:")
    for r in suite.records:
        print(f"    - {r.stage_name:<38}: {r.elapsed_sec:6.2f} s (Peak Mem: {r.peak_memory_mb:5.2f} MB)")
    print("=" * 80)
    print("  ALL 4 ABLATION STUDIES COMPLETED SUCCESSFULLY (Figs. 16-22 generated).")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
