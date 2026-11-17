"""mod06_runtime_and_parfor_scaling/src/run_mod06.py

Master CLI Runner for Module 6: Runtime Efficiency, Two-Stage Breakdown, and Parfor Scaling.

Implements Section IV-C4 of the IEEE Sensors Journal paper:
  - Table II: Runtime comparison across 6 dataset categories for all 11 comparison methods,
    Ours (Single-Thread), and Ours (Parfor).
  - Fig. 23: Stage 1 vs. Stage 2 computation latency breakdown in Single-Thread mode (~70% vs. ~30%)
    and multi-threaded Parfor mode (> 1.8x speedup).

Exports:
  - results/table2_runtime_comparison.csv
  - docs/table2_runtime.tex
  - results/stage_breakdown.csv
  - results/fig23_stage_time_comparison.png
and synchronizes artifacts into reports/ieee_sensors_nuc/.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import pandas as pd

from common.benchmark import BenchmarkSuite, profile_stage
from common.plot_style import set_ieee_style
from mod06_runtime_and_parfor_scaling.src.runtime_profiler import run_runtime_profiler
from mod06_runtime_and_parfor_scaling.src.stage_breakdown import run_stage_breakdown


def copy_artifacts_to_reports(results_dir: Path, workspace_root: Path) -> None:
    """Copy all generated Module 6 figures and CSVs into reports/ieee_sensors_nuc/."""
    rep_figs = workspace_root / "reports" / "ieee_sensors_nuc" / "figures"
    rep_tbls = workspace_root / "reports" / "ieee_sensors_nuc" / "tables"
    rep_figs.mkdir(parents=True, exist_ok=True)
    rep_tbls.mkdir(parents=True, exist_ok=True)

    # Copy Fig. 23
    for f in results_dir.glob("fig23_*.png"):
        dest = rep_figs / f.name
        shutil.copy2(f, dest)
        print(f"  [Report Sync] Figure copied -> {dest}")

    # Copy Table II and breakdown CSVs
    for csv_file in results_dir.glob("*.csv"):
        dest = rep_tbls / csv_file.name
        shutil.copy2(csv_file, dest)
        print(f"  [Report Sync] Table copied -> {dest}")

    # Copy Table II LaTeX
    tex_src = workspace_root / "mod06_runtime_and_parfor_scaling" / "docs" / "table2_runtime.tex"
    if tex_src.exists():
        tex_dest = rep_tbls / "table2_runtime.tex"
        shutil.copy2(tex_src, tex_dest)
        print(f"  [Report Sync] Table II LaTeX copied -> {tex_dest}")


def main() -> int:
    """Master CLI execution pipeline for Module 6."""
    set_ieee_style()
    suite = BenchmarkSuite(name="IEEE_Sensors_Mod06_Runtime_Scaling")

    workspace_root = Path(__file__).resolve().parent.parent.parent
    mod06_dir = workspace_root / "mod06_runtime_and_parfor_scaling"
    results_dir = mod06_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  IEEE SENSORS JOURNAL (FEB 2026) - MODULE 6: RUNTIME & PARFOR SCALING")
    print(f"  Workspace:        {workspace_root}")
    print(f"  Output Directory: {results_dir}")
    print("=" * 80)

    # 1. Multi-Dataset Runtime Profiler (Table II)
    @profile_stage(name="Table2_Multi_Dataset_Runtime_Profiling", suite=suite)
    def _run_table2():
        print("\n>>> 1. Benchmarking All 12 Methods Across 6 Datasets for Table II...")
        df_tbl2 = run_runtime_profiler(
            workspace_root=workspace_root,
            results_dir=results_dir,
            fast_mode=False,
        )
        print(f"    Completed. {len(df_tbl2)} methods evaluated.")

    _run_table2()

    # 2. Two-Stage Time Ablation (Fig. 23 & stage_breakdown.csv)
    @profile_stage(name="Fig23_Two_Stage_Time_Ablation", suite=suite)
    def _run_stage_breakdown():
        print("\n>>> 2. Measuring Stage 1 vs. Stage 2 Time Breakdown (Fig. 23)...")
        df_bd = run_stage_breakdown(
            workspace_root=workspace_root,
            results_dir=results_dir,
            fast_mode=False,
        )
        print(f"    Completed. {len(df_bd)} dataset profiles evaluated.")

    _run_stage_breakdown()

    # 3. Synchronize generated artifacts with reports/
    print("\n>>> 3. Synchronizing Generated Figures & Tables with reports/ieee_sensors_nuc/...")
    copy_artifacts_to_reports(results_dir, workspace_root)

    print("\n" + "=" * 80)
    print("  MODULE 6 BENCHMARK & EXECUTION SUMMARY:")
    for r in suite.records:
        print(f"    - {r.stage_name:<42}: {r.elapsed_sec:6.2f} s (Peak RAM: {r.peak_memory_mb:5.2f} MB)")
    print("=" * 80)
    print("  MODULE 6 COMPLETED SUCCESSFULLY: Table II and Fig. 23 Generated.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
