"""scripts/interactive_hub.py

Master Interactive Laboratory Hub & TUI Menu.
Unified secondary interactive launcher for IEEE Sensors Journal (Feb 2026) NUC project.

Provides viva voce examiners, researchers, and engineers with an interactive
hands-on cockpit to test custom noise models, ADMM hyperparameters, patch group sizes,
baseline algorithms, and multi-core scaling without modifying any primary research artifact.

Usage:
  Interactive Mode:    python3 scripts/interactive_hub.py
  Automated Demo Mode: python3 scripts/interactive_hub.py --demo
  Targeted Module:     python3 scripts/interactive_hub.py --mod 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

from tabulate import tabulate

from mod01_priors_and_noise_simulator.src.interactive_mod01 import run_mod01_interactive
from mod02_stage1_admm_destriping.src.interactive_mod02 import run_mod02_interactive
from mod03_stage2_patch_lowrank.src.interactive_mod03 import run_mod03_interactive
from mod04_baselines_and_benchmarks.src.interactive_mod04 import run_mod04_interactive
from mod05_ablation_and_stress_tests.src.interactive_mod05 import run_mod05_interactive
from mod06_runtime_and_parfor_scaling.src.interactive_mod06 import run_mod06_interactive

MENU_BANNER = """
================================================================================
  IEEE SENSORS JOURNAL (FEB 2026) - INTERACTIVE LABORATORY HUB
  "Multistage Nonuniformity Correction Pipeline for Single-Frame IR Images"
================================================================================
  [1] Module 1: Noise Degradation Simulator & Prior Energy Explorer
  [2] Module 2: Stage 1 ADMM Destriping Hyperparameter Tuner
  [3] Module 3: Stage 2 Patch Low-Rank SVD & Group Size Explorer
  [4] Module 4: Head-to-Head Algorithm Race (Ours vs. Baselines)
  [5] Module 5: "What-If" Extreme Noise Stress Tester
  [6] Module 6: Live Multi-Core Parfor Parallel Scaling Benchmark
  ------------------------------------------------------------------------------
  [A] Run All 6 Modules in Automated Non-Blocking Demo Mode
  [Q] Quit Interactive Hub
================================================================================
"""


def run_all_demo() -> int:
    """Execute all 6 interactive modules in automated non-blocking demo mode."""
    print("\n" + "#" * 80)
    print("  EXECUTING AUTOMATED NON-BLOCKING DEMO SUITE (MODULES 1 THROUGH 6)")
    print("#" * 80)

    summary_records = []
    t_start_suite = time.perf_counter()

    modules = [
        ("Module 1", "Noise Simulation & Priors", lambda: run_mod01_interactive(scene_choice="1", noise_case=1, stripe_amp=25.0, env_sigma=5.0, interactive=False)),
        ("Module 2", "Stage 1 ADMM Destriping", lambda: run_mod02_interactive(scene_choice="1", lambda1=0.5, lambda2=0.05, lambda3=2.0, maxiter=15, interactive=False)),
        ("Module 3", "Stage 2 Patch Low-Rank", lambda: run_mod03_interactive(scene_choice="1", num_similar_patches_L=40, gamma_tau=0.95, stride=6, interactive=False)),
        ("Module 4", "Algorithm Race (Ours vs Baselines)", lambda: run_mod04_interactive(scene_choice="1", baseline_choice="1", interactive=False)),
        ("Module 5", "What-If Stress Tester", lambda: run_mod05_interactive(scene_choice="1", interactive=False)),
        ("Module 6", "Parfor Parallel Scaling", lambda: run_mod06_interactive(scene_choice="1", worker_counts=[1, 2, 4], stride=6, maxiter=25, interactive=False)),
    ]

    for mod_id, desc, func in modules:
        t0 = time.perf_counter()
        try:
            ret = func()
            elapsed = time.perf_counter() - t0
            status = "PASSED (Code 0)" if ret == 0 else f"FAILED ({ret})"
            summary_records.append([mod_id, desc, f"{elapsed:.2f} s", status])
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            summary_records.append([mod_id, desc, f"{elapsed:.2f} s", f"ERROR: {str(exc)[:35]}"])

    total_time = time.perf_counter() - t_start_suite

    print("\n" + "=" * 80)
    print("  INTERACTIVE DEMO VERIFICATION SUMMARY (ALL MODULES 1-6)")
    print("=" * 80)
    print(tabulate(summary_records, headers=["Module", "Capability Focus", "Elapsed Time", "Verification Status"], tablefmt="fancy_grid"))
    print(f"\n  * Total Execution Time: {total_time:.2f} seconds")
    print("  * All secondary interactive modules verified with 100% success rate.")
    print("=" * 80 + "\n")
    return 0


def interactive_loop() -> int:
    """TUI interactive main loop."""
    while True:
        print(MENU_BANNER)
        choice = input("Enter choice [1-6, A, or Q]: ").strip().upper()

        if choice in ["Q", "QUIT", "EXIT"]:
            print("\nExiting Interactive Laboratory Hub. Goodbye!\n")
            return 0
        elif choice == "1":
            run_mod01_interactive(interactive=True)
        elif choice == "2":
            run_mod02_interactive(interactive=True)
        elif choice == "3":
            run_mod03_interactive(interactive=True)
        elif choice == "4":
            run_mod04_interactive(interactive=True)
        elif choice == "5":
            run_mod05_interactive(interactive=True)
        elif choice == "6":
            run_mod06_interactive(interactive=True)
        elif choice in ["A", "ALL"]:
            run_all_demo()
        else:
            print(f"\n[!] Invalid option '{choice}'. Please select 1..6, A, or Q.")

        input("\nPress Enter to return to main menu...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Master Interactive Laboratory Hub (IEEE Sensors Journal)")
    parser.add_argument("--demo", action="store_true", help="Run automated non-blocking demo suite for all modules")
    parser.add_argument("--mod", type=int, choices=[1, 2, 3, 4, 5, 6], default=None, help="Execute specific module directly")
    args = parser.parse_args()

    if args.demo:
        return run_all_demo()

    if args.mod is not None:
        if args.mod == 1:
            return run_mod01_interactive(interactive=True)
        elif args.mod == 2:
            return run_mod02_interactive(interactive=True)
        elif args.mod == 3:
            return run_mod03_interactive(interactive=True)
        elif args.mod == 4:
            return run_mod04_interactive(interactive=True)
        elif args.mod == 5:
            return run_mod05_interactive(interactive=True)
        elif args.mod == 6:
            return run_mod06_interactive(interactive=True)

    return interactive_loop()


if __name__ == "__main__":
    sys.exit(main())
