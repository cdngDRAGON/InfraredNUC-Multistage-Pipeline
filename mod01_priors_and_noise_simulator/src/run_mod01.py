"""mod01_priors_and_noise_simulator/src/run_mod01.py

Master Execution Runner for Module 1:
- Generates 3-case simulated degradations (Eq. 5 & Fig. 9).
- Produces Fig. 2 (Horizontal gradient energy distribution).
- Produces Fig. 3 (Vertical gradient energy distribution).
- Produces Fig. 4 (SVD singular value decay distribution across 4 scenes).
- Produces Fig. 5 (1st- and 2nd-order directional derivative maps).
- Produces Fig. 9 (2D zoom + 3D global energy surfaces across simulation cases).
- Exports SVD Frobenius energy distribution table (svd_energy_distribution.csv).
- Exports directional gradient anisotropy ratio table (gradient_anisotropy_ratio.csv).
- Profiles all stages with wall-clock timing and memory tracking.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image

from common.benchmark import BenchmarkSuite, profile_stage
from common.dataset_generator import BENCHMARK_SCENES, load_scene
from common.metrics import evaluate_all_metrics

from mod01_priors_and_noise_simulator.src.gradient_priors import (
    compute_gradient_anisotropy_ratio,
    plot_fig2_horizontal_gradient_energy,
    plot_fig3_vertical_gradient_energy,
    plot_fig5_first_and_second_order_gradients,
)
from mod01_priors_and_noise_simulator.src.lowrank_svd_analysis import (
    generate_svd_energy_table,
    plot_fig4_svd_singular_value_distribution,
    plot_fig9_simulation_cases_3d_energy,
)
from mod01_priors_and_noise_simulator.src.noise_simulator import (
    SimulationResult,
    simulate_case1_conventional,
    simulate_case2_periodic,
    simulate_case3_enhanced,
)


@profile_stage(name="Mod01_SimulateDegradations")
def run_simulation_pipeline(
    scenes_dir: str = "data/clean_scenes",
    degraded_dir: str = "data/degraded_scenes",
) -> Dict[str, Dict[str, SimulationResult]]:
    """Simulate Cases 1, 2, and 3 across benchmark scenes and save degraded outputs."""
    out_dir = Path(degraded_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, SimulationResult]] = {}

    for scene_filename in BENCHMARK_SCENES:
        clean = load_scene(scene_filename, clean_scenes_dir=scenes_dir)
        scene_stem = Path(scene_filename).stem

        c1 = simulate_case1_conventional(clean, seed=42)
        c2 = simulate_case2_periodic(clean, seed=42)
        c3 = simulate_case3_enhanced(clean, seed=42)

        results[scene_filename] = {
            "case1": c1,
            "case2": c2,
            "case3": c3,
        }

        # Save degraded scenes for downstream stages
        for c_key, c_res in [("case1", c1), ("case2", c2), ("case3", c3)]:
            save_name = f"{scene_stem}_{c_key}.png"
            uint8_arr = np.clip(np.round(c_res.degraded), 0, 255).astype(np.uint8)
            Image.fromarray(uint8_arr, mode="L").save(out_dir / save_name)

    return results


@profile_stage(name="Mod01_GradientPriorAnalysis")
def run_gradient_prior_analysis(
    results_dir: str = "mod01_priors_and_noise_simulator/results",
) -> None:
    """Generate Figs. 2, 3, 5 and export gradient anisotropy ratio table."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary illustrative scene: Img3_BuildingSkyline_He (matching paper Figs. 2, 3, 5)
    primary_scene = "Img3_BuildingSkyline_He.png"
    clean = load_scene(primary_scene)
    sim = simulate_case2_periodic(clean, seed=42)
    degraded = sim.degraded
    stripe = sim.stripe

    print("  Generating Fig. 2: Horizontal gradient energy distribution...")
    plot_fig2_horizontal_gradient_energy(
        clean=clean,
        degraded=degraded,
        save_path=str(out_dir / "fig2_horizontal_gradient_energy.png"),
    )

    print("  Generating Fig. 3: Vertical gradient energy distribution...")
    plot_fig3_vertical_gradient_energy(
        clean=clean,
        degraded=degraded,
        save_path=str(out_dir / "fig3_vertical_gradient_energy.png"),
    )

    print("  Generating Fig. 5: 1st- and 2nd-order directional derivative maps...")
    plot_fig5_first_and_second_order_gradients(
        clean=clean,
        degraded=degraded,
        save_path=str(out_dir / "fig5_first_and_second_order_gradients.png"),
    )

    # Compute anisotropy ratios across all 6 scenes
    anisotropy_rows = []
    for scene_filename in BENCHMARK_SCENES:
        c_img = load_scene(scene_filename)
        for case_idx, sim_fn in enumerate([simulate_case1_conventional, simulate_case2_periodic, simulate_case3_enhanced], 1):
            s_res = sim_fn(c_img, seed=42)
            metrics = compute_gradient_anisotropy_ratio(s_res.stripe)
            anisotropy_rows.append({
                "Scene": scene_filename,
                "Case": f"Case {case_idx}",
                "Energy_grad_x": round(metrics["Energy_grad_x"], 2),
                "Energy_grad_y": round(metrics["Energy_grad_y"], 2),
                "Anisotropy_Ratio": round(metrics["Anisotropy_Ratio"], 2),
            })

    df_aniso = pd.DataFrame(anisotropy_rows)
    csv_aniso = out_dir / "gradient_anisotropy_ratio.csv"
    df_aniso.to_csv(csv_aniso, index=False)
    print(f"  Exported gradient anisotropy table to: {csv_aniso}")


@profile_stage(name="Mod01_LowRankSVDAnalysis")
def run_lowrank_svd_analysis(
    results_dir: str = "mod01_priors_and_noise_simulator/results",
) -> None:
    """Generate Fig. 4, Fig. 9, and SVD energy distribution CSV table."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather 4 scenes for SVD analysis matching Fig. 4
    svd_scenes = [
        "Img1_Checkerboard_Yohann.png",
        "Img2_IndoorMonitor_Yohann.png",
        "Img3_BuildingSkyline_He.png",
        "Img4_HousePedestrian_He.png",
    ]
    scenes_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for s_name in svd_scenes:
        c = load_scene(s_name)
        sim = simulate_case2_periodic(c, seed=42)
        scenes_data[s_name] = (c, sim.degraded, sim.stripe)

    print("  Generating Fig. 4: SVD singular value distribution L-curves...")
    plot_fig4_svd_singular_value_distribution(
        scenes_data=scenes_data,
        save_path=str(out_dir / "fig4_svd_singular_value_distribution.png"),
        max_singular_values=50,
    )

    print("  Generating SVD Frobenius energy capture table...")
    csv_svd = out_dir / "svd_energy_distribution.csv"
    generate_svd_energy_table(
        scenes_dict=scenes_data,
        k_values=(1, 2, 5, 10, 20),
        save_csv_path=str(csv_svd),
    )
    print(f"  Exported SVD energy table to: {csv_svd}")

    # 2. Generate Fig. 9 (Simulation cases 3D energy surface) on Img3_BuildingSkyline_He
    print("  Generating Fig. 9: 2D zoomed and 3D energy surfaces across Cases 1-3...")
    primary_clean = load_scene("Img3_BuildingSkyline_He.png")
    c1 = simulate_case1_conventional(primary_clean, seed=42).degraded
    c2 = simulate_case2_periodic(primary_clean, seed=42).degraded
    c3 = simulate_case3_enhanced(primary_clean, seed=42).degraded

    plot_fig9_simulation_cases_3d_energy(
        clean=primary_clean,
        case1_deg=c1,
        case2_deg=c2,
        case3_deg=c3,
        roi_box=(70, 75, 45, 45),
        save_path=str(out_dir / "fig9_simulation_cases_3d_energy.png"),
    )


def main() -> None:
    print("======================================================================")
    print("  IEEE Sensors Journal - Module 1: Priors & Noise Simulation Runner")
    print("======================================================================")

    # 1. Run simulation pipeline
    print("\n[Step 1/3] Simulating degradation cases across 6 benchmark scenes...")
    sim_results = run_simulation_pipeline()
    print("  Successfully simulated and saved degraded scenes in data/degraded_scenes/")

    # 2. Gradient prior analysis (Figs 2, 3, 5)
    print("\n[Step 2/3] Analyzing directional gradient energy and high-order derivatives...")
    run_gradient_prior_analysis()

    # 3. SVD low-rank analysis (Figs 4, 9)
    print("\n[Step 3/3] Performing SVD low-rank verification and 3D energy rendering...")
    run_lowrank_svd_analysis()

    # 4. Telemetry and Benchmarks
    print("\n======================================================================")
    print("  Computational Profiling Summary for Module 1")
    print("======================================================================")
    suite = BenchmarkSuite.get_global_suite()
    print(suite.summary_table())
    print("\nModule 1 execution successfully finished. All figures and tables created.")


if __name__ == "__main__":
    main()
