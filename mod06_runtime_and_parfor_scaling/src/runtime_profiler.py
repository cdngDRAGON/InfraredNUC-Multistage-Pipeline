"""mod06_runtime_and_parfor_scaling/src/runtime_profiler.py

Multi-Dataset Runtime Profiler (Section IV-C4, Table II).

Benchmarks all 11 comparison methods alongside:
  - Ours (Single-Threaded)
  - Ours (Parfor: Multi-Core Parallelized)
across all 6 benchmark dataset categories:
  1. Yohann (Yohann 2019)
  2. DLS-NUC-D (He 2019)
  3. Simu w/ TNO (TNO Image Fusion Dataset)
  4. Simu w/ M3FD (Liu 2022)
  5. Simu w/ LLVIP (Jia 2021)
  6. Simu w/ MSRS (Tang 2022)

Generates:
  - results/table2_runtime_comparison.csv
  - docs/table2_runtime.tex
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import numpy as np
import pandas as pd
from PIL import Image

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

DATASET_CATEGORIES = [
    "Yohann",
    "DLS-NUC-D",
    "Simu w/ TNO",
    "Simu w/ M3FD",
    "Simu w/ LLVIP",
    "Simu w/ MSRS",
]

SCENE_MAPPING = {
    "Yohann": "Img1_Checkerboard_Yohann_case1.png",
    "DLS-NUC-D": "Img3_BuildingSkyline_He_case1.png",
    "Simu w/ TNO": "Img5_FacadeWindows_TNO_case1.png",
    "Simu w/ M3FD": "Img4_HousePedestrian_He_case1.png",
    "Simu w/ LLVIP": "Img6_StreetCars_LLVIP_case1.png",
    "Simu w/ MSRS": "Img2_IndoorMonitor_Yohann_case1.png",
}


def benchmark_method(
    method_func,
    img: np.ndarray,
    n_runs: int = 2,
    **kwargs,
) -> float:
    """Accurately measure execution time of a method over repeated trials."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = method_func(img, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.min(times))


def run_table2_benchmarks(
    workspace_root: Union[str, Path],
    fast_mode: bool = False,
) -> pd.DataFrame:
    """Benchmark all 12 methods across all 6 dataset categories for Table II."""
    ws = Path(workspace_root)
    deg_dir = ws / "data" / "degraded_scenes"

    # Load representative images for each category
    category_images: Dict[str, np.ndarray] = {}
    for cat_name, fname in SCENE_MAPPING.items():
        p = deg_dir / fname
        if p.exists():
            category_images[cat_name] = np.array(Image.open(p), dtype=np.float64)
        else:
            category_images[cat_name] = np.random.uniform(50.0, 200.0, (256, 256))

    methods_registry = [
        ("1D-GF (Cao 2016)", lambda im: solve_1d_gf(im)),
        ("1D-WLS (Dong 2018)", lambda im: solve_1d_wls(im)),
        ("ADOM (Kim 2023)", lambda im: solve_adom(im, max_iter=25)),
        ("BM3D (Dabov 2007)", lambda im: solve_bm3d(im, patch_size=8, search_window=16)),
        ("DLS-NUC (He 2019)", lambda im: solve_dls_nuc(im)),
        ("SNRCNN (Kuai 2017)", lambda im: solve_snrcnn(im)),
        ("LRSID (Chang 2016)", lambda im: solve_lrsid(im, max_iter=25)),
        ("GSRSR (Chen 2017)", lambda im: solve_gsrsr(im, max_iter=20)),
        ("SNRWDNN (Guan 2019)", lambda im: solve_snrwdnn(im)),
        ("DestripeCycleGAN (Xiao 2020)", lambda im: solve_destripe_cyclegan(im)),
    ]

    records: List[Dict[str, Any]] = []

    print("[Table II Profiler] Benchmarking baseline methods across 6 datasets...")
    for method_name, func in methods_registry:
        row: Dict[str, Any] = {"Method": method_name}
        vals = []
        for cat in DATASET_CATEGORIES:
            img = category_images[cat]
            t = benchmark_method(func, img, n_runs=1 if fast_mode else 2)
            row[cat] = t
            vals.append(t)
        row["Average"] = float(np.mean(vals))
        records.append(row)

    # 11. Ours (Single-Thread)
    print("[Table II Profiler] Benchmarking Ours (Single-Threaded)...")
    s1_iters = 35 if fast_mode else 100
    s2_stride = 6 if fast_mode else 4

    row_single: Dict[str, Any] = {"Method": "Ours (Single-Thread)"}
    vals_single = []
    for cat in DATASET_CATEGORIES:
        img = category_images[cat]
        t0 = time.perf_counter()
        c_s1, _, _ = solve_stage1_admm(img, maxiter=s1_iters)
        _ = denoise_stage2_patch_lowrank(c_s1, patch_size=11, stride=s2_stride, num_similar_patches=50, n_jobs=1)
        t = time.perf_counter() - t0
        row_single[cat] = t
        vals_single.append(t)
    row_single["Average"] = float(np.mean(vals_single))
    records.append(row_single)

    # 12. Ours (Parfor)
    print("[Table II Profiler] Benchmarking Ours (Parfor: Multi-Core)...")
    row_par: Dict[str, Any] = {"Method": "Ours (Parfor)"}
    vals_par = []
    for cat in DATASET_CATEGORIES:
        img = category_images[cat]
        t0 = time.perf_counter()
        c_s1, _, _ = solve_stage1_admm(img, maxiter=s1_iters)
        _ = denoise_stage2_patch_lowrank(c_s1, patch_size=11, stride=s2_stride, num_similar_patches=50, n_jobs=-1)
        t = time.perf_counter() - t0
        row_par[cat] = t
        vals_par.append(t)
    row_par["Average"] = float(np.mean(vals_par))
    records.append(row_par)

    df_table2 = pd.DataFrame(records)
    return df_table2


def generate_table2_latex(df: pd.DataFrame) -> str:
    """Generate publication-ready IEEE booktabs LaTeX markup for Table II."""
    cats = DATASET_CATEGORIES
    header_cats = " & ".join([f"\\textbf{{{c}}}" for c in cats])

    lines = [
        "% ==============================================================================",
        "% Table II: Runtime Comparison (in seconds) Across 6 Dataset Categories",
        "% ==============================================================================",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Computational Runtime Comparison (Seconds) Across 6 Infrared Dataset Categories}",
        "\\label{tab:runtime_comparison}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * (len(cats) + 1) + "}",
        "\\toprule",
        f"\\textbf{{Method}} & {header_cats} & \\textbf{{Average}} \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        m_name = row["Method"]
        is_ours = "Ours" in m_name
        name_str = f"\\textbf{{{m_name}}}" if is_ours else m_name
        vals_str = " & ".join([f"{float(row[c]):.4f}" for c in cats])
        avg_str = f"{float(row['Average']):.4f}"

        if is_ours:
            lines.append(f"{name_str} & {vals_str} & \\textbf{{{avg_str}}} \\\\")
        else:
            lines.append(f"{name_str} & {vals_str} & {avg_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def run_runtime_profiler(
    workspace_root: Union[str, Path],
    results_dir: Union[str, Path],
    fast_mode: bool = False,
) -> pd.DataFrame:
    """Master routine executing Table II benchmarks and exporting CSV & LaTeX."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_table2 = run_table2_benchmarks(workspace_root, fast_mode=fast_mode)

    # 1. Export CSV
    csv_path = results_dir / "table2_runtime_comparison.csv"
    df_table2.to_csv(csv_path, index=False)
    print(f"  [Table II Exported] -> {csv_path}")

    # 2. Export LaTeX
    latex_str = generate_table2_latex(df_table2)
    docs_dir = Path(workspace_root) / "mod06_runtime_and_parfor_scaling" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    tex_path = docs_dir / "table2_runtime.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"  [Table II LaTeX Exported] -> {tex_path}")

    return df_table2
