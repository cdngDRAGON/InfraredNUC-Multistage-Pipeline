"""Resolution Scaling Benchmark for Infrared Destriping Pipeline.

Evaluates Stage 1 (ADMM) and Stage 2 (Patch Low-Rank SVD) runtime and RAM footprint
across varying spatial resolutions: 128x128, 256x256, 512x512, and 1024x1024.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from mod01_priors_and_noise_simulator.src.noise_simulator import simulate_case1_conventional
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank


def synthesize_benchmark_scene(size: int) -> np.ndarray:
    """Generate smooth synthetic infrared scene of given resolution."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    img = 120.0 + 40.0 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    return img


def benchmark_resolution_scaling(
    resolutions: List[int] = [128, 256, 512, 1024],
    stage1_iters: int = 25,
    stage2_stride: int = 6,
) -> pd.DataFrame:
    """Benchmark Stage 1, Stage 2, and Full Pipeline across spatial resolutions.

    Parameters
    ----------
    resolutions : List[int]
        Image side dimensions to benchmark.
    stage1_iters : int
        Number of ADMM iterations for benchmark consistency.
    stage2_stride : int
        Patch stride (stride=6 provides fast benchmarking across large resolutions).

    Returns
    -------
    df : pd.DataFrame
        DataFrame recording [Resolution, Pixels, Stage1_Time_s, Stage2_Time_s, Total_Time_s, Peak_RAM_MB].
    """
    rows = []
    print(f"\n[Resolution Scaling] Evaluating spatial dimensions {resolutions}...")

    for N in resolutions:
        clean = synthesize_benchmark_scene(N)
        sim = simulate_case1_conventional(clean, stripe_sigma=12.0, env_sigma=5.0, seed=42)
        deg = sim.degraded

        tracemalloc.start()
        tracemalloc.reset_peak()

        # 1. Stage 1 ADMM Benchmark
        t0 = time.perf_counter()
        c_s1, _, _ = solve_stage1_admm(
            deg, lambda1=0.5, lambda2=0.05, lambda3=2.0, rho=0.5, maxiter=stage1_iters
        )
        t_stage1 = time.perf_counter() - t0

        # 2. Stage 2 Patch SVD Benchmark
        t0 = time.perf_counter()
        _ = denoise_stage2_patch_lowrank(
            c_s1,
            patch_size=11,
            stride=stage2_stride,
            search_radius=12,
            num_similar_patches=50,
            n_jobs=-1,
        )
        t_stage2 = time.perf_counter() - t0

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024.0 * 1024.0)
        t_total = t_stage1 + t_stage2

        print(
            f"  --> {N:4d}x{N:<4d} | Pixels: {N*N:8,d} | Stage 1: {t_stage1:.3f}s | "
            f"Stage 2: {t_stage2:.3f}s | Total: {t_total:.3f}s | RAM: {peak_mb:.1f} MB"
        )

        rows.append(
            {
                "Resolution": f"{N}x{N}",
                "Pixels": N * N,
                "Stage1_Time_s": t_stage1,
                "Stage2_Time_s": t_stage2,
                "Total_Time_s": t_total,
                "Peak_RAM_MB": peak_mb,
            }
        )

    return pd.DataFrame(rows)
