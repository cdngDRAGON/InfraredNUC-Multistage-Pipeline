"""Multi-Core Parallel (Parfor) Scaling Benchmark for Stage 2 Patch Low-Rank SVD.

Evaluates parallel scaling efficiency, speedup factors, and Amdahl's Law parameters
across varying worker process counts p in [1, 2, 4, 8].
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank


def benchmark_core_scaling(
    img: np.ndarray,
    core_counts: List[int] | None = None,
    stride: int = 4,
    search_radius: int = 12,
    num_similar_patches: int = 50,
) -> pd.DataFrame:
    """Benchmark Stage 2 runtime and compute speedup across core counts.

    Parameters
    ----------
    img : np.ndarray
        Input infrared image of shape (H, W), float64.
    core_counts : List[int], optional
        List of core counts to evaluate. Defaults to [1, 2, 4, min(8, os.cpu_count())].
    stride : int
        Patch extraction stride.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns [Cores, Time_sec, Speedup, Efficiency_pct, Theoretical_Amdahl].
    """
    max_cpus = os.cpu_count() or 4
    if core_counts is None:
        core_counts = [1, 2, 4]
        if max_cpus >= 8:
            core_counts.append(8)
        elif max_cpus > 4:
            core_counts.append(max_cpus)

    arr = np.asarray(img, dtype=np.float64)
    runtimes = []

    print(f"\n[Parallel Scaling] Benchmarking cores {core_counts} on {arr.shape} image...")

    # Warmup
    _ = denoise_stage2_patch_lowrank(
        arr, patch_size=11, stride=stride, search_radius=search_radius,
        num_similar_patches=num_similar_patches, n_jobs=1
    )

    for p in core_counts:
        # Multiple runs for statistical stability
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            _ = denoise_stage2_patch_lowrank(
                arr,
                patch_size=11,
                stride=stride,
                search_radius=search_radius,
                num_similar_patches=num_similar_patches,
                n_jobs=p,
            )
            times.append(time.perf_counter() - t0)
        t_avg = float(np.min(times))
        runtimes.append((p, t_avg))
        print(f"  --> Cores: {p:2d} | Min Execution Time: {t_avg:.3f} s")

    # Compute speedup S_p = T_1 / T_p and efficiency
    t_serial = runtimes[0][1]
    rows = []

    # Fit empirical Amdahl parallel fraction P:
    # S_p = 1 / ((1 - P) + P / p) => (1 - P) + P/p = 1 / S_p
    # Estimate P from the highest tested core count
    p_max, t_max = runtimes[-1]
    s_max = t_serial / t_max
    if p_max > 1:
        P_est = min(0.98, max(0.5, (1.0 - 1.0 / s_max) / (1.0 - 1.0 / p_max)))
    else:
        P_est = 0.90

    for p, t_p in runtimes:
        speedup = t_serial / t_p
        efficiency = (speedup / p) * 100.0
        amdahl_speedup = 1.0 / ((1.0 - P_est) + P_est / p)

        rows.append(
            {
                "Cores": p,
                "Time_sec": t_p,
                "Speedup": speedup,
                "Efficiency_pct": efficiency,
                "Amdahl_Speedup": amdahl_speedup,
                "Amdahl_P": P_est,
            }
        )

    return pd.DataFrame(rows)
