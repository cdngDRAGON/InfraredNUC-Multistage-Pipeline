"""mod03_stage2_patch_lowrank/src/svd_patch_denoiser.py

Stage 2 Adaptive SVD Low-Rank Patch Denoiser & Multi-Threaded Back-Projection (Eqs. 24-26).

Mathematical Formulation:
- Observation matrix SVD:
    G = U * Sigma * V^T = sum_{i=1}^n sigma_i * u_i * v_i^T (Eq. 24)
  where n = min(d, L) = min(121, 50) = 50.
- Noise energy criterion for adaptive truncation rank k (Eq. 26):
    sum_{i=k}^n sigma_i^2 > tau^2 >= sum_{i=k+1}^n sigma_i^2
  where tau^2 = gamma * d * L * tau_sigma^2 is the calibrated noise energy bound.
- Low-rank reconstruction:
    G_hat_k = U_k * Sigma_k * V_k^T (Eq. 25)
- Multi-threaded back-projection (joblib.Parallel Parfor equivalent) using 2D Kaiser
  window weighting to aggregate denoised patches onto the image canvas.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
from joblib import Parallel, delayed
import numpy as np

from mod03_stage2_patch_lowrank.src.patch_matching import (
    PatchGroup,
    extract_sliding_patches_view,
    find_similar_patches_for_ref,
    get_reference_patch_grid,
)


def estimate_noise_sigma_mad(img: np.ndarray) -> float:
    """Robust Median Absolute Deviation (MAD) noise standard deviation estimator."""
    arr = np.asarray(img, dtype=np.float64)
    # High-pass filter via diagonal second differences
    diff = arr[1:, 1:] - arr[:-1, 1:] - arr[1:, :-1] + arr[:-1, :-1]
    sigma_est = float(np.median(np.abs(diff)) / (0.6745 * 2.0))
    return max(1.0, sigma_est)


def create_2d_kaiser_window(patch_size: int = 11, beta: float = 2.0) -> np.ndarray:
    """Create a 2D separable Kaiser window for seamless patch aggregation."""
    kw = np.kaiser(patch_size, beta)
    window_2d = np.outer(kw, kw)
    # Normalize so peak is 1.0
    window_2d /= np.max(window_2d)
    return window_2d


def select_adaptive_rank_k(
    singular_values: np.ndarray,
    tau_squared: float,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> int:
    """Select the adaptive truncation rank k satisfying Eq. (26).

    Equation 26:
        sum_{i=k}^n sigma_i^2 > tau^2 >= sum_{i=k+1}^n sigma_i^2

    Parameters
    ----------
    singular_values : np.ndarray
        Singular values in non-increasing order: sigma_1 >= sigma_2 >= ... >= sigma_n >= 0.
    tau_squared : float
        Calibrated noise energy threshold tau^2.
    min_rank : int
        Minimum allowed rank (default 1 to preserve DC structural energy).
    max_rank : Optional[int]
        Maximum allowed rank.

    Returns
    -------
    k : int
        Optimal truncation rank k in [min_rank, n].
    """
    s = singular_values
    n = len(s)
    limit_rank = n if max_rank is None else min(n, max_rank)

    # Compute tail sums: tail_sums[j] = sum_{i=j}^{n-1} s_i^2 (0-indexed)
    s_sq = s ** 2
    tail_sums = np.zeros(n + 1, dtype=np.float64)
    tail_sums[:n] = np.cumsum(s_sq[::-1])[::-1]
    tail_sums[n] = 0.0

    # Find the smallest k >= min_rank such that tail_sums[k] <= tau_squared
    # (where tail_sums[k] corresponds to sum_{i=k+1}^n sigma_i^2 in 1-based indexing)
    selected_k = min_rank
    for k in range(min_rank, limit_rank + 1):
        if tail_sums[k] <= tau_squared:
            selected_k = k
            break
        selected_k = k

    return int(selected_k)


def denoise_single_patch_group(
    ref_coord: Tuple[int, int],
    all_patches_view: np.ndarray,
    patch_size: int,
    search_radius: int,
    num_similar_patches: int,
    tau_squared: float,
    kaiser_window_flat: np.ndarray,
) -> Tuple[List[Tuple[int, int]], np.ndarray, float, int]:
    """Process a single reference patch: matching, SVD, adaptive truncation, and weighting."""
    group = find_similar_patches_for_ref(
        ref_coord=ref_coord,
        all_patches_view=all_patches_view,
        patch_size=patch_size,
        search_radius=search_radius,
        num_similar_patches=num_similar_patches,
    )

    G = group.observation_matrix  # Shape (d, L)
    d, L = G.shape

    # Mean centering per patch column
    col_means = np.mean(G, axis=0, keepdims=True)
    G_centered = G - col_means

    # SVD (Eq. 24)
    U, s, Vt = np.linalg.svd(G_centered, full_matrices=False)

    # Adaptive rank selection (Eq. 26)
    k = select_adaptive_rank_k(s, tau_squared, min_rank=1, max_rank=min(15, len(s)))

    # Low-rank reconstruction (Eq. 25)
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    G_hat_centered = (U_k * s_k) @ Vt_k
    G_hat = G_hat_centered + col_means

    # Cluster weight: inversely proportional to rank complexity + noise
    weight = 1.0 / float(k + 0.1)

    # Multiply reconstructed patches by Kaiser window
    weighted_patches = G_hat * kaiser_window_flat[:, None]

    return group.patch_coords, weighted_patches, weight, k


def denoise_stage2_patch_lowrank(
    img: np.ndarray,
    patch_size: int = 11,
    stride: int = 3,
    search_radius: int = 15,
    num_similar_patches: int = 50,
    tau_sigma: Optional[float] = None,
    gamma: float = 0.95,
    n_jobs: int = -1,
    return_diagnostics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Stage 2 Low-Rank Patch Approximation & Back-Projection (Section III-D, Eqs. 22-26).

    Parameters
    ----------
    img : np.ndarray
        Stage 1 restored infrared image C of shape (H, W) in [0, 255].
    patch_size : int
        Square patch dimension B = 11 (Eq. 22).
    stride : int
        Step size between reference patches (default 3 for dense overlapping coverage).
    search_radius : int
        Radius R of local search window (default 15).
    num_similar_patches : int
        Number of similar patches L grouped into matrix G (default 50).
    tau_sigma : Optional[float]
        Noise standard deviation. If None, estimated via robust MAD.
    gamma : float
        Noise threshold energy calibration factor (default 0.95).
    n_jobs : int
        Parallel worker threads for joblib (default -1 uses all available CPU cores).
    return_diagnostics : bool
        If True, returns (C_stage2, diagnostics_dict).

    Returns
    -------
    C_stage2 : np.ndarray
        Final restored thermal infrared image in [0, 255].
    diagnostics : Dict[str, Any]
        (Optional) Telemetry dictionary with rank statistics and timing.
    """
    arr = np.asarray(img, dtype=np.float64)
    H, W = arr.shape
    d = patch_size * patch_size

    t_start = time.perf_counter()

    # 1. Calibrate noise energy threshold tau^2 (Eq. 26)
    sigma_n = tau_sigma if tau_sigma is not None else estimate_noise_sigma_mad(arr)
    # Expected noise energy for centered d x L matrix
    tau_squared = gamma * (d - 1) * (num_similar_patches - 1) * (sigma_n ** 2)

    # 2. Extract sliding patch view and reference coordinates
    patches_view = extract_sliding_patches_view(arr, patch_size=patch_size)
    ref_coords = get_reference_patch_grid((H, W), patch_size=patch_size, stride=stride)

    kaiser_window_2d = create_2d_kaiser_window(patch_size, beta=2.0)
    kaiser_flat = kaiser_window_2d.ravel()

    # 3. Parallel Execution across Reference Patches (Parfor equivalent)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(denoise_single_patch_group)(
            ref_coord=coord,
            all_patches_view=patches_view,
            patch_size=patch_size,
            search_radius=search_radius,
            num_similar_patches=num_similar_patches,
            tau_squared=tau_squared,
            kaiser_window_flat=kaiser_flat,
        )
        for coord in ref_coords
    )

    # 4. Aggregation Back-Projection onto Canvas
    canvas_image = np.zeros((H, W), dtype=np.float64)
    canvas_weight = np.zeros((H, W), dtype=np.float64)
    selected_ranks: List[int] = []

    B = patch_size
    for patch_coords, weighted_patches, weight, k in results:
        selected_ranks.append(k)
        effective_weight = weight * kaiser_window_2d

        for idx, (r, c) in enumerate(patch_coords):
            patch_2d = weighted_patches[:, idx].reshape(B, B)
            canvas_image[r : r + B, c : c + B] += weight * patch_2d
            canvas_weight[r : r + B, c : c + B] += effective_weight

    # Normalize by accumulated weights
    mask = canvas_weight > 1e-8
    C_stage2 = np.copy(arr)
    C_stage2[mask] = canvas_image[mask] / canvas_weight[mask]
    C_stage2 = np.clip(C_stage2, 0.0, 255.0)

    elapsed_sec = time.perf_counter() - t_start

    if return_diagnostics:
        diagnostics = {
            "estimated_sigma": sigma_n,
            "tau_squared": tau_squared,
            "total_ref_patches": len(ref_coords),
            "mean_rank": float(np.mean(selected_ranks)),
            "median_rank": float(np.median(selected_ranks)),
            "min_rank": int(np.min(selected_ranks)),
            "max_rank": int(np.max(selected_ranks)),
            "selected_ranks": selected_ranks,
            "elapsed_sec": elapsed_sec,
        }
        return C_stage2, diagnostics

    return C_stage2
