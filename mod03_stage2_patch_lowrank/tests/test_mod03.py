"""mod03_stage2_patch_lowrank/tests/test_mod03.py

Unit tests for Stage 2 Patch Low-Rank Prior Recovery:
1. Eq. 22 patch similarity Euclidean distance calculation and ascending sorting.
2. Eq. 26 adaptive SVD rank truncation noise energy tail bound.
3. Stage 2 restoration improvement over Stage 1 (PSNR and SSIM gain).
4. Parallel (Parfor) vs. serial consistency.
"""

from pathlib import Path
import numpy as np
import pytest

from common.dataset_generator import load_scene
from common.metrics import compute_psnr, compute_ssim
from mod03_stage2_patch_lowrank.src.patch_matching import (
    extract_sliding_patches_view,
    find_similar_patches_for_ref,
    get_reference_patch_grid,
)
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import (
    create_2d_kaiser_window,
    denoise_stage2_patch_lowrank,
    select_adaptive_rank_k,
)


@pytest.fixture
def synthetic_scene():
    """Deterministic synthetic thermal patch for unit testing."""
    rng = np.random.RandomState(42)
    base = np.linspace(60.0, 190.0, 64)[:, None] + np.linspace(-15.0, 15.0, 64)[None, :]
    base += rng.normal(0, 1.0, (64, 64))
    return np.clip(base, 0.0, 255.0)


# ==============================================================================
# 1. Patch Matching and Distance Sorting Tests (Eqs. 22-23)
# ==============================================================================

def test_patch_similarity_distance_sorting(synthetic_scene):
    """Verify Eq. 22 distance calculation, monotonic ascending sort, and G shape (Eq. 23)."""
    B = 11
    L = 50
    patches_view = extract_sliding_patches_view(synthetic_scene, patch_size=B)
    ref_coord = (20, 20)

    group = find_similar_patches_for_ref(
        ref_coord=ref_coord,
        all_patches_view=patches_view,
        patch_size=B,
        search_radius=15,
        num_similar_patches=L,
    )

    # Observation matrix G must have dimension (121, 50)
    assert group.observation_matrix.shape == (B * B, L)

    # First selected patch must be the reference patch itself with distance 0.0
    assert group.patch_coords[0] == ref_coord
    assert pytest.approx(0.0, abs=1e-8) == group.distances[0]

    # Distances must be sorted in non-decreasing order
    dists = group.distances
    assert np.all(dists[1:] >= dists[:-1])


def test_reference_grid_coverage(synthetic_scene):
    """Verify reference patch grid covers boundaries and maintains correct stride."""
    H, W = synthetic_scene.shape
    B = 11
    stride = 4
    grid = get_reference_patch_grid((H, W), patch_size=B, stride=stride)

    assert len(grid) > 0
    # Top-left corner must be (0, 0)
    assert grid[0] == (0, 0)
    # Bottom-right corner must terminate at (H - B, W - B)
    assert grid[-1] == (H - B, W - B)


# ==============================================================================
# 2. Adaptive SVD Rank Tail Bound Tests (Eq. 26)
# ==============================================================================

def test_adaptive_rank_tail_bound():
    """Verify Eq. 26: sum_{i=k}^n sigma_i^2 > tau^2 >= sum_{i=k+1}^n sigma_i^2."""
    # Synthetic singular values: [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]
    s = np.array([100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0], dtype=np.float64)
    n = len(s)

    # Tail energies for k = 1..n:
    # tail(1) = 50^2 + 20^2 + 10^2 + 5^2 + 2^2 + 1^2 = 2500 + 400 + 100 + 25 + 4 + 1 = 3030
    # tail(2) = 400 + 100 + 25 + 4 + 1 = 530
    # tail(3) = 100 + 25 + 4 + 1 = 130
    # tail(4) = 25 + 4 + 1 = 30
    # tail(5) = 4 + 1 = 5
    # tail(6) = 1
    # tail(7) = 0

    # If tau^2 = 500, then tail(2) = 530 > 500 >= tail(3) = 130 -> k should be 3
    tau_sq = 500.0
    k_sel = select_adaptive_rank_k(s, tau_squared=tau_sq)
    assert k_sel == 3

    # If tau^2 = 10, tail(4) = 30 > 10 >= tail(5) = 5 -> k should be 5
    k_sel_low = select_adaptive_rank_k(s, tau_squared=10.0)
    assert k_sel_low == 5

    # If tau^2 is very large (e.g. 50000), minimum rank 1 must be preserved
    k_sel_huge = select_adaptive_rank_k(s, tau_squared=50000.0)
    assert k_sel_huge == 1


# ==============================================================================
# 3. Stage 2 Denoising Fidelity and Improvement Tests
# ==============================================================================

def test_stage2_psnr_ssim_gain(synthetic_scene):
    """Verify Stage 2 low-rank denoiser improves PSNR and SSIM over noisy image."""
    rng = np.random.RandomState(42)
    # Simulate Gaussian environmental noise on synthetic scene
    noisy = synthetic_scene + rng.normal(0, 8.0, synthetic_scene.shape)

    psnr_noisy = compute_psnr(synthetic_scene, noisy)
    ssim_noisy = compute_ssim(synthetic_scene, noisy)

    denoised = denoise_stage2_patch_lowrank(
        noisy,
        patch_size=11,
        stride=4,
        search_radius=10,
        num_similar_patches=30,
        tau_sigma=8.0,
        n_jobs=1,
    )

    psnr_denoised = compute_psnr(synthetic_scene, denoised)
    ssim_denoised = compute_ssim(synthetic_scene, denoised)

    # Must yield distinct improvements
    assert psnr_denoised > psnr_noisy + 1.0, f"Expected PSNR gain > 1 dB, got +{psnr_denoised - psnr_noisy:.2f} dB"
    assert ssim_denoised > ssim_noisy + 0.05, f"Expected SSIM gain > 0.05, got +{ssim_denoised - ssim_noisy:.4f}"
    assert denoised.shape == synthetic_scene.shape
    assert denoised.min() >= 0.0
    assert denoised.max() <= 255.0


def test_parallel_vs_serial_consistency(synthetic_scene):
    """Verify joblib multi-threaded execution matches single-threaded execution."""
    rng = np.random.RandomState(42)
    noisy = synthetic_scene + rng.normal(0, 5.0, synthetic_scene.shape)

    res_serial = denoise_stage2_patch_lowrank(
        noisy, patch_size=11, stride=6, search_radius=8, num_similar_patches=20, tau_sigma=5.0, n_jobs=1
    )
    res_parallel = denoise_stage2_patch_lowrank(
        noisy, patch_size=11, stride=6, search_radius=8, num_similar_patches=20, tau_sigma=5.0, n_jobs=2
    )

    # Must be identical to machine precision
    assert np.allclose(res_serial, res_parallel, atol=1e-10)
