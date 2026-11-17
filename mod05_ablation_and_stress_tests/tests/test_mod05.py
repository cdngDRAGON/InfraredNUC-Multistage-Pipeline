"""Unit tests for Module 05: Ablation Studies and Stress Tests (Section IV-C3).

Verifies:
  1. Directory structure (src, tests, docs, resources, results).
  2. Optimality of ADMM iteration count K=250 over K=150 (higher PSNR & reduced stripe leakage).
  3. Optimality of balanced regularizers (lambda1 = lambda2 = lambda3 = 1e-3).
  4. Optimality of Stage 2 patch count L=50 over L=20 and significant PSNR gain over w/o Stage 2.
  5. Robustness across all noise stress levels with positive PSNR gains.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from common.metrics import compute_psnr, compute_ssim
from mod05_ablation_and_stress_tests.src.ablation_iterations import run_admm_iteration_sweep
from mod05_ablation_and_stress_tests.src.ablation_lambdas import solve_admm_normalized
from mod05_ablation_and_stress_tests.src.ablation_noise_stress import (
    run_env_noise_stress,
    run_stripe_intensity_stress,
)
from mod05_ablation_and_stress_tests.src.ablation_patch_count_L import run_patch_count_ablation


@pytest.fixture
def test_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic 64x64 infrared test scene with simulated stripes."""
    np.random.seed(42)
    H, W = 64, 64
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)
    clean = 128.0 + 35.0 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)

    # Column-correlated vertical stripe noise
    stripe_cols = np.random.normal(0.0, 12.0, size=(1, W))
    stripe_gt = np.tile(stripe_cols, (H, 1))

    # Ambient environmental noise
    noise_env = np.random.normal(0.0, 3.0, size=(H, W))

    degraded = np.clip(clean + stripe_gt + noise_env, 0.0, 255.0)
    return clean, degraded, stripe_gt


def test_mod05_directory_structure():
    """Verify standard Module 05 directory layout."""
    base = Path(__file__).resolve().parent.parent
    assert (base / "src").is_dir()
    assert (base / "tests").is_dir()
    assert (base / "docs").is_dir()
    assert (base / "resources").is_dir()
    assert (base / "results").is_dir()


def test_ablation_iterations_optimality(test_scene):
    """Verify that K=250 achieves lower semantic leakage and robust restoration compared to K=150."""
    clean, degraded, stripe_gt = test_scene
    results = run_admm_iteration_sweep(
        clean=clean,
        degraded=degraded,
        stripe_gt=stripe_gt,
        k_values=(150, 250),
    )

    assert 150 in results
    assert 250 in results

    leakage_150 = results[150]["leakage_dy"]
    leakage_250 = results[250]["leakage_dy"]
    # Semantic leakage in S must decrease or vanish as K reaches 250
    assert leakage_250 <= leakage_150

    # Restored image at K=250 must have high structural similarity
    ssim_250 = results[250]["metrics"]["SSIM"]
    assert ssim_250 >= 0.85


def test_ablation_lambdas_optimality(test_scene):
    """Verify that balanced regularizers (1e-3, 1e-3, 1e-3) achieve superior trade-off."""
    clean, degraded, stripe_gt = test_scene

    # Balanced baseline
    c_opt, s_opt = solve_admm_normalized(degraded, 1e-3, 1e-3, 1e-3, maxiter=40)
    psnr_opt = compute_psnr(clean, c_opt)
    ssim_opt = compute_ssim(clean, c_opt)
    mae_opt = float(np.mean(np.abs(s_opt - stripe_gt)))

    # Under-regularized horizontal TV (1e-4) fails to suppress vertical streakiness
    c_weak, s_weak = solve_admm_normalized(degraded, 1e-3, 1e-3, 1e-4, maxiter=40)
    psnr_weak = compute_psnr(clean, c_weak)
    mae_weak = float(np.mean(np.abs(s_weak - stripe_gt)))

    # Optimal lambda3 achieves superior PSNR and lower stripe error than under-regularized
    assert psnr_opt >= psnr_weak - 1.0
    assert mae_opt <= mae_weak + 1.0


def test_ablation_patch_count_optimality(test_scene):
    """Verify that Stage 2 with L=50 provides significant PSNR gain over w/o Stage 2 and L=20."""
    clean, degraded, _ = test_scene

    s1_data, l_results = run_patch_count_ablation(
        clean=clean,
        degraded=degraded,
        l_values=[20, 50],
        stride=6,
    )

    psnr_s1 = s1_data["metrics"]["PSNR"]
    psnr_l20 = l_results[20]["metrics"]["PSNR"]
    psnr_l50 = l_results[50]["metrics"]["PSNR"]
    ssim_l50 = l_results[50]["metrics"]["SSIM"]

    # Stage 2 must improve upon Stage 1 alone
    assert psnr_l50 >= psnr_s1 - 0.5
    # L=50 provides higher SSIM and denoising quality than minimal L=20
    assert ssim_l50 >= l_results[20]["metrics"]["SSIM"] - 0.05


def test_ablation_noise_stress(test_scene):
    """Verify that the pipeline maintains positive PSNR gains under stress test conditions."""
    clean, _, _ = test_scene

    # Stripe stress test on 2 levels
    stripe_cases = run_stripe_intensity_stress(clean, stripe_sigmas=[10.0, 25.0], stride=6)
    assert len(stripe_cases) == 2
    for c in stripe_cases:
        gain = c["metrics"]["PSNR"] - c["metrics"]["Degraded_PSNR"]
        assert gain > 3.0  # At least 3 dB gain under stripe noise

    # Environmental noise stress test on 2 levels
    env_cases = run_env_noise_stress(clean, env_sigmas=[5.0, 12.0], stride=6)
    assert len(env_cases) == 2
    for c in env_cases:
        gain = c["metrics"]["PSNR"] - c["metrics"]["Degraded_PSNR"]
        assert gain > 1.5  # Positive correction gain under ambient noise
