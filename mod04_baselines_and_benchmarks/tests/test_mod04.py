"""Unit tests for Module 04: 11-Method Benchmark Suite."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from common.metrics import evaluate_all_metrics
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
from mod04_baselines_and_benchmarks.src.proposed_pipeline import solve_proposed_ours
from mod04_baselines_and_benchmarks.src.run_mod04 import (
    BENCHMARK_METHODS,
    evaluate_scene_all_methods,
    run_single_method,
)


@pytest.fixture
def test_scene() -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic test scene with vertical stripes."""
    np.random.seed(42)
    H, W = 64, 64
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)
    clean = 128.0 + 35.0 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)

    # Vertical stripe noise (rank-1 column offsets)
    stripe_cols = np.random.normal(0.0, 15.0, size=(1, W))
    stripe = np.tile(stripe_cols, (H, 1))

    # Add environmental Gaussian noise
    noise = np.random.normal(0.0, 4.0, size=(H, W))

    degraded = np.clip(clean + stripe + noise, 0.0, 255.0)
    return clean, degraded


def test_mod04_structure():
    """Verify standard directory structure."""
    base = Path(__file__).resolve().parent.parent
    assert (base / "src").is_dir()
    assert (base / "docs").is_dir()
    assert (base / "resources").is_dir()
    assert (base / "results").is_dir()


def test_filter_baselines(test_scene):
    """Test 1D-GF, 1D-WLS, and BM3D execution and properties."""
    clean, degraded = test_scene

    c_gf, s_gf = solve_1d_gf(degraded)
    assert c_gf.shape == degraded.shape
    assert np.all(np.isfinite(c_gf))

    c_wls, s_wls = solve_1d_wls(degraded)
    assert c_wls.shape == degraded.shape
    assert np.all(np.isfinite(c_wls))

    c_bm3d, s_bm3d = solve_bm3d(degraded, stride=8)
    assert c_bm3d.shape == degraded.shape
    assert np.all(np.isfinite(c_bm3d))


def test_optim_baselines(test_scene):
    """Test ADOM, LRSID, and GSRSR execution and properties."""
    clean, degraded = test_scene

    c_adom, s_adom = solve_adom(degraded, max_iter=15)
    assert c_adom.shape == degraded.shape
    assert np.all(np.isfinite(c_adom))

    c_lrsid, s_lrsid = solve_lrsid(degraded, max_iter=15)
    assert c_lrsid.shape == degraded.shape
    assert np.all(np.isfinite(c_lrsid))

    c_gsrsr, s_gsrsr = solve_gsrsr(degraded, max_iter=15)
    assert c_gsrsr.shape == degraded.shape
    assert np.all(np.isfinite(c_gsrsr))


def test_learning_baselines(test_scene):
    """Test SNRCNN, DLS-NUC, SNRWDNN, and DestripeCycleGAN execution."""
    clean, degraded = test_scene

    c_cnn, s_cnn = solve_snrcnn(degraded)
    assert c_cnn.shape == degraded.shape
    assert np.all(np.isfinite(c_cnn))

    c_dls, s_dls = solve_dls_nuc(degraded)
    assert c_dls.shape == degraded.shape
    assert np.all(np.isfinite(c_dls))

    c_wdnn, s_wdnn = solve_snrwdnn(degraded)
    assert c_wdnn.shape == degraded.shape
    assert np.all(np.isfinite(c_wdnn))

    c_gan, s_gan = solve_destripe_cyclegan(degraded)
    assert c_gan.shape == degraded.shape
    assert np.all(np.isfinite(c_gan))


def test_proposed_pipeline(test_scene):
    """Test proposed two-stage unified pipeline."""
    clean, degraded = test_scene
    c_ours, s_ours, diag = solve_proposed_ours(degraded, maxiter_s1=20, fast_mode=True)
    assert c_ours.shape == degraded.shape
    assert np.all(np.isfinite(c_ours))
    assert "stage1_time_sec" in diag
    assert "stage2_time_sec" in diag


def test_all_11_methods_benchmark(test_scene):
    """Verify all 11 methods run through evaluation and Ours achieves top performance."""
    clean, degraded = test_scene
    scene_evals = evaluate_scene_all_methods(clean, degraded, fast_mode=True)

    assert "Degraded" in scene_evals
    for method in BENCHMARK_METHODS:
        assert method in scene_evals
        metrics = scene_evals[method]["metrics"]
        assert "PSNR" in metrics
        assert "SSIM" in metrics
        assert "AVGE" in metrics
        assert "rho" in metrics

    # Verify Ours achieves significant PSNR improvement over Degraded
    psnr_deg = scene_evals["Degraded"]["metrics"]["PSNR"]
    psnr_ours = scene_evals["Ours"]["metrics"]["PSNR"]
    assert psnr_ours > psnr_deg + 4.0

    # Verify Ours achieves top-2 PSNR among all tested algorithms
    all_psnrs = [scene_evals[m]["metrics"]["PSNR"] for m in BENCHMARK_METHODS]
    rank_ours = sum(1 for p in all_psnrs if p > psnr_ours)
    assert rank_ours <= 1, f"Ours rank ({rank_ours}) should be 0 (1st) or 1 (2nd)"
