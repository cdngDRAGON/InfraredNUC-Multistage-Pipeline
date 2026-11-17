"""Unit tests for common/metrics.py (IEEE Sensors Journal Equations 27-31)."""

import numpy as np
import pytest
from common.metrics import (
    compute_psnr,
    compute_ssim,
    compute_avge,
    compute_roughness_rho,
    compute_icv,
    compute_aad,
    compute_mrd,
    evaluate_all_metrics,
)


@pytest.fixture
def clean_image():
    """Deterministic synthetic test image in [0, 255]."""
    rng = np.random.RandomState(42)
    img = np.linspace(50.0, 200.0, 64 * 64, dtype=np.float64).reshape((64, 64))
    img += rng.normal(0, 2.0, (64, 64))
    return np.clip(img, 0.0, 255.0)


def test_metrics_identity(clean_image):
    """When estimated image equals reference, metrics must yield identity values."""
    psnr = compute_psnr(clean_image, clean_image)
    assert np.isinf(psnr) or psnr > 100.0

    ssim = compute_ssim(clean_image, clean_image)
    assert pytest.approx(1.0, abs=1e-5) == ssim

    aad = compute_aad(clean_image, clean_image)
    assert pytest.approx(0.0, abs=1e-8) == aad

    mrd = compute_mrd(clean_image, clean_image)
    assert pytest.approx(0.0, abs=1e-8) == mrd

    avge = compute_avge(clean_image, clean_image)
    assert pytest.approx(0.0, abs=1e-8) == avge


def test_metrics_noise_sensitivity(clean_image):
    """PSNR and SSIM should decrease monotonically with additive Gaussian noise."""
    rng = np.random.RandomState(42)
    noise_low = clean_image + rng.normal(0, 2.0, clean_image.shape)
    noise_high = clean_image + rng.normal(0, 15.0, clean_image.shape)

    psnr_low = compute_psnr(clean_image, noise_low)
    psnr_high = compute_psnr(clean_image, noise_high)
    assert psnr_low > psnr_high

    ssim_low = compute_ssim(clean_image, noise_low)
    ssim_high = compute_ssim(clean_image, noise_high)
    assert ssim_low > ssim_high

    aad_low = compute_aad(clean_image, noise_low)
    aad_high = compute_aad(clean_image, noise_high)
    assert aad_low < aad_high


def test_avge_vertical_fidelity(clean_image):
    """AVGE (Eq. 27) must detect vertical distortion and remain zero for identical gradients."""
    h, w = clean_image.shape
    # Add purely horizontal stripe (varies along row/y axis) -> disturbs vertical gradient
    disturbed_v = clean_image.copy()
    disturbed_v += np.random.normal(0, 10, (h, 1))

    # Add purely vertical stripe (constant down columns) -> vertical diff is untouched!
    stripe_v = np.random.normal(0, 10, (1, w))
    disturbed_col = clean_image + stripe_v

    avge_disturbed_v = compute_avge(clean_image, disturbed_v)
    avge_disturbed_col = compute_avge(clean_image, disturbed_col)

    # Pure vertical stripes don't alter vertical differences (diff along y is constant 0)
    assert avge_disturbed_col < 1e-10
    assert avge_disturbed_v > 0.5


def test_roughness_rho(clean_image):
    """Roughness rho (Eq. 28) should increase when high-frequency stripe noise is added."""
    rho_clean = compute_roughness_rho(clean_image)
    assert rho_clean > 0.0

    # Add high-frequency vertical stripe pattern
    h, w = clean_image.shape
    stripes = np.tile([15.0, -15.0], (h, w // 2))
    noisy_striped = np.clip(clean_image + stripes, 0.0, 255.0)

    rho_striped = compute_roughness_rho(noisy_striped)
    assert rho_striped > rho_clean


def test_icv_homogeneous_regions():
    """ICV (Eq. 29) should be significantly higher for smoother homogeneous patches."""
    smooth_patch = np.full((32, 32), 150.0, dtype=np.float64)
    # Add minimal variance
    smooth_patch[0, 0] = 150.01

    noisy_patch = 150.0 + np.random.RandomState(42).normal(0, 10.0, (32, 32))

    icv_smooth = compute_icv(smooth_patch)
    icv_noisy = compute_icv(noisy_patch)

    assert icv_smooth > icv_noisy
    assert icv_noisy > 0.0


def test_evaluate_all_metrics(clean_image):
    """evaluate_all_metrics must return all 7 specified metrics."""
    est = clean_image + np.random.RandomState(42).normal(0, 3.0, clean_image.shape)
    metrics = evaluate_all_metrics(clean_image, est, degraded=est)

    expected_keys = [
        "PSNR",
        "SSIM",
        "AVGE",
        "Roughness_rho",
        "ICV",
        "AAD",
        "MRD",
        "Degraded_PSNR",
        "Degraded_SSIM",
        "Degraded_Roughness_rho",
        "Degraded_ICV",
    ]
    for key in expected_keys:
        assert key in metrics
        assert np.isfinite(metrics[key])
