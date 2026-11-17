"""mod02_stage1_admm_destriping/tests/test_mod02.py

Unit tests for Stage 1 ADMM Destriping:
1. Linear operator exact adjoint identity: <A(u), v> == <u, A^T(v)> to machine precision.
2. ISTA L1 soft-thresholding and Schatten-1/2 singular value shrinkage.
3. Stage 1 ADMM restoration fidelity: PSNR gain > 5 dB over degraded input.
"""

from pathlib import Path
import numpy as np
import pytest

from common.dataset_generator import load_scene
from common.metrics import compute_psnr, compute_ssim
from mod01_priors_and_noise_simulator.src.noise_simulator import simulate_case2_periodic
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod02_stage1_admm_destriping.src.operators import (
    compute_fft_eigenvalues,
    d2_dx2,
    d2_dx2_adj,
    div_x,
    div_y,
    grad_x,
    grad_x_adj,
    grad_y,
    grad_y_adj,
)
from mod02_stage1_admm_destriping.src.proximal_ista import (
    half_threshold_scalar,
    project_column_stripe,
    schatten_half_shrinkage,
    soft_threshold,
)


@pytest.fixture
def random_test_pair():
    """Two independent random matrices for adjoint inner-product testing."""
    rng = np.random.RandomState(42)
    u = rng.randn(64, 64)
    v = rng.randn(64, 64)
    return u, v


# ==============================================================================
# 1. Operator Adjoint Identity Tests: <A(u), v> == <u, A^T(v)>
# ==============================================================================

def test_grad_x_adjoint_identity(random_test_pair):
    """Verify <grad_x(u), v> == <u, grad_x_adj(v)> to machine precision."""
    u, v = random_test_pair
    Ax = grad_x(u)
    ATy = grad_x_adj(v)

    inner_1 = float(np.sum(Ax * v))
    inner_2 = float(np.sum(u * ATy))
    assert pytest.approx(inner_1, abs=1e-11) == inner_2


def test_grad_y_adjoint_identity(random_test_pair):
    """Verify <grad_y(u), v> == <u, grad_y_adj(v)> to machine precision."""
    u, v = random_test_pair
    Ay = grad_y(u)
    ATy = grad_y_adj(v)

    inner_1 = float(np.sum(Ay * v))
    inner_2 = float(np.sum(u * ATy))
    assert pytest.approx(inner_1, abs=1e-11) == inner_2


def test_d2_dx2_adjoint_identity(random_test_pair):
    """Verify <d2_dx2(u), v> == <u, d2_dx2_adj(v)> to machine precision."""
    u, v = random_test_pair
    Au = d2_dx2(u)
    ATv = d2_dx2_adj(v)

    inner_1 = float(np.sum(Au * v))
    inner_2 = float(np.sum(u * ATv))
    assert pytest.approx(inner_1, abs=1e-11) == inner_2


def test_fft_eigenvalues_accuracy():
    """Verify analytical FFT eigenvalues match numerical FFT of impulse response."""
    shape = (32, 32)
    eig_gx_sq, eig_gy_sq, eig_hxx_sq = compute_fft_eigenvalues(shape)

    # Numerical impulse response for grad_x
    delta = np.zeros(shape, dtype=np.float64)
    delta[0, 0] = 1.0
    k_gx = grad_x(delta)
    F_gx_sq = np.abs(np.fft.fft2(k_gx)) ** 2
    assert np.allclose(eig_gx_sq, F_gx_sq, atol=1e-12)

    # Numerical impulse response for grad_y
    k_gy = grad_y(delta)
    F_gy_sq = np.abs(np.fft.fft2(k_gy)) ** 2
    assert np.allclose(eig_gy_sq, F_gy_sq, atol=1e-12)

    # Second derivative
    k_hxx = d2_dx2(delta)
    F_hxx_sq = np.abs(np.fft.fft2(k_hxx)) ** 2
    assert np.allclose(eig_hxx_sq, F_hxx_sq, atol=1e-12)


# ==============================================================================
# 2. ISTA and Schatten-1/2 Shrinkage Tests
# ==============================================================================

def test_soft_threshold_properties():
    """Verify ISTA soft-thresholding on scalar and tensor inputs."""
    vals = np.array([-5.0, -1.0, 0.0, 1.5, 6.0])
    threshold = 2.0
    shrunk = soft_threshold(vals, threshold)

    expected = np.array([-3.0, 0.0, 0.0, 0.0, 4.0])
    assert np.allclose(shrunk, expected, atol=1e-12)


def test_half_threshold_scalar():
    """Verify analytical L_{1/2} half-thresholding operator."""
    tau = 1.0
    # Critical threshold T = 0.75 * (2 * 1.0)^(2/3) ≈ 1.19055
    T = 0.75 * ((2.0 * tau) ** (2.0 / 3.0))

    # Below threshold must be exactly 0.0
    assert half_threshold_scalar(0.5 * T, tau) == 0.0
    assert half_threshold_scalar(T, tau) == 0.0

    # Above threshold must be non-zero and less than original input
    val_high = 3.0 * T
    res = half_threshold_scalar(val_high, tau)
    assert 0.0 < res < val_high


def test_schatten_half_shrinkage():
    """Verify Schatten-1/2 singular value shrinkage on low-rank matrices."""
    rng = np.random.RandomState(42)
    # Construct rank-2 matrix + small noise
    u1, v1 = rng.randn(32, 1), rng.randn(32, 1)
    u2, v2 = rng.randn(32, 1), rng.randn(32, 1)
    mat = 50.0 * (u1 @ v1.T) + 30.0 * (u2 @ v2.T) + rng.randn(32, 32) * 0.1

    low_rank_mat, s_shrunk = schatten_half_shrinkage(mat, tau=1.0, rank_limit=2)

    # Rank must be <= 2
    rank_shrunk = np.linalg.matrix_rank(low_rank_mat, tol=1e-5)
    assert rank_shrunk <= 2
    assert s_shrunk[2] == 0.0


# ==============================================================================
# 3. Stage 1 ADMM Destriping Restoration Tests (PSNR Gain > 5 dB)
# ==============================================================================

def test_stage1_admm_psnr_gain():
    """Verify Stage 1 ADMM achieves at least +5 dB PSNR improvement over degraded input."""
    clean = load_scene("Img3_BuildingSkyline_He.png")
    sim = simulate_case2_periodic(clean, seed=42)
    deg = sim.degraded

    psnr_deg = compute_psnr(clean, deg)
    ssim_deg = compute_ssim(clean, deg)

    C_stage1, S_est, hist_df = solve_stage1_admm(
        D=deg,
        clean_ref=clean,
        lambda1=0.5,
        lambda2=0.05,
        lambda3=2.0,
        rho=0.5,
        maxiter=60,
        tol=1e-4,
        verbose=False,
    )

    psnr_stage1 = compute_psnr(clean, C_stage1)
    ssim_stage1 = compute_ssim(clean, C_stage1)
    psnr_gain = psnr_stage1 - psnr_deg
    ssim_gain = ssim_stage1 - ssim_deg

    # Assert rigorous quantitative benchmarks
    assert psnr_gain >= 5.0, f"Expected PSNR gain >= 5 dB, got +{psnr_gain:.2f} dB"
    assert ssim_gain > 0.20, f"Expected SSIM gain > 0.20, got +{ssim_gain:.4f}"

    # Verify structural bounds
    assert C_stage1.shape == clean.shape
    assert C_stage1.min() >= 0.0
    assert C_stage1.max() <= 255.0

    # Verify estimated stripe low-rankness
    _, s_stripe, _ = np.linalg.svd(S_est, full_matrices=False)
    assert s_stripe[2] / (s_stripe[0] + 1e-12) < 0.01

    # Verify objective function monotonicity
    objs = hist_df["objective"].values
    assert objs[-1] < objs[0]
