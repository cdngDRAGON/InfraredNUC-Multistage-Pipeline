"""mod01_priors_and_noise_simulator/tests/test_mod01.py

Unit tests for:
1. SVD rank-k approximation and Eckart-Young-Mirsky theorem (Eqs. 1-4).
2. Directional gradient energy anisotropy (||grad_x(S)||_F^2 >> ||grad_y(S)||_F^2).
3. 3-case nonuniform noise simulator (Eq. 5, Cases 1-3).
4. High-order directional derivative calculations.
"""

from pathlib import Path
import numpy as np
import pytest

from common.dataset_generator import load_scene
from mod01_priors_and_noise_simulator.src.gradient_priors import (
    compute_first_order_gradients,
    compute_gradient_anisotropy_ratio,
    compute_second_order_derivatives,
)
from mod01_priors_and_noise_simulator.src.lowrank_svd_analysis import (
    analyze_singular_value_decay,
    compute_rank_k_reconstruction,
    compute_svd,
)
from mod01_priors_and_noise_simulator.src.noise_simulator import (
    SimulationResult,
    generate_column_stripe,
    simulate_case1_conventional,
    simulate_case2_periodic,
    simulate_case3_enhanced,
    simulate_degradation_by_case,
)


@pytest.fixture
def synthetic_scene():
    """Deterministic synthetic thermal scene for unit testing."""
    rng = np.random.RandomState(42)
    base = np.linspace(80.0, 180.0, 128)[:, None] + np.linspace(-20.0, 20.0, 128)[None, :]
    base += rng.normal(0, 1.5, (128, 128))
    return np.clip(base, 0.0, 255.0)


# ==============================================================================
# 1. SVD and Eckart-Young-Mirsky Theorem Tests (Eqs. 1-4)
# ==============================================================================

def test_svd_reconstruction_identity(synthetic_scene):
    """Full SVD reconstruction must recover matrix exactly (Eq. 1)."""
    U, s, Vt = compute_svd(synthetic_scene)
    reconstructed = (U * s) @ Vt
    frobenius_diff = np.linalg.norm(synthetic_scene - reconstructed, "fro")
    assert frobenius_diff < 1e-10


def test_eckart_young_mirsky_error_bound(synthetic_scene):
    """Optimal rank-k approximation error must match sqrt(sum_{i=k+1} sigma_i^2) (Eqs. 2-4)."""
    U, s, Vt = compute_svd(synthetic_scene)

    for k in [1, 2, 5, 10, 20]:
        J_hat_k, err_theoretical, eta_k = compute_rank_k_reconstruction(synthetic_scene, k)

        # Numerical Frobenius error
        err_empirical = float(np.linalg.norm(synthetic_scene - J_hat_k, "fro"))
        assert pytest.approx(err_theoretical, rel=1e-5) == err_empirical

        # Energy ratio must match ||J_hat_k||_F^2 / ||J||_F^2
        energy_empirical = float(np.linalg.norm(J_hat_k, "fro") ** 2) / float(
            np.linalg.norm(synthetic_scene, "fro") ** 2
        )
        assert pytest.approx(eta_k, rel=1e-5) == energy_empirical

        # Reconstruction matrix rank must not exceed k
        rank_hat = np.linalg.matrix_rank(J_hat_k, tol=1e-6)
        assert rank_hat <= k


def test_svd_energy_monotonicity(synthetic_scene):
    """Cumulative energy capture eta_k must increase monotonically with rank k."""
    analysis = analyze_singular_value_decay(synthetic_scene, top_k_list=[1, 2, 5, 10, 20, 50])
    fractions = analysis["energy_fractions"]

    prev = 0.0
    for k in [1, 2, 5, 10, 20, 50]:
        val = fractions[k]
        assert val >= prev
        assert 0.0 < val <= 1.0000001
        prev = val


# ==============================================================================
# 2. Directional Gradient Anisotropy Tests (Figs. 2, 3, 5)
# ==============================================================================

def test_stripe_gradient_anisotropy(synthetic_scene):
    """Vertical stripes must exhibit extreme horizontal-to-vertical gradient energy ratio."""
    sim = simulate_case2_periodic(synthetic_scene, seed=42)
    stripe = sim.stripe

    aniso = compute_gradient_anisotropy_ratio(stripe)
    # Energy in horizontal gradient must vastly outpace vertical gradient (ratio > 10^3)
    assert aniso["Energy_grad_x"] > 1000.0 * aniso["Energy_grad_y"]
    assert aniso["Anisotropy_Ratio"] > 1000.0


def test_gradient_operators_analytical():
    """Verify first- and second-order derivative operators against analytical gradients."""
    y = np.arange(64, dtype=np.float64)[:, None]
    x = np.arange(64, dtype=np.float64)[None, :]

    # Plane f(x, y) = 3*x + 7*y
    plane = 3.0 * x + 7.0 * y
    gx, gy = compute_first_order_gradients(plane)

    # In interior pixels: gx == 3.0, gy == 7.0
    assert np.allclose(gx[:, :-1], 3.0, atol=1e-8)
    assert np.allclose(gy[:-1, :], 7.0, atol=1e-8)

    # Parabolic surface f(x, y) = 2 * x^2 + 5 * y^2
    parabola = 2.0 * (x**2) + 5.0 * (y**2)
    derivs = compute_second_order_derivatives(parabola)

    # d2/dx2 should be 4.0, d2/dy2 should be 10.0
    assert np.allclose(derivs["xx"][:, 1:-1], 4.0, atol=1e-8)
    assert np.allclose(derivs["yy"][1:-1, :], 10.0, atol=1e-8)


def test_second_order_derivative_stripe_corruption(synthetic_scene):
    """Stripes must severely corrupt d2/dx2 while leaving d2/dy2 relatively clean."""
    sim = simulate_case2_periodic(synthetic_scene, seed=42)
    clean = sim.clean
    deg = sim.degraded
    stripe = sim.stripe

    # 1. Pure stripe second derivative anisotropy
    stripe_derivs = compute_second_order_derivatives(stripe)
    stripe_xx_norm = float(np.linalg.norm(stripe_derivs["xx"], "fro"))
    stripe_yy_norm = float(np.linalg.norm(stripe_derivs["yy"], "fro"))
    assert stripe_xx_norm > 50.0 * stripe_yy_norm

    # 2. Total degraded observation comparison
    clean_derivs = compute_second_order_derivatives(clean)
    deg_derivs = compute_second_order_derivatives(deg)

    diff_xx = float(np.linalg.norm(deg_derivs["xx"] - clean_derivs["xx"], "fro"))
    diff_yy = float(np.linalg.norm(deg_derivs["yy"] - clean_derivs["yy"], "fro"))

    # Horizontal second derivative distortion exceeds vertical distortion
    assert diff_xx > diff_yy


# ==============================================================================
# 3. Nonuniform Noise Simulator Tests (Eq. 5, Cases 1-3)
# ==============================================================================

def test_noise_cases_structure(synthetic_scene):
    """Verify all 3 simulation cases return valid SimulationResult dataclass."""
    for case_num in [1, 2, 3]:
        res = simulate_degradation_by_case(synthetic_scene, case_number=case_num, seed=42)
        assert isinstance(res, SimulationResult)
        assert res.degraded.shape == synthetic_scene.shape
        assert res.clean.shape == synthetic_scene.shape
        assert res.stripe.shape == synthetic_scene.shape
        assert res.noise.shape == synthetic_scene.shape

        # Values should be clipped within [0, 255]
        assert res.degraded.min() >= 0.0
        assert res.degraded.max() <= 255.0


def test_noise_cases_severity_progression(synthetic_scene):
    """Case 3 must have higher degradation error than Case 2, and Case 2 higher than Case 1."""
    res1 = simulate_case1_conventional(synthetic_scene, seed=42, clip=False)
    res2 = simulate_case2_periodic(synthetic_scene, seed=42, clip=False)
    res3 = simulate_case3_enhanced(synthetic_scene, seed=42, clip=False)

    mse1 = float(np.mean((res1.degraded - synthetic_scene) ** 2))
    mse2 = float(np.mean((res2.degraded - synthetic_scene) ** 2))
    mse3 = float(np.mean((res3.degraded - synthetic_scene) ** 2))

    assert mse3 > mse2 > mse1


def test_stripe_matrix_low_rank_bound(synthetic_scene):
    """Stripe matrix S must have low effective rank (rank <= 3) across all cases."""
    for case_fn in [simulate_case1_conventional, simulate_case2_periodic, simulate_case3_enhanced]:
        res = case_fn(synthetic_scene, seed=42)
        _, s, _ = compute_svd(res.stripe)

        # Top 2 singular values must capture > 99% of Frobenius energy
        cum_energy = np.cumsum(s**2) / np.sum(s**2)
        assert cum_energy[1] > 0.99
        # Singular values past rank 3 must drop by multiple orders of magnitude
        assert s[3] / (s[0] + 1e-12) < 0.05


def test_noise_simulation_reproducibility(synthetic_scene):
    """Same random seed must yield identical degradation matrices."""
    res_a = simulate_case2_periodic(synthetic_scene, seed=99)
    res_b = simulate_case2_periodic(synthetic_scene, seed=99)
    res_c = simulate_case2_periodic(synthetic_scene, seed=100)

    assert np.array_equal(res_a.degraded, res_b.degraded)
    assert not np.array_equal(res_a.degraded, res_c.degraded)
