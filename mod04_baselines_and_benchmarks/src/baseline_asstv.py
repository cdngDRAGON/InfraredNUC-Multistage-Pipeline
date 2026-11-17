"""Anisotropic Spectral-Spatial Total Variation (ASSTV) Destriping Baseline.

Implements unidirectional anisotropic TV destriping separating D = C + S:
    min_{C, S} (1/2) * ||D - C - S||_F^2 + lambda_c * ||grad_y(C)||_1 + lambda_s * ||grad_x(S)||_1 + gamma * ||grad_y(S)||_F^2

Leverages the directional prior that vertical stripe noise has zero vertical gradient (grad_y(S) ~ 0)
and sparse horizontal gradient (grad_x(S)), while the image structure is preserved along the vertical axis.
Solved via Alternating Direction Method of Multipliers (ADMM).
"""

from __future__ import annotations

import numpy as np


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-thresholding operator for L1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def solve_asstv_destriping(
    D: np.ndarray,
    lambda_c: float = 0.8,
    lambda_s: float = 1.2,
    gamma: float = 15.0,
    rho: float = 1.0,
    max_iter: int = 40,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ASSTV destriping via ADMM.

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    lambda_c : float
        Weight on vertical TV of the clean image: ||grad_y(C)||_1.
    lambda_s : float
        Weight on horizontal TV of the stripe component: ||grad_x(S)||_1.
    gamma : float
        Weight on vertical smoothness of stripes: ||grad_y(S)||_F^2.
    rho : float
        ADMM penalty parameter.
    max_iter : int
        Maximum number of ADMM iterations.
    tol : float
        Relative convergence tolerance.

    Returns
    -------
    C : np.ndarray
        Restored clean image estimate of shape (H, W).
    S : np.ndarray
        Estimated stripe noise component of shape (H, W).
    """
    img = D.astype(np.float64)
    H, W = img.shape

    # Precompute 2D FFT transfer functions for finite differences
    # Forward difference kernels with periodic boundary conditions
    # grad_x: [1, -1] along horizontal axis (axis 1)
    # grad_y: [1, -1] along vertical axis (axis 0)
    fx = np.zeros((H, W), dtype=np.complex128)
    fx[0, 0] = -1.0
    fx[0, 1] = 1.0
    eig_fx = np.fft.fft2(fx)

    fy = np.zeros((H, W), dtype=np.complex128)
    fy[0, 0] = -1.0
    fy[1, 0] = 1.0
    eig_fy = np.fft.fft2(fy)

    eig_fx_sq = np.abs(eig_fx) ** 2
    eig_fy_sq = np.abs(eig_fy) ** 2

    # Denominators for FFT updates
    denom_C = 1.0 + rho * eig_fy_sq
    denom_S = 1.0 + 2.0 * gamma * eig_fy_sq + rho * eig_fx_sq

    # Initialize variables
    C = np.copy(img)
    S = np.zeros((H, W), dtype=np.float64)

    # Auxiliary variables for TV
    # V_c ~ grad_y(C)
    # V_s ~ grad_x(S)
    Vc = np.zeros((H, W), dtype=np.float64)
    Vs = np.zeros((H, W), dtype=np.float64)

    # Dual multipliers
    Uc = np.zeros((H, W), dtype=np.float64)
    Us = np.zeros((H, W), dtype=np.float64)

    for _ in range(max_iter):
        C_prev = np.copy(C)

        # 1. Update S:
        # min_S (1/2)||D - C - S||_F^2 + gamma ||grad_y(S)||_F^2 + (rho/2)||grad_x(S) - (Vs - Us)||_F^2
        # Normal equation in frequency domain:
        rhs_S = (img - C) + rho * np.fft.ifft2(
            np.conj(eig_fx) * np.fft.fft2(Vs - Us)
        ).real
        S = np.fft.ifft2(np.fft.fft2(rhs_S) / denom_S).real

        # 2. Update C:
        # min_C (1/2)||D - C - S||_F^2 + (rho/2)||grad_y(C) - (Vc - Uc)||_F^2
        rhs_C = (img - S) + rho * np.fft.ifft2(
            np.conj(eig_fy) * np.fft.fft2(Vc - Uc)
        ).real
        C = np.fft.ifft2(np.fft.fft2(rhs_C) / denom_C).real

        # 3. Compute gradients for auxiliary updates
        grad_y_C = np.fft.ifft2(eig_fy * np.fft.fft2(C)).real
        grad_x_S = np.fft.ifft2(eig_fx * np.fft.fft2(S)).real

        # 4. Soft-thresholding for Vc and Vs
        Vc = _soft_threshold(grad_y_C + Uc, lambda_c / rho)
        Vs = _soft_threshold(grad_x_S + Us, lambda_s / rho)

        # 5. Dual multiplier updates
        Uc += grad_y_C - Vc
        Us += grad_x_S - Vs

        # Convergence check
        rel_diff = np.linalg.norm(C - C_prev) / (np.linalg.norm(C_prev) + 1e-8)
        if rel_diff < tol:
            break

    C = np.clip(C, 0.0, 255.0)
    S = img - C
    return C, S
