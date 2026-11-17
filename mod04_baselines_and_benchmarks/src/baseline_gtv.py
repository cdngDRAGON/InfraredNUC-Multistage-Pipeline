"""Guided Total Variation (GTV) Destriping Baseline.

Implements structural gradient-guided destriping:
    min_C (1/2) * ||C - D||_F^2 + lambda_x * ||W_x * grad_x(C)||_1 + lambda_y * ||grad_y(C)||_1

where the spatial guidance weight matrix W_x(x, y) is adaptively constructed from
vertical edge indicators:
    W_x = exp( - (|grad_y(D)| / (sigma_g + 1e-6))^alpha )
This relaxes horizontal smoothing along genuine structural boundaries while applying strong
stripe suppression in smooth and vertically homogeneous regions.
Solved via iteratively reweighted least squares / ADMM.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _soft_threshold(x: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def solve_gtv_destriping(
    D: np.ndarray,
    lambda_x: float = 3.5,
    lambda_y: float = 0.5,
    sigma_g: float = 8.0,
    alpha: float = 1.0,
    rho: float = 1.0,
    max_iter: int = 40,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Guided Total Variation destriping via ADMM.

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    lambda_x : float
        Regularization weight on horizontal guided gradient.
    lambda_y : float
        Regularization weight on vertical gradient.
    sigma_g : float
        Sensitivity scale for structural edge detection.
    alpha : float
        Nonlinear weighting exponent.
    rho : float
        ADMM penalty parameter.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    C : np.ndarray
        Restored clean image estimate.
    S : np.ndarray
        Estimated stripe noise residual.
    """
    img = D.astype(np.float64)
    H, W = img.shape

    # Compute guidance weights from vertical gradient of D (smoothed to resist noise)
    D_smooth = ndimage.gaussian_filter(img, sigma=1.0)
    grad_y_guidance = np.zeros_like(img)
    grad_y_guidance[:-1, :] = np.abs(D_smooth[1:, :] - D_smooth[:-1, :])
    grad_y_guidance[-1, :] = grad_y_guidance[-2, :]

    # W_x in (0, 1]: close to 1 in flat/stripe regions, close to 0 near strong edges
    Wx = np.exp(-((grad_y_guidance / sigma_g) ** alpha))
    Wx = np.clip(Wx, 0.05, 1.0)

    # FFT eigenvalues for periodic forward difference operators
    fx = np.zeros((H, W), dtype=np.complex128)
    fx[0, 0] = -1.0
    fx[0, 1] = 1.0
    eig_fx = np.fft.fft2(fx)

    fy = np.zeros((H, W), dtype=np.complex128)
    fy[0, 0] = -1.0
    fy[1, 0] = 1.0
    eig_fy = np.fft.fft2(fy)

    eig_laplacian = rho * (np.abs(eig_fx) ** 2 + np.abs(eig_fy) ** 2)
    denom = 1.0 + eig_laplacian

    # Variables
    C = np.copy(img)
    Vx = np.zeros_like(img)
    Vy = np.zeros_like(img)
    Ux = np.zeros_like(img)
    Uy = np.zeros_like(img)

    for _ in range(max_iter):
        C_prev = np.copy(C)

        # 1. Update C via FFT
        rhs = img + rho * (
            np.fft.ifft2(np.conj(eig_fx) * np.fft.fft2(Vx - Ux)).real
            + np.fft.ifft2(np.conj(eig_fy) * np.fft.fft2(Vy - Uy)).real
        )
        C = np.fft.ifft2(np.fft.fft2(rhs) / denom).real

        # 2. Forward gradients
        grad_x_C = np.fft.ifft2(eig_fx * np.fft.fft2(C)).real
        grad_y_C = np.fft.ifft2(eig_fy * np.fft.fft2(C)).real

        # 3. Soft-thresholding for guided auxiliary variables
        thresh_x = (lambda_x * Wx) / rho
        thresh_y = lambda_y / rho

        Vx = _soft_threshold(grad_x_C + Ux, thresh_x)
        Vy = _soft_threshold(grad_y_C + Uy, thresh_y)

        # 4. Multipliers
        Ux += grad_x_C - Vx
        Uy += grad_y_C - Vy

        diff = np.linalg.norm(C - C_prev) / (np.linalg.norm(C_prev) + 1e-8)
        if diff < tol:
            break

    C = np.clip(C, 0.0, 255.0)
    S = img - C
    return C, S
