"""Total Variation (TV) Regularization Destriping Baseline.

Implements standard Total Variation destriping for single-frame infrared images:
    min_C (1/2) * ||C - D||_F^2 + lambda_tv * (||grad_x(C)||_1 + ||grad_y(C)||_1)

Solved via Chambolle's dual projection / proximal gradient algorithm.
"""

from __future__ import annotations

import numpy as np


def solve_tv_destriping(
    D: np.ndarray,
    lambda_tv: float = 12.0,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve TV destriping via Chambolle's dual projection method.

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    lambda_tv : float
        Regularization trade-off parameter controlling total variation penalty.
    max_iter : int
        Maximum number of dual ascent iterations.
    tol : float
        Relative convergence tolerance on the dual variables.

    Returns
    -------
    C : np.ndarray
        Restored clean image estimate of shape (H, W).
    S : np.ndarray
        Estimated stripe noise residual (D - C).
    """
    img = D.astype(np.float64)
    H, W = img.shape

    # Dual variables (p_x, p_y) corresponding to horizontal and vertical gradients
    px = np.zeros_like(img)
    py = np.zeros_like(img)

    # Step size for dual gradient ascent (Chambolle tau <= 1/8 for 2D)
    tau = 0.24

    for _ in range(max_iter):
        # Compute divergence of dual vector p: div(p) = d_x^T p_x + d_y^T p_y
        div_p = np.zeros_like(img)

        # Horizontal divergence: px(x, y) - px(x-1, y)
        div_p[:, :-1] += px[:, :-1]
        div_p[:, 1:] -= px[:, :-1]

        # Vertical divergence: py(x, y) - py(x, y-1)
        div_p[:-1, :] += py[:-1, :]
        div_p[1:, :] -= py[:-1, :]

        # Primal estimate: C = D - lambda_tv * div(p)
        C_est = img - lambda_tv * div_p

        # Forward gradients of C_est
        grad_x = np.zeros_like(img)
        grad_x[:, :-1] = C_est[:, 1:] - C_est[:, :-1]

        grad_y = np.zeros_like(img)
        grad_y[:-1, :] = C_est[1:, :] - C_est[:-1, :]

        # Dual ascent step
        px_new = px + (tau / lambda_tv) * grad_x
        py_new = py + (tau / lambda_tv) * grad_y

        # Projection onto L_infty ball: ||(px, py)||_2 <= 1
        norm_p = np.maximum(1.0, np.sqrt(px_new**2 + py_new**2))
        px_next = px_new / norm_p
        py_next = py_new / norm_p

        diff = np.sqrt(np.mean((px_next - px) ** 2 + (py_next - py) ** 2))
        px = px_next
        py = py_next

        if diff < tol:
            break

    # Final primal reconstruction
    div_p = np.zeros_like(img)
    div_p[:, :-1] += px[:, :-1]
    div_p[:, 1:] -= px[:, :-1]
    div_p[:-1, :] += py[:-1, :]
    div_p[1:, :] -= py[:-1, :]

    C = np.clip(img - lambda_tv * div_p, 0.0, 255.0)
    S = img - C
    return C, S
