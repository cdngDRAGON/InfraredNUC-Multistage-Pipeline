"""Robust Principal Component Analysis (RPCA) Low-Rank + Sparse Baseline.

Implements Candès et al. (2011) RPCA solved via Inexact Augmented Lagrange Multipliers (IALM):
    min_{L, S} ||L||_* + lambda_rpca * ||S||_1   s.t.  D = L + S

Separates degraded image D into a low-rank background component L and a sparse corruption matrix S.
"""

from __future__ import annotations

import numpy as np


def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def _svd_shrinkage(Y: np.ndarray, tau: float) -> np.ndarray:
    """Singular value thresholding (SVT) operator: D_tau(Y) = U * S_tau(Sigma) * V^T."""
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return (U * s_thresh) @ Vt


def solve_rpca_destriping(
    D: np.ndarray,
    lambda_rpca: float | None = None,
    mu: float | None = None,
    rho_mu: float = 1.5,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve RPCA via Inexact Augmented Lagrange Multiplier (IALM).

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    lambda_rpca : float, optional
        Weight on sparse component. Defaults to 1 / sqrt(max(H, W)).
    mu : float, optional
        Initial penalty multiplier. Defaults to 1.25 / norm_2(D).
    rho_mu : float
        Growth factor for mu: mu_{k+1} = rho_mu * mu_k.
    max_iter : int
        Maximum IALM iterations.
    tol : float
        Convergence stopping tolerance on ||D - L - S||_F / ||D||_F.

    Returns
    -------
    L : np.ndarray
        Low-rank restored image estimate.
    S : np.ndarray
        Sparse corruption / stripe residual estimate.
    """
    img = D.astype(np.float64)
    H, W = img.shape
    norm_fro_D = np.linalg.norm(img, "fro")

    if lambda_rpca is None:
        lambda_rpca = 1.0 / np.sqrt(max(H, W))

    # Initializations
    norm_two_D = np.linalg.norm(img, 2)
    norm_inf_D = np.max(np.abs(img)) / lambda_rpca
    dual_norm = max(norm_two_D, norm_inf_D)

    # Multiplier Y
    Y = img / (dual_norm + 1e-8)

    if mu is None:
        mu = 1.25 / (norm_two_D + 1e-8)
    mu_max = mu * 1e7

    L = np.zeros_like(img)
    S = np.zeros_like(img)

    for _ in range(max_iter):
        # 1. Update L via singular value thresholding
        L = _svd_shrinkage(img - S + (1.0 / mu) * Y, 1.0 / mu)

        # 2. Update S via soft-thresholding
        S = _soft_threshold(img - L + (1.0 / mu) * Y, lambda_rpca / mu)

        # 3. Residual and multiplier update
        residual = img - L - S
        Y += mu * residual

        # 4. Check convergence
        rel_err = np.linalg.norm(residual, "fro") / (norm_fro_D + 1e-8)
        if rel_err < tol:
            break

        mu = min(mu * rho_mu, mu_max)

    L = np.clip(L, 0.0, 255.0)
    S = img - L
    return L, S
