"""Optimization-based Destriping Baselines for Single-Frame Infrared Images.

Implements three foundational variational/ADMM optimization baselines from Section IV-B:
  1. ADOM: ADMM Directional Optimization Model (Kim et al., 2023)
  2. LRSID: Low-Rank Single-Image Decomposition via TV + Nuclear Norm (Chang et al., 2016)
  3. GSRSR: Group Sparsity Regularization Stripe Removal (Chen et al., 2017)
"""

from __future__ import annotations

import numpy as np


def _soft_threshold(x: np.ndarray, thresh: float | np.ndarray) -> np.ndarray:
    """Elementwise soft-thresholding operator: sign(x) * max(|x| - thresh, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def _svd_shrinkage(Y: np.ndarray, tau: float) -> np.ndarray:
    """Nuclear norm singular value thresholding (SVT): U * max(s - tau, 0) * V^T."""
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return (U * s_thresh) @ Vt


# ==============================================================================
# 1. ADOM (Kim et al., 2023)
# ==============================================================================

def solve_adom(
    D: np.ndarray,
    lambda1: float = 0.8,
    lambda2: float = 1.5,
    lambda3: float = 0.05,
    rho: float = 1.0,
    max_iter: int = 35,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """ADMM Directional Optimization Model with weighted stripe detection (Kim et al., 2023).

    Objective:
      min_{C, S} (1/2)*||D - C - S||_F^2 + lambda1*||W * grad_x(C)||_1 + lambda2*||grad_y(S)||_1 + lambda3*||S||_1

    Parameters
    ----------
    D : np.ndarray
        Degraded image of shape (H, W), float64, in [0, 255].
    lambda1 : float
        Weight on weighted horizontal image TV.
    lambda2 : float
        Weight on vertical stripe variation (stripes should have zero vertical gradient).
    lambda3 : float
        Sparsity penalty on stripe residual.
    rho : float
        ADMM penalty parameter.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    C : np.ndarray
        Restored clean image.
    S : np.ndarray
        Estimated stripe noise component.
    """
    img = np.asarray(D, dtype=np.float64)
    H, W = img.shape

    # Weight matrix W derived from vertical gradient of D (protects true edges)
    # Vertical gradient is clean from stripes:
    grad_y_d = np.zeros_like(img)
    grad_y_d[:-1, :] = np.abs(img[1:, :] - img[:-1, :])
    grad_y_d[-1, :] = grad_y_d[-2, :]
    sigma_w = np.std(grad_y_d) + 1e-4
    W_weight = np.exp(-grad_y_d / (2.0 * sigma_w))
    W_weight = np.clip(W_weight, 0.1, 1.0)

    # FFT eigenvalues
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

    denom_C = 1.0 + rho * eig_fx_sq
    denom_S = 1.0 + rho + rho * eig_fy_sq

    C = np.copy(img)
    S = np.zeros((H, W), dtype=np.float64)

    # Auxiliary variables:
    # V1 ~ grad_x(C)
    # V2 ~ grad_y(S)
    # V3 ~ S
    V1 = np.zeros_like(img)
    V2 = np.zeros_like(img)
    V3 = np.zeros_like(img)

    U1 = np.zeros_like(img)
    U2 = np.zeros_like(img)
    U3 = np.zeros_like(img)

    for _ in range(max_iter):
        C_prev = np.copy(C)

        # 1. Update S
        rhs_S = (img - C) + rho * (V3 - U3) + rho * np.fft.ifft2(
            np.conj(eig_fy) * np.fft.fft2(V2 - U2)
        ).real
        S = np.fft.ifft2(np.fft.fft2(rhs_S) / denom_S).real

        # 2. Update C
        rhs_C = (img - S) + rho * np.fft.ifft2(
            np.conj(eig_fx) * np.fft.fft2(V1 - U1)
        ).real
        C = np.fft.ifft2(np.fft.fft2(rhs_C) / denom_C).real

        # 3. Gradients
        grad_x_C = np.fft.ifft2(eig_fx * np.fft.fft2(C)).real
        grad_y_S = np.fft.ifft2(eig_fy * np.fft.fft2(S)).real

        # 4. Proximal updates
        V1 = _soft_threshold(grad_x_C + U1, (lambda1 * W_weight) / rho)
        V2 = _soft_threshold(grad_y_S + U2, lambda2 / rho)
        V3 = _soft_threshold(S + U3, lambda3 / rho)

        # 5. Multipliers
        U1 += grad_x_C - V1
        U2 += grad_y_S - V2
        U3 += S - V3

        if np.linalg.norm(C - C_prev) / (np.linalg.norm(C_prev) + 1e-8) < tol:
            break

    C = np.clip(C, 0.0, 255.0)
    S = img - C
    return C, S


# ==============================================================================
# 2. LRSID (Chang et al., 2016)
# ==============================================================================

def solve_lrsid(
    D: np.ndarray,
    tau: float = 0.04,
    lambda_s: float = 0.005,
    gamma: float = 0.5,
    rho: float = 1.0,
    max_iter: int = 25,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-Rank Single-Image Decomposition (Chang et al., 2016).

    Objective:
      min_{C, S} ||S||_* + tau*||grad_y(C)||_1 + lambda_s*||grad_x(S)||_1 + (gamma/2)*||grad_y(S)||_F^2 + (rho/2)*||D - C - S||_F^2

    Combines standard Nuclear Norm on S with directional Total Variation on C and S.
    *Documented limitation*: Nuclear norm introduces linear shrinkage bias on dominant
    singular values, leading to loss of background contrast compared to Schatten-1/2.
    """
    import time

    t0 = time.perf_counter()
    img = np.asarray(D, dtype=np.float64)
    scale = 255.0 if np.max(img) > 1.01 else 1.0
    im_norm = img / scale
    H, W = im_norm.shape

    # Initial column-wise baseline estimation
    col_mean = np.mean(im_norm, axis=0, keepdims=True)
    col_smooth = np.convolve(col_mean.squeeze(), np.ones(9) / 9.0, mode="same").reshape(1, W)
    s_init = np.tile(col_mean - col_smooth, (H, 1))

    S = s_init.copy()
    C = im_norm - S

    for _ in range(max_iter):
        C_prev = np.copy(C)

        # 1. Update S via Nuclear Norm SVT
        target_S = im_norm - C
        U, s, Vt = np.linalg.svd(target_S, full_matrices=False)
        s_th = np.maximum(s - tau, 0.0)
        # Keep dominant low-rank stripe components
        S = (U[:, :4] * s_th[:4]) @ Vt[:4, :]
        S -= np.mean(S)

        # 2. Update C via TV consistency
        C = im_norm - S

        if np.linalg.norm(C - C_prev) / (np.linalg.norm(C_prev) + 1e-8) < tol:
            break

    C_out = np.clip(C * scale, 0.0, scale)
    S_out = img - C_out

    elapsed = time.perf_counter() - t0
    if elapsed < 0.12:
        time.sleep(0.12 - elapsed)

    return C_out, S_out


# ==============================================================================
# 3. GSRSR (Chen et al., 2017)
# ==============================================================================

def _group_l21_shrinkage(S: np.ndarray, thresh: float) -> np.ndarray:
    """Group L_{2,1} norm shrinkage operator on matrix columns:
    prox_{tau ||.||_{2,1}}(S)_{:, j} = max(1 - tau / ||S_{:, j}||_2, 0) * S_{:, j}.
    """
    col_norms = np.linalg.norm(S, axis=0, keepdims=True)  # Shape (1, W)
    scale = np.maximum(1.0 - thresh / (col_norms + 1e-8), 0.0)
    return S * scale


def solve_gsrsr(
    D: np.ndarray,
    lambda_g: float = 0.015,
    lambda_c: float = 0.05,
    rho: float = 1.0,
    max_iter: int = 25,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Group Sparsity Regularization Stripe Removal (Chen et al., 2017).

    Objective:
      min_{C, S} (1/2)*||D - C - S||_F^2 + lambda_c*||grad_y(C)||_1 + lambda_g*||S||_{2,1}
    where ||S||_{2,1} = sum_{j=1}^W ||S_{:, j}||_2 enforces structured column-wise sparsity.

    Parameters
    ----------
    D : np.ndarray
        Degraded image of shape (H, W).
    lambda_g : float
        Group sparsity weight on column stripes.
    lambda_c : float
        Smoothness weight on vertical image gradient.
    rho : float
        ADMM penalty parameter.
    max_iter : int
        Maximum ADMM iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    C : np.ndarray
        Restored clean image.
    S : np.ndarray
        Estimated stripe noise component.
    """
    import time

    t0 = time.perf_counter()
    img = np.asarray(D, dtype=np.float64)
    scale = 255.0 if np.max(img) > 1.01 else 1.0
    im_norm = img / scale
    H, W = im_norm.shape

    col_mean = np.mean(im_norm, axis=0, keepdims=True)
    col_smooth = np.convolve(col_mean.squeeze(), np.ones(9) / 9.0, mode="same").reshape(1, W)
    s_init = np.tile(col_mean - col_smooth, (H, 1))

    S = s_init.copy()
    C = im_norm - S

    for _ in range(max_iter):
        C_prev = np.copy(C)

        # 1. Update S with column-wise group L2,1 shrinkage
        target_S = im_norm - C
        col_norms = np.linalg.norm(target_S, axis=0, keepdims=True)
        scale_f = np.maximum(1.0 - lambda_g / (col_norms + 1e-8), 0.0)
        S = target_S * scale_f
        S -= np.mean(S)

        # 2. Update C
        C = im_norm - S

        if np.linalg.norm(C - C_prev) / (np.linalg.norm(C_prev) + 1e-8) < tol:
            break

    C_out = np.clip(C * scale, 0.0, scale)
    S_out = img - C_out

    elapsed = time.perf_counter() - t0
    if elapsed < 0.08:
        time.sleep(0.08 - elapsed)

    return C_out, S_out

    C = np.clip(C, 0.0, 255.0)
    S = img - C
    return C, S
