"""Filter-based Destriping Baselines for Single-Frame Infrared Images.

Implements three representative spatial/transform-domain filtering techniques:
  1. 1D-GF: 1-D Guided Filter Destriping (Cao et al., 2016)
  2. 1D-WLS: 1-D Weighted Least Squares Destriping (Dong et al., 2018)
  3. BM3D: 3D Transform-Domain Collaborative Patch Filter (Dabov et al., 2007)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# ==============================================================================
# 1. 1D Guided Filter (1D-GF) - Cao et al. (2016)
# ==============================================================================

def _guided_filter_1d_horizontal(
    p: np.ndarray,
    I: np.ndarray,
    radius: int = 8,
    eps: float = 0.04,
) -> np.ndarray:
    """Apply 1D horizontal guided filter row-by-row.

    Parameters
    ----------
    p : np.ndarray
        Filtering input image of shape (H, W).
    I : np.ndarray
        Guidance image of shape (H, W).
    radius : int
        1D filter window radius.
    eps : float
        Regularization parameter penalizing large a coefficients.

    Returns
    -------
    q : np.ndarray
        Filtered output image of shape (H, W).
    """
    H, W = p.shape
    ksize = 2 * radius + 1
    # 1D box filter kernel along horizontal axis (axis 1)
    kernel_1d = np.ones((1, ksize), dtype=np.float64) / ksize

    mean_I = ndimage.convolve(I, kernel_1d, mode="reflect")
    mean_p = ndimage.convolve(p, kernel_1d, mode="reflect")

    mean_Ip = ndimage.convolve(I * p, kernel_1d, mode="reflect")
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = ndimage.convolve(I * I, kernel_1d, mode="reflect")
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = ndimage.convolve(a, kernel_1d, mode="reflect")
    mean_b = ndimage.convolve(b, kernel_1d, mode="reflect")

    q = mean_a * I + mean_b
    return q


def solve_1d_gf(
    D: np.ndarray,
    radius: int = 8,
    eps: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """1D Guided Filter Destriping (Cao et al. 2016).

    Separates high-frequency vertical stripe components from the low-frequency
    radiometric background by performing 1D guided smoothing along the horizontal axis.

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    radius : int
        Horizontal filter radius. Default 8.
    eps : float
        Regularization parameter. Default 0.04.

    Returns
    -------
    C : np.ndarray
        Restored infrared image in [0, 255].
    S : np.ndarray
        Estimated stripe noise residual.
    """
    img = np.asarray(D, dtype=np.float64)

    # In infrared destriping, vertical stripes are constant along columns.
    # Vertical mean profile or vertically smoothed image acts as guidance.
    # Apply vertical pre-smoothing to isolate column-correlated signal
    I_guidance = ndimage.uniform_filter1d(img, size=15, axis=0, mode="reflect")

    # Apply 1D horizontal guided filter
    base_background = _guided_filter_1d_horizontal(img, I_guidance, radius=radius, eps=eps)

    # High frequency detail containing stripes
    detail = img - base_background

    # Column-wise average of detail represents the estimated stripe component
    stripe_est = np.mean(detail, axis=0, keepdims=True)
    stripe_est = np.tile(stripe_est, (img.shape[0], 1))

    C = np.clip(img - stripe_est, 0.0, 255.0)
    S = img - C
    return C, S


# ==============================================================================
# 2. 1D Weighted Least Squares (1D-WLS) - Dong et al. (2018)
# ==============================================================================

def solve_1d_wls(
    D: np.ndarray,
    lambda_val: float = 1.5,
    alpha: float = 1.2,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D Weighted Least Squares Destriping (Dong et al. 2018).

    Solves a 1-D edge-preserving tridiagonal linear system across the mean column
    intensity profile to extract the vertical stripe offset vector:
        min_s sum_j (s_j - y_j)^2 + lambda * sum_j a_j (s_j - s_{j-1})^2

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    lambda_val : float
        Smoothness penalty parameter.
    alpha : float
        Edge sensitivity exponent.
    eps : float
        Small offset avoiding division by zero.

    Returns
    -------
    C : np.ndarray
        Restored infrared image.
    S : np.ndarray
        Estimated stripe noise component.
    """
    img = np.asarray(D, dtype=np.float64)
    H, W = img.shape

    # Compute mean vertical projection profile y in R^W
    col_profile = np.mean(img, axis=0)

    # Compute horizontal difference of profile
    diff = np.diff(col_profile)
    # Affinities: small gradient -> high affinity (smooth); large gradient -> low affinity
    a = 1.0 / (np.abs(diff) ** alpha + eps)
    a = a / np.mean(a)

    # Build tridiagonal matrix T of size (W, W): (I + lambda * L) s = y
    # Main diagonal
    diag = np.ones(W, dtype=np.float64)
    diag[0] += lambda_val * a[0]
    diag[1:-1] += lambda_val * (a[:-1] + a[1:])
    diag[-1] += lambda_val * a[-1]

    # Off diagonals
    sub_diag = -lambda_val * a
    sup_diag = -lambda_val * a

    # Solve tridiagonal system using Thomas algorithm
    # Forward elimination
    c_prime = np.zeros(W - 1, dtype=np.float64)
    d_prime = np.zeros(W, dtype=np.float64)

    c_prime[0] = sup_diag[0] / diag[0]
    d_prime[0] = col_profile[0] / diag[0]

    for i in range(1, W - 1):
        denom = diag[i] - sub_diag[i - 1] * c_prime[i - 1]
        c_prime[i] = sup_diag[i] / denom
        d_prime[i] = (col_profile[i] - sub_diag[i - 1] * d_prime[i - 1]) / denom

    denom_last = diag[-1] - sub_diag[-1] * c_prime[-1]
    d_prime[-1] = (col_profile[-1] - sub_diag[-1] * d_prime[-1]) / denom_last

    # Back substitution
    s_smoothed = np.zeros(W, dtype=np.float64)
    s_smoothed[-1] = d_prime[-1]
    for i in range(W - 2, -1, -1):
        s_smoothed[i] = d_prime[i] - c_prime[i] * s_smoothed[i + 1]

    # The stripe component is the high-frequency residual of the column profile
    stripe_1d = col_profile - s_smoothed
    S = np.tile(stripe_1d, (H, 1))

    C = np.clip(img - S, 0.0, 255.0)
    return C, S


# ==============================================================================
# 3. BM3D Collaborative Patch Filter (Dabov et al. 2007)
# ==============================================================================

def solve_bm3d(
    D: np.ndarray,
    sigma_psd: float = 12.0,
    patch_size: int = 8,
    stride: int = 4,
    search_window: int = 24,
    max_matches: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """BM3D-style Transform-Domain Collaborative Patch Filter.

    Groups mutually similar 2D patches into 3D groups, applies 3D DCT transform
    shrinkage (hard-thresholding), and inverts back with weighted aggregation.
    Represents the state-of-the-art for additive random Gaussian noise.
    *Key failure mode documented in paper*: Because BM3D treats spatially correlated
    column structures as coherent textures, it fails to extinguish vertical stripes.

    Parameters
    ----------
    D : np.ndarray
        Degraded image of shape (H, W), float64.
    sigma_psd : float
        Noise standard deviation for transform-domain thresholding.
    patch_size : int
        Square patch size (default 8).
    stride : int
        Step between reference patches.
    search_window : int
        Local search window side length.
    max_matches : int
        Maximum number of grouped similar patches.

    Returns
    -------
    C : np.ndarray
        Denoised image.
    S : np.ndarray
        Residual noise map.
    """
    from scipy.fft import dctn, idctn

    img = np.asarray(D, dtype=np.float64)
    H, W = img.shape
    B = patch_size

    canvas = np.zeros((H, W), dtype=np.float64)
    weights = np.zeros((H, W), dtype=np.float64)

    # Kaiser-like 2D weight window
    w_1d = np.hamming(B)
    w_2d = np.outer(w_1d, w_1d)

    # Threshold for 3D DCT coefficients (3 * sigma)
    thresh = 2.7 * sigma_psd

    half_search = search_window // 2

    # Step through reference patches
    for r in range(0, H - B + 1, stride):
        r_min = max(0, r - half_search)
        r_max = min(H - B + 1, r + half_search)
        for c in range(0, W - B + 1, stride):
            c_min = max(0, c - half_search)
            c_max = min(W - B + 1, c + half_search)

            ref_patch = img[r : r + B, c : c + B]

            # Find similar candidate patches
            candidates = []
            coords = []
            for rr in range(r_min, r_max, 2):
                for cc in range(c_min, c_max, 2):
                    cand = img[rr : rr + B, cc : cc + B]
                    dist = np.mean((ref_patch - cand) ** 2)
                    candidates.append((dist, (rr, cc), cand))

            # Sort by distance and select top matches
            candidates.sort(key=lambda x: x[0])
            top_matches = candidates[:max_matches]
            K = len(top_matches)

            # Stack into 3D group of shape (B, B, K)
            group_3d = np.stack([m[2] for m in top_matches], axis=-1)

            # 3D DCT
            group_dct = dctn(group_3d, norm="ortho")

            # Hard thresholding
            mask = np.abs(group_dct) > thresh
            group_dct_thresh = group_dct * mask

            # Inverse 3D DCT
            group_denoised = idctn(group_dct_thresh, norm="ortho")

            # Non-zero coefficient count weight
            n_nonzero = np.sum(mask)
            w_group = 1.0 / (n_nonzero + 1.0)

            # Aggregate back
            for k_idx, (_, (rr, cc), _) in enumerate(top_matches):
                canvas[rr : rr + B, cc : cc + B] += group_denoised[:, :, k_idx] * w_2d * w_group
                weights[rr : rr + B, cc : cc + B] += w_2d * w_group

    valid = weights > 1e-8
    C = np.copy(img)
    C[valid] = canvas[valid] / weights[valid]
    C = np.clip(C, 0.0, 255.0)
    S = img - C
    return C, S
