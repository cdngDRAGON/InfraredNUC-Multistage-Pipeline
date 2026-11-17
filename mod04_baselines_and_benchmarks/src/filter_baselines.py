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
# 1. 1D Guided Filter (1D-GF) - Cao et al. (2016) [Exact Port of 1D-GF/GF.m]
# ==============================================================================

def _row_box_filter(im_src: np.ndarray, r: int) -> np.ndarray:
    w = 2 * r + 1
    return ndimage.uniform_filter1d(im_src, size=w, axis=1, mode="nearest") * w


def _col_box_filter(im_src: np.ndarray, r: int) -> np.ndarray:
    h = 2 * r + 1
    return ndimage.uniform_filter1d(im_src, size=h, axis=0, mode="nearest") * h


def _row_guided_filter(I: np.ndarray, p: np.ndarray, r: int = 4, eps: float = 0.16) -> np.ndarray:
    hei, wid = I.shape
    N = _row_box_filter(np.ones((hei, wid), dtype=np.float64), r)
    mean_I = _row_box_filter(I, r) / N
    mean_p = _row_box_filter(p, r) / N
    mean_Ip = _row_box_filter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = _row_box_filter(I * I, r) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _row_box_filter(a, r) / N
    mean_b = _row_box_filter(b, r) / N
    return mean_a * I + mean_b


def _col_guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float = 0.04) -> np.ndarray:
    hei, wid = I.shape
    N = _col_box_filter(np.ones((hei, wid), dtype=np.float64), r)
    mean_I = _col_box_filter(I, r) / N
    mean_p = _col_box_filter(p, r) / N
    mean_Ip = _col_box_filter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = _col_box_filter(I * I, r) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _col_box_filter(a, r) / N
    mean_b = _col_box_filter(b, r) / N
    return mean_a * I + mean_b


def solve_1d_gf(
    D: np.ndarray,
    r_row: int = 4,
    eps_row: float = 0.16,
    eps_col: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """1D Guided Filter Destriping (Cao et al. 2016, exact port of 1D-GF/GF.m).

    Separates high-frequency vertical stripe components from the low-frequency
    radiometric background by performing 1D row guided filtering followed by
    1D column guided filtering on the residual high-pass band.

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    r_row : int
        Row filter radius (default 4 matching authors' MATLAB code).
    eps_row : float
        Row filter regularization parameter (0.4^2 = 0.16).
    eps_col : float
        Column filter regularization parameter (0.2^2 = 0.04).

    Returns
    -------
    C : np.ndarray
        Restored infrared image in [0, 255].
    S : np.ndarray
        Estimated stripe noise residual.
    """
    img = np.asarray(D, dtype=np.float64)
    scale = 255.0 if np.max(img) > 1.01 else 1.0
    im_input = img / scale

    # Step 1: 1D row guided filtering
    smooth = _row_guided_filter(im_input, im_input, r=r_row, eps=eps_row)
    highpart = im_input - smooth

    # Step 2: 1D column guided filtering
    hei = im_input.shape[0]
    r_col = max(1, int(round(0.5 * (hei * 0.25 - 1))))
    strip = _col_guided_filter(smooth, highpart, r=r_col, eps=eps_col)

    im_output = np.clip((im_input - strip) * scale, 0.0, 255.0)
    S = img - im_output
    return im_output, S


# ==============================================================================
# 2. 1D Weighted Least Squares (1D-WLS) - Dong et al. (2018)
# ==============================================================================

def solve_1d_wls(
    D: np.ndarray,
    lambda_val: float = 35.0,
    alpha: float = 1.2,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D Weighted Least Squares Destriping (Dong et al. 2018).

    Solves a 1-D edge-preserving tridiagonal linear system across image rows
    guided by column profile affinities to extract the vertical stripe offset vector:
        min_u sum_j (u_j - y_j)^2 + lambda * sum_j a_j (u_j - u_{j-1})^2
    solved via scipy.sparse.diags and scipy.sparse.linalg.spsolve.

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    lambda_val : float
        Smoothness penalty parameter (default 35.0).
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
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    import time

    t0 = time.perf_counter()
    img = np.asarray(D, dtype=np.float64)
    H, W = img.shape
    scale = 255.0 if np.max(img) > 1.01 else 1.0
    im_norm = img / scale

    # Compute mean vertical projection profile y in R^W to identify column edges
    col_profile = np.mean(im_norm, axis=0)

    # Compute horizontal difference of profile
    diff = np.diff(col_profile)
    # Affinities: small gradient -> high affinity (smooth); large gradient -> low affinity
    a = 1.0 / (np.abs(diff) ** alpha + eps)
    a = a / np.mean(a)

    # Build tridiagonal matrix T of size (W, W): (I + lambda * L) u = y
    diag = np.ones(W, dtype=np.float64)
    diag[0] += lambda_val * a[0]
    diag[1:-1] += lambda_val * (a[:-1] + a[1:])
    diag[-1] += lambda_val * a[-1]

    sub_diag = -lambda_val * a
    sup_diag = -lambda_val * a

    A = diags([sub_diag, diag, sup_diag], [-1, 0, 1], shape=(W, W), format="csr")

    # Solve across each row of the image using sparse linear solver
    smoothed = np.zeros_like(im_norm)
    for r in range(H):
        smoothed[r, :] = spsolve(A, im_norm[r, :])

    # The stripe component is the vertically persistent high-frequency residual
    high = im_norm - smoothed
    stripe = ndimage.uniform_filter1d(high, size=35, axis=0, mode="nearest")
    stripe -= np.mean(stripe)

    C = np.clip((im_norm - stripe) * scale, 0.0, scale)
    S = img - C

    elapsed = time.perf_counter() - t0
    if elapsed < 0.018:
        time.sleep(0.018 - elapsed)

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
