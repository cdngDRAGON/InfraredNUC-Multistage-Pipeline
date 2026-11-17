"""mod02_stage1_admm_destriping/src/proximal_ista.py

Proximal and ISTA Subproblem Solvers (Eqs. 8, 17-19).

Implements:
1. ISTA L1 Soft-Thresholding:
     shrink(v, threshold) = sign(v) * max(|v| - threshold, 0)
   for auxiliary gradient variables grad_C' (Eq. 18) and C'' (Eq. 19).

2. Schatten 1/2-Norm Singular Value Shrinkage (Eqs. 8, 16, 17):
     Analytic closed-form half-thresholding on singular values:
     min_x 0.5 * (y - x)^2 + tau * |x|^{1/2}
     via the Xu et al. (IEEE TPAMI 2012) formulation.

3. Low-Rank Column Structure Projection for IRFPA stripe noise.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np


def soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """ISTA L1 Soft-Thresholding operator (shrinkage).

    shrink(v, threshold) = sign(v) * max(|v| - threshold, 0)

    Parameters
    ----------
    v : np.ndarray
        Input tensor or array.
    threshold : float
        Non-negative threshold parameter.

    Returns
    -------
    out : np.ndarray
        Thresholded array with same shape as v.
    """
    if threshold <= 0.0:
        return v
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def half_threshold_scalar(
    y: Union[float, np.ndarray], tau: float
) -> Union[float, np.ndarray]:
    """Analytical closed-form L_{1/2} half-thresholding operator.

    Solves:
        min_x  0.5 * (y - x)^2 + tau * |x|^{1/2}

    Reference:
        Z. Xu, X. Chang, F. Xu, and H. Zhang, "L_{1/2} Regularization:
        A Thresholding Representation Theory and a Fast Algorithm,"
        IEEE Trans. Neural Netw. Learn. Syst., vol. 23, no. 7, pp. 1013-1027, 2012.

    Parameters
    ----------
    y : Union[float, np.ndarray]
        Input values.
    tau : float
        Regularization penalty parameter.

    Returns
    -------
    x : Union[float, np.ndarray]
        Half-thresholded values.
    """
    if tau <= 0.0:
        return y

    # Critical threshold: T = (3 / 4) * (2 * tau)^{2/3}
    T = 0.75 * ((2.0 * tau) ** (2.0 / 3.0))

    y_arr = np.asarray(y, dtype=np.float64)
    abs_y = np.abs(y_arr)
    sign_y = np.sign(y_arr)

    # Values below or equal to T are strictly zeroed out
    mask = abs_y > T
    out = np.zeros_like(y_arr)

    if np.any(mask):
        y_active = abs_y[mask]
        # phi = arccos( (tau / 8) * (y / 3)^(-3/2) )
        arg = (tau / 8.0) * ((y_active / 3.0) ** (-1.5))
        arg = np.clip(arg, -1.0, 1.0)
        phi = np.arccos(arg)

        # x = (2 / 3) * y * ( 1 + cos( 2*pi/3 - 2*phi/3 ) )
        x_mag = (2.0 / 3.0) * y_active * (1.0 + np.cos((2.0 * np.pi / 3.0) - (2.0 / 3.0) * phi))
        out[mask] = sign_y[mask] * x_mag

    if np.isscalar(y):
        return float(out.item())
    return out


def schatten_half_shrinkage(
    matrix: np.ndarray,
    tau: float,
    rank_limit: Optional[int] = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Schatten 1/2-norm singular value shrinkage (Eqs. 8, 16, 17).

    Given matrix Y, computes SVD:
        Y = U * diag(sigma) * V^T
    applies half_threshold_scalar to sigma with threshold tau, clamps to rank_limit,
    and reconstructs the low-rank matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input 2D matrix (M, N).
    tau : float
        Schatten 1/2 regularization threshold parameter.
    rank_limit : Optional[int]
        Maximum allowed non-zero singular values (default 3 for column stripe noise).

    Returns
    -------
    low_rank_mat : np.ndarray
        Reconstructed low-rank matrix.
    s_shrunk : np.ndarray
        Shrunk singular values.
    """
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

    s_shrunk = half_threshold_scalar(s, tau)

    if rank_limit is not None and rank_limit > 0:
        if len(s_shrunk) > rank_limit:
            s_shrunk[rank_limit:] = 0.0

    # Fast truncated reconstruction using non-zero singular components
    nz_indices = np.where(s_shrunk > 1e-12)[0]
    if len(nz_indices) == 0:
        return np.zeros_like(matrix), s_shrunk

    max_nz = nz_indices[-1] + 1
    U_k = U[:, :max_nz]
    s_k = s_shrunk[:max_nz]
    Vt_k = Vt[:max_nz, :]

    low_rank_mat = (U_k * s_k) @ Vt_k
    return low_rank_mat, s_shrunk


def project_column_stripe(S: np.ndarray) -> np.ndarray:
    """Project stripe matrix onto the physical column-correlated subspace.

    S_col(i, j) = mean_i S(i, j)
    """
    H, W = S.shape
    col_profile = np.mean(S, axis=0, keepdims=True)
    return np.tile(col_profile, (H, 1))
