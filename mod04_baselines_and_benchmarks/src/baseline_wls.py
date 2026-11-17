"""Weighted Least Squares (WLS) Filter Destriping Baseline.

Implements edge-preserving smoothing tailored for stripe suppression:
    min_C sum_p ( (C_p - D_p)^2 + lambda * ( a_{x,p} (grad_x C)_p^2 + a_{y,p} (grad_y C)_p^2 ) )

where the spatial affinity weights are defined by:
    a_{x,p} = (|grad_x D_p|^gamma + eps)^(-1)  (adapted to smooth cross-column oscillations)
    a_{y,p} = (|grad_y D_p|^gamma + eps)^(-1) * alpha_y  (attenuated to preserve vertical structure)

Solved efficiently via preconditioned conjugate gradient (PCG) on sparse Laplacian.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_wls_destriping(
    D: np.ndarray,
    lambda_wls: float = 1.8,
    alpha_y: float = 0.2,
    gamma: float = 1.2,
    eps: float = 1e-4,
    cg_tol: float = 1e-4,
    cg_maxiter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve WLS destriping via sparse linear system solver.

    Parameters
    ----------
    D : np.ndarray
        Degraded input image of shape (H, W), float64, in [0, 255].
    lambda_wls : float
        Regularization balance between data fidelity and smoothness.
    alpha_y : float
        Ratio weighting vertical vs horizontal regularization.
    gamma : float
        Affinity exponent.
    eps : float
        Small offset avoiding division by zero.
    cg_tol : float
        Conjugate gradient relative tolerance.
    cg_maxiter : int
        Maximum iterations for conjugate gradient.

    Returns
    -------
    C : np.ndarray
        Restored clean image estimate.
    S : np.ndarray
        Estimated stripe noise residual.
    """
    img = D.astype(np.float64)
    H, W = img.shape
    N = H * W

    # Horizontal and vertical forward differences for weights
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)

    # Pad differences to full image size
    dx_padded = np.pad(dx, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
    dy_padded = np.pad(dy, ((0, 1), (0, 0)), mode="constant", constant_values=0.0)

    # Affinities: high affinity = smooth more, low affinity = preserve edge
    ax = 1.0 / (np.abs(dx_padded) ** gamma + eps)
    ay = (alpha_y) / (np.abs(dy_padded) ** gamma + eps)

    # Normalize weights so mean is 1.0
    ax = ax / np.mean(ax)
    ay = ay / np.mean(ay)

    ax_flat = ax.ravel()
    ay_flat = ay.ravel()

    # Construct 5-point sparse Laplacian matrix
    # Index grid
    idx = np.arange(N).reshape((H, W))

    # Right neighbors: (i, j) with (i, j+1)
    r_from = idx[:, :-1].ravel()
    r_to = idx[:, 1:].ravel()
    w_r = -lambda_wls * ax[:, :-1].ravel()

    # Down neighbors: (i, j) with (i+1, j)
    d_from = idx[:-1, :].ravel()
    d_to = idx[1:, :].ravel()
    w_d = -lambda_wls * ay[:-1, :].ravel()

    rows = np.concatenate([r_from, r_to, d_from, d_to])
    cols = np.concatenate([r_to, r_from, d_to, d_from])
    vals = np.concatenate([w_r, w_r, w_d, w_d])

    # Build off-diagonal matrix
    A_off = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    # Diagonal entries: 1.0 - sum(off-diagonal entries in each row)
    diag_vals = 1.0 - np.array(A_off.sum(axis=1)).ravel()
    A = A_off + sp.diags(diag_vals, 0, shape=(N, N), format="csr")

    # Solve A * c = d via Conjugate Gradient
    rhs = img.ravel()
    c_flat, info = spla.cg(A, rhs, rtol=cg_tol, maxiter=cg_maxiter)

    C = np.clip(c_flat.reshape((H, W)), 0.0, 255.0)
    S = img - C
    return C, S
