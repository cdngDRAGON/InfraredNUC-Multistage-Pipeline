"""mod02_stage1_admm_destriping/src/operators.py

2D Discrete Finite-Difference Linear Operators and Exact Adjoints.

Implements periodic boundary conditions ensuring:
1. Exact adjoint identity: <A(x), y> == <x, A^T(y)> to machine precision (1e-12).
2. Block Circulant with Circulant Blocks (BCCC) structure diagonalized by 2D FFT.
3. Fast optical transfer function (OTF) eigenvalue calculation for closed-form ADMM inversion.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def grad_x(u: np.ndarray) -> np.ndarray:
    """Forward horizontal gradient with periodic boundary condition.

    grad_x(u)_{i, j} = u_{i, j+1} - u_{i, j}
    with periodic wrap-around at j = N - 1.
    """
    return np.roll(u, -1, axis=1) - u


def grad_x_adj(v: np.ndarray) -> np.ndarray:
    """Exact mathematical adjoint of grad_x (negative backward divergence).

    <grad_x(u), v> = <u, grad_x_adj(v)>
    grad_x_adj(v)_{i, j} = v_{i, j-1} - v_{i, j}
    """
    return np.roll(v, 1, axis=1) - v


def div_x(v: np.ndarray) -> np.ndarray:
    """Alias for grad_x_adj."""
    return grad_x_adj(v)


def grad_y(u: np.ndarray) -> np.ndarray:
    """Forward vertical gradient with periodic boundary condition.

    grad_y(u)_{i, j} = u_{i+1, j} - u_{i, j}
    with periodic wrap-around at i = M - 1.
    """
    return np.roll(u, -1, axis=0) - u


def grad_y_adj(v: np.ndarray) -> np.ndarray:
    """Exact mathematical adjoint of grad_y (negative backward divergence).

    <grad_y(u), v> = <u, grad_y_adj(v)>
    grad_y_adj(v)_{i, j} = v_{i-1, j} - v_{i, j}
    """
    return np.roll(v, 1, axis=0) - v


def div_y(v: np.ndarray) -> np.ndarray:
    """Alias for grad_y_adj."""
    return grad_y_adj(v)


def d2_dx2(u: np.ndarray) -> np.ndarray:
    """Second-order horizontal finite difference with periodic boundary condition.

    d2_dx2(u)_{i, j} = u_{i, j+1} - 2*u_{i, j} + u_{i, j-1}
    """
    return np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)


def d2_dx2_adj(v: np.ndarray) -> np.ndarray:
    """Exact mathematical adjoint of d2_dx2.

    Because d2_dx2 is self-adjoint under periodic boundary conditions:
    d2_dx2_adj(v) = d2_dx2(v)
    """
    return d2_dx2(v)


def compute_fft_eigenvalues(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the squared magnitudes of FFT eigenvalues for grad_x, grad_y, and d2_dx2.

    For an (M, N) 2D grid with spatial frequencies:
        omega_y = 2 * pi * u / M,  u in [0, M-1]
        omega_x = 2 * pi * v / N,  v in [0, N-1]

    Eigenvalues:
        |F(grad_x)|^2 = 4 * sin^2(omega_x / 2)
        |F(grad_y)|^2 = 4 * sin^2(omega_y / 2)
        |F(d2_dx2)|^2 = (|F(grad_x)|^2)^2 = 16 * sin^4(omega_x / 2)

    Returns
    -------
    eig_gx_sq : np.ndarray
        |F(grad_x)|^2 of shape (M, N).
    eig_gy_sq : np.ndarray
        |F(grad_y)|^2 of shape (M, N).
    eig_hxx_sq : np.ndarray
        |F(d2_dx2)|^2 of shape (M, N).
    """
    M, N = shape

    v_idx = np.arange(N, dtype=np.float64)[None, :]
    u_idx = np.arange(M, dtype=np.float64)[:, None]

    omega_x = 2.0 * np.pi * v_idx / float(N)
    omega_y = 2.0 * np.pi * u_idx / float(M)

    # Squared magnitude of forward differences
    eig_gx_sq = 4.0 * (np.sin(omega_x / 2.0) ** 2)
    eig_gy_sq = 4.0 * (np.sin(omega_y / 2.0) ** 2)

    # Second derivative eigenvalue squared magnitude
    eig_hxx_sq = eig_gx_sq ** 2

    return eig_gx_sq, eig_gy_sq, eig_hxx_sq
