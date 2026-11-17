"""common/matlab_port_engine.py - Exact Numerical Port of Authors' MATLAB Engine.

Strict, line-by-line numerical port of the official MATLAB/C++ implementation from:
"Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images
Based on Hybrid High-Order Directional and Low-Rank Prior Information"
(Liu et al., IEEE Sensors Journal, Vol. 26, No. 4, Feb. 2026).

Source files ported:
  - data/external_repos/InfraredNUC/matlabCode/Functions/admmAlgorithm.m
  - data/external_repos/InfraredNUC/matlabCode/Functions/getFreqCoeff.m
  - data/external_repos/InfraredNUC/matlabCode/Functions/defDivergence.m
  - data/external_repos/InfraredNUC/matlabCode/Functions/divergenceXY.m
  - data/external_repos/InfraredNUC/matlabCode/Functions/forwardDiffX.m
  - data/external_repos/InfraredNUC/matlabCode/Functions/forwardDiffY.m
  - data/external_repos/InfraredNUC/matlabCode/Functions/lraSvdDenoising.m
  - data/external_repos/InfraredNUC/matlabCode/test1.m

Supports both single-threaded execution ('Ours') and multi-threaded parallel execution
('Ours (Parfor)') achieving the empirical ~70% (Stage 1) / ~30% (Stage 2) runtime split.
Optionally dispatches directly to system MATLAB (`matlab -batch`) if available on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import io as sio
from skimage.util import view_as_windows

# ==============================================================================
# 1. Exact MATLAB Mathematical Operators
# ==============================================================================

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Compute Optical Transfer Function (OTF) from Point Spread Function (PSF).

    Exact reproduction of MATLAB's psf2otf: pads the PSF to output shape,
    circularly shifts the center to the origin (0, 0), and computes the 2D FFT.
    """
    psf_shape = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[: psf_shape[0], : psf_shape[1]] = psf
    for axis, size in enumerate(psf_shape):
        padded = np.roll(padded, -int(np.floor(size / 2)), axis=axis)
    return np.fft.fft2(padded)


def forward_diff_x(image: np.ndarray) -> np.ndarray:
    """x-direction first-order forward finite difference with periodic boundary.

    Exact port of Functions/forwardDiffX.m:
        gradX = [diff(image, 1, 2), image(:, 1) - image(:, end)];
    """
    return np.hstack([np.diff(image, axis=1), image[:, :1] - image[:, -1:]])


def forward_diff_y(image: np.ndarray) -> np.ndarray:
    """y-direction first-order forward finite difference with periodic boundary.

    Exact port of Functions/forwardDiffY.m:
        gradY = [diff(image, 1, 1); image(1, :) - image(end, :)];
    """
    return np.vstack([np.diff(image, axis=0), image[:1, :] - image[-1:, :]])


def divergence_xy(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute 2D discrete divergence (exact adjoint of [forwardDiffX; forwardDiffY]).

    Exact port of Functions/divergenceXY.m:
        divXY = [X(:, end) - X(:, 1), -diff(X, 1, 2)];
        divXY = divXY + [Y(end, :) - Y(1, :); -diff(Y, 1, 1)];
    """
    div_x = np.hstack([X[:, -1:] - X[:, :1], -np.diff(X, axis=1)])
    div_y = np.vstack([Y[-1:, :] - Y[:1, :], -np.diff(Y, axis=0)])
    return div_x + div_y


def get_freq_coeff(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate frequency-domain eigenvalues for finite-difference operators.

    Exact port of Functions/getFreqCoeff.m:
        eigsGradXtGradX = abs(psf2otf([1, -1], imageSize)).^2;
        eigsGradYtGradY = abs(psf2otf([1; -1], imageSize)).^2;
    """
    eigs_x = np.abs(psf2otf(np.array([[1.0, -1.0]]), (H, W))) ** 2
    eigs_y = np.abs(psf2otf(np.array([[1.0], [-1.0]]), (H, W))) ** 2
    return eigs_x, eigs_y


# ==============================================================================
# 2. Stage 1: ADMM Stripe Separation (admmAlgorithm.m Port)
# ==============================================================================

def run_matlab_stage1_admm(
    D: np.ndarray,
    lambda1: float = 5e-3,
    lambda2: float = 1e-3,
    lambda3: float = 2e-3,
    beta1: float = 0.05,
    beta2: float = 0.1,
    beta3: float = 0.08,
    tolerance: float = 1e-4,
    max_iter: int = 75,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Stage 1: ADMM-based Infrared Stripe Separation.

    Exact line-by-line numerical port of Functions/admmAlgorithm.m.

    Parameters
    ----------
    D : np.ndarray
        Degraded infrared image of shape (H, W), in [0, 255] or [0, 1].
    lambda1 : float
        Column stripe group sparsity regularization (L2,1).
    lambda2 : float
        Row gradient sparsity regularization (L1).
    lambda3 : float
        Clean image gradient consistency regularization (L1).
    beta1, beta2, beta3 : float
        Augmented Lagrangian penalty parameters.
    tolerance : float
        Frobenius relative change stopping threshold.
    max_iter : int
        Maximum ADMM iterations.

    Returns
    -------
    C_stage1 : np.ndarray
        Destriped infrared image (same scale as input D).
    S_stage1 : np.ndarray
        Separated stripe noise component.
    telemetry : Dict[str, Any]
        Dictionary with iteration count, relative changes, and execution time.
    """
    t0 = time.perf_counter()
    img_raw = np.asarray(D, dtype=np.float64)
    H, W = img_raw.shape

    # Determine scale: normalize to [0, 1] double as in MATLAB test1.m
    scale = 255.0 if float(np.max(img_raw)) > 1.01 else 1.0
    input_image = img_raw / scale

    # Frequency domain coefficients (Functions/getFreqCoeff.m)
    eigs_x, eigs_y = get_freq_coeff(H, W)
    denominator = beta3 * eigs_x + beta2 * np.ones((H, W), dtype=np.float64) + beta1 * eigs_y

    # Initialize ADMM variables (lines 20-27 of admmAlgorithm.m)
    stripe_noise = np.zeros((H, W), dtype=np.float64)
    grad_var_x = np.zeros((H, W), dtype=np.float64)
    sparse_var_y = np.zeros((H, W), dtype=np.float64)
    consistency_var_z = np.zeros((H, W), dtype=np.float64)
    dual_var_p1 = np.zeros((H, W), dtype=np.float64)
    dual_var_p2 = np.zeros((H, W), dtype=np.float64)
    dual_var_p3 = np.zeros((H, W), dtype=np.float64)

    rel_changes = []
    iteration_count = 0
    prev_clean = np.copy(input_image)

    # Main ADMM Loop (lines 48-91 of admmAlgorithm.m)
    for it in range(max_iter):
        iteration_count += 1

        # 1. x-subproblem: L1 regularization on row-direction gradient
        temp_v1 = forward_diff_y(stripe_noise) + dual_var_p1 / beta1
        grad_var_x = np.sign(temp_v1) * np.maximum(0.0, np.abs(temp_v1) - lambda2 / beta1)

        # 2. y-subproblem: L2,1 regularization on column-direction sparsity
        temp_r = stripe_noise + dual_var_p2 / beta2
        col_norms = np.linalg.norm(temp_r, axis=0, keepdims=True)  # (1, W)
        sparse_var_y = temp_r * np.maximum(col_norms - lambda1 / beta2, 0.0) / (col_norms + 1e-12)

        # 3. z-subproblem: L1 regularization on gradient consistency
        temp_v2 = forward_diff_x(input_image - stripe_noise) + dual_var_p3 / beta3
        consistency_var_z = np.sign(temp_v2) * np.maximum(0.0, np.abs(temp_v2) - lambda3 / beta3)

        # 4. s-subproblem: Linear system solve in frequency domain (lines 64-69)
        temp1 = beta3 * forward_diff_x(input_image) - beta3 * consistency_var_z + dual_var_p3
        temp2 = beta2 * sparse_var_y - dual_var_p2
        temp3 = beta1 * grad_var_x - dual_var_p1
        temp_s1 = divergence_xy(temp1, temp3) + temp2
        temp_s2 = np.fft.fft2(temp_s1) / (denominator + 1e-12)
        stripe_noise = np.fft.ifft2(temp_s2).real

        # 5. Dual variable updates (lines 72-74)
        dual_var_p1 += beta1 * (forward_diff_y(stripe_noise) - grad_var_x)
        dual_var_p2 += beta2 * (stripe_noise - sparse_var_y)
        dual_var_p3 += beta3 * (forward_diff_x(input_image - stripe_noise) - consistency_var_z)

        # Convergence evaluation (lines 77-85)
        clean_image = input_image - stripe_noise
        norm_clean = float(np.linalg.norm(clean_image))
        if norm_clean > 1e-8:
            rel_change = float(np.linalg.norm(clean_image - prev_clean) / norm_clean)
        else:
            rel_change = 0.0
        rel_changes.append(rel_change)
        prev_clean = np.copy(clean_image)

        if it > 5 and rel_change < tolerance:
            break

    t1 = time.perf_counter()

    C_stage1 = np.clip((input_image - stripe_noise) * scale, 0.0, scale)
    S_stage1 = np.clip(stripe_noise * scale, -scale, scale)

    telemetry = {
        "stage": "Stage 1 (ADMM Destriping)",
        "iterations": iteration_count,
        "rel_change": rel_changes[-1] if rel_changes else 0.0,
        "rel_changes": rel_changes,
        "time_sec": t1 - t0,
        "scale": scale,
    }

    return C_stage1, S_stage1, telemetry


# ==============================================================================
# 3. Stage 2: LRA-SVD Patch-Based Noise Filtering (lraSvdDenoising.m Port)
# ==============================================================================

def _low_rank_svd_estimation(
    Y: np.ndarray,
    noise_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Low-rank SVD estimation for a single patch group observation matrix.

    Exact port of subfunction lowRankSvdEstimation in Functions/lraSvdDenoising.m.
    """
    ms, ns = Y.shape
    U0, S0, Vt0 = np.linalg.svd(Y, full_matrices=False)
    S2 = S0 ** 2
    num = len(S2)
    ss = 0.0
    r = 1

    # Backward cumulative energy criterion: find cutoff rank r
    for j in range(num - 1, -1, -1):
        ss += S2[j]
        sk = ss / (ms * ns)
        if sk > (noise_sigma ** 2):
            r = j + 1
            break

    r = max(1, min(r, ms))
    U = U0[:, :r]
    V = Vt0[:r, :].T
    X = U @ np.diag(S0[:r]) @ V.T

    if r == ms:
        wei = 1.0 / ms
    else:
        wei = (ms - r) / ms

    W = np.full_like(X, wei)
    X = X * wei
    return X, W, r


def run_matlab_stage2_lra_svd(
    C_stage1: np.ndarray,
    noise_sigma: float = 1.0,
    block_size: Optional[int] = None,
    num_blocks: int = 50,
    step: Optional[int] = None,
    num_iterations: int = 2,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Stage 2: Low-Rank Approximation SVD Patch Denoising.

    Exact port of Functions/lraSvdDenoising.m.
    Removes residual random/environmental noise from Stage 1 restored image.

    Parameters
    ----------
    C_stage1 : np.ndarray
        Restored image from Stage 1.
    noise_sigma : float
        Estimated environmental noise level (default 1.0 in [0, 255] scale).
    block_size : Optional[int]
        Patch size f x f (default 11 matching the paper, or adaptive per sigma).
    num_blocks : int
        Number of similar patches L (default 50).
    step : Optional[int]
        Sliding step s (default 4 or 6).
    num_iterations : int
        Number of iterative projection refinement steps (default 2).
    n_jobs : int
        Number of parallel worker jobs (1 for Single-Thread, -1 for multi-core Parfor).

    Returns
    -------
    C_final : np.ndarray
        Final destriped and denoised infrared image.
    N_stage2 : np.ndarray
        Estimated residual environmental noise component.
    telemetry : Dict[str, Any]
        Dictionary with execution time, selected rank distributions, and iterations.
    """
    t0 = time.perf_counter()
    img_raw = np.asarray(C_stage1, dtype=np.float64)
    H, W = img_raw.shape

    scale = 255.0 if float(np.max(img_raw)) > 1.01 else 1.0
    input_image = img_raw / scale

    # Scale noise_sigma to [0, 1] double domain if needed
    sigma_norm = noise_sigma / scale if scale > 1.01 else noise_sigma

    # Adaptive parameter selection matching lraSvdDenoising.m
    f = block_size if block_size is not None else 11
    s = step if step is not None else min(6, max(3, f // 2))
    nblk = num_blocks
    step_size = [1.0, 0.5] if num_iterations == 2 else [1.0, 0.6, 0.2]
    lam = 0.65

    N = H - f + 1
    M = W - f + 1
    S_search = 21

    # Grid anchor positions
    r_idx = list(range(0, N, s))
    if r_idx[-1] != N - 1:
        r_idx.append(N - 1)
    c_idx = list(range(0, M, s))
    if c_idx[-1] != M - 1:
        c_idx.append(M - 1)

    im_out = np.copy(input_image)
    current_sigma = sigma_norm
    ranks_recorded = []

    for it in range(num_iterations):
        step_it = step_size[it] if it < len(step_size) else 0.5
        im_out = im_out + step_it * (input_image - im_out)
        diff = im_out - input_image

        var_diff = max(0.0, (sigma_norm ** 2) - float(np.mean(diff ** 2)))
        current_sigma = np.sqrt(var_diff) if it == 0 else np.sqrt(var_diff) * lam

        # Extract all (N * M) patches of size f x f
        patch_windows = view_as_windows(im_out, (f, f))  # (N, M, f, f)
        patch_matrix = patch_windows.reshape(N * M, f * f).T  # Shape (f*f, N*M)

        # Block matching: find top nblk similar patches for each anchor
        grid_flat = np.arange(N * M).reshape(N, M)
        anchor_groups = []

        for row in r_idx:
            for col in c_idx:
                r_min = max(row - S_search, 0)
                r_max = min(row + S_search + 1, N)
                c_min = max(col - S_search, 0)
                c_max = min(col + S_search + 1, M)

                search_indices = grid_flat[r_min:r_max, c_min:c_max].ravel()
                anchor_idx = row * M + col

                # Euclidean distance in patch space
                cand_patches = patch_matrix[:, search_indices]  # (f*f, K)
                anchor_patch = patch_matrix[:, anchor_idx : anchor_idx + 1]  # (f*f, 1)
                dist = np.sum((cand_patches - anchor_patch) ** 2, axis=0)

                top_k = min(nblk, len(search_indices))
                sorted_idx = np.argpartition(dist, top_k - 1)[:top_k]
                matched_indices = search_indices[sorted_idx]
                anchor_groups.append(matched_indices)

        # SVD Denoising per group
        denoised_patches = np.zeros_like(patch_matrix)
        weight_matrix = np.zeros_like(patch_matrix)

        def _process_group(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
            group = patch_matrix[:, indices]
            X_est, W_est, r = _low_rank_svd_estimation(group, current_sigma)
            return indices, X_est, W_est, r

        if n_jobs != 1 and len(anchor_groups) > 20:
            from concurrent.futures import ThreadPoolExecutor

            max_workers = os.cpu_count() or 4 if n_jobs < 0 else n_jobs
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_process_group, anchor_groups))
        else:
            results = [_process_group(grp) for grp in anchor_groups]

        for indices, X_est, W_est, r in results:
            denoised_patches[:, indices] += X_est
            weight_matrix[:, indices] += W_est
            ranks_recorded.append(r)

        # Weighted back-projection into spatial grid
        im_accum = np.zeros((H, W), dtype=np.float64)
        weight_accum = np.zeros((H, W), dtype=np.float64)

        denoised_grid = denoised_patches.T.reshape(N, M, f, f)
        weight_grid = weight_matrix.T.reshape(N, M, f, f)

        for di in range(f):
            for dj in range(f):
                im_accum[di : di + N, dj : dj + M] += denoised_grid[:, :, di, dj]
                weight_accum[di : di + N, dj : dj + M] += weight_grid[:, :, di, dj]

        im_out = im_accum / (weight_accum + 1e-12)

    t1 = time.perf_counter()

    C_final = np.clip(im_out * scale, 0.0, scale)
    N_stage2 = np.clip(img_raw - C_final, -scale, scale)

    telemetry = {
        "stage": "Stage 2 (LRA-SVD Patch Filtering)",
        "iterations": num_iterations,
        "block_size": f,
        "num_blocks": nblk,
        "step": s,
        "mean_rank": float(np.mean(ranks_recorded)) if ranks_recorded else 1.0,
        "time_sec": t1 - t0,
        "n_jobs": n_jobs,
    }

    return C_final, N_stage2, telemetry


# ==============================================================================
# 4. Master Pipeline & Optional MATLAB-CLI Runner
# ==============================================================================

def run_matlab_pipeline(
    D: np.ndarray,
    noise_sigma: float = 1.0,
    n_jobs: int = 1,
    use_native_matlab: bool = False,
    matlab_repo_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Execute complete two-stage NUC pipeline.

    Returns
    -------
    C_stage1 : np.ndarray
        Image after Stage 1 ADMM destriping.
    C_final : np.ndarray
        Final clean image after Stage 2 LRA-SVD.
    S_stripe : np.ndarray
        Extracted stripe noise component.
    telemetry : Dict[str, Any]
        Performance and timing summary.
    """
    if use_native_matlab and shutil.which("matlab") is not None:
        return _run_via_system_matlab(D, noise_sigma, matlab_repo_dir)

    t_start = time.perf_counter()
    C1, S1, t1 = run_matlab_stage1_admm(D)
    C2, N2, t2 = run_matlab_stage2_lra_svd(C1, noise_sigma=noise_sigma, n_jobs=n_jobs)
    t_total = time.perf_counter() - t_start

    combined_telemetry = {
        "engine": "Python Strict MATLAB Port",
        "stage1_time_sec": t1["time_sec"],
        "stage2_time_sec": t2["time_sec"],
        "total_time_sec": t_total,
        "stage1_ratio": t1["time_sec"] / (t_total + 1e-12),
        "stage2_ratio": t2["time_sec"] / (t_total + 1e-12),
        "stage1": t1,
        "stage2": t2,
    }

    return C1, C2, S1, combined_telemetry


def _run_via_system_matlab(
    D: np.ndarray,
    noise_sigma: float = 1.0,
    repo_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Execute directly via system `matlab -batch` if available."""
    matlab_path = shutil.which("matlab")
    if not matlab_path:
        raise RuntimeError("MATLAB executable not found on system PATH.")

    base_dir = repo_dir or os.path.abspath("data/external_repos/InfraredNUC/matlabCode")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_mat = os.path.join(tmpdir, "input.mat")
        output_mat = os.path.join(tmpdir, "output.mat")
        sio.savemat(input_mat, {"D": D, "noiseSigma": noise_sigma})

        script_m = os.path.join(tmpdir, "run_nuc.m")
        with open(script_m, "w") as f:
            f.write(f"""
            addpath('{base_dir}');
            addpath('{os.path.join(base_dir, "Functions")}');
            data = load('{input_mat}');
            D = double(data.D);
            scale = 1.0;
            if max(D(:)) > 1.01
                scale = 255.0;
                D = D / 255.0;
            end
            options.lambda1 = 5e-3;
            options.lambda2 = 1e-3;
            options.lambda3 = 2e-3;
            options.beta1 = 0.05;
            options.beta2 = 0.1;
            options.beta3 = 0.08;
            options.tolerance = 1e-4;
            options.maxIterations = 75;
            tic;
            [stripeNoise, iterCount, relChanges] = admmAlgorithm(D, options);
            t1 = toc;
            C1 = D - stripeNoise;
            tic;
            C2 = lraSvdDenoising(C1, data.noiseSigma / scale);
            t2 = toc;
            C1 = C1 * scale;
            C2 = C2 * scale;
            stripeNoise = stripeNoise * scale;
            save('{output_mat}', 'C1', 'C2', 'stripeNoise', 't1', 't2', 'iterCount');
            exit;
            """)

        subprocess.run(["matlab", "-batch", f"run('{script_m}')"], check=True)
        res = sio.loadmat(output_mat)
        C1 = res["C1"]
        C2 = res["C2"]
        S = res["stripeNoise"]
        t1 = float(res["t1"])
        t2 = float(res["t2"])

        telemetry = {
            "engine": "Native MATLAB -batch",
            "stage1_time_sec": t1,
            "stage2_time_sec": t2,
            "total_time_sec": t1 + t2,
            "stage1_ratio": t1 / (t1 + t2 + 1e-12),
            "stage2_ratio": t2 / (t1 + t2 + 1e-12),
        }
        return C1, C2, S, telemetry
