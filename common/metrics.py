"""common/metrics.py - IEEE Sensors Journal Quantitative Evaluation Metrics.

Implements all 7 evaluation metrics from Section IV-A2 (Eqs. 27-31) of:
"Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images
Based on Hybrid High-Order Directional and Low-Rank Prior Information".

Metrics:
1. PSNR (Peak Signal-to-Noise Ratio, dB)
2. SSIM (Structural Similarity Index)
3. AVGE (Average Vertical Gradient Error, Eq. 27)
4. Roughness rho (Image Roughness Index, Eq. 28)
5. ICV (Inverse Coefficient of Variation, Eq. 29)
6. AAD (Average Absolute Difference, Eq. 30)
7. MRD (Mean Relative Deviation, Eq. 31)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import signal
from skimage.metrics import structural_similarity


def _check_and_prepare_images(
    ref: np.ndarray, est: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Validate shapes and determine data range for evaluation."""
    r = np.asarray(ref, dtype=np.float64)
    e = np.asarray(est, dtype=np.float64)

    if r.shape != e.shape:
        raise ValueError(f"Shape mismatch: ref shape {r.shape} != est shape {e.shape}")

    # Determine dynamic range: 1.0 if normalized to [0, 1], else 255.0
    max_val = max(float(np.max(r)), float(np.max(e)))
    data_range = 1.0 if max_val <= 1.01 else 255.0

    return r, e, data_range


def compute_psnr(
    ref: np.ndarray,
    est: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Parameters
    ----------
    ref : np.ndarray
        Clean reference ground-truth infrared image.
    est : np.ndarray
        Estimated (corrected) infrared image.
    data_range : Optional[float]
        Peak dynamic range (default: 255.0 for [0, 255] or 1.0 for [0, 1]).

    Returns
    -------
    psnr : float
        PSNR in dB (higher indicates better fidelity).
    """
    r, e, auto_range = _check_and_prepare_images(ref, est)
    dr = data_range if data_range is not None else auto_range

    mse = float(np.mean((r - e) ** 2))
    if mse <= 1e-12:
        return float("inf")

    return float(10.0 * np.log10((dr**2) / mse))


def compute_ssim(
    ref: np.ndarray,
    est: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Parameters
    ----------
    ref : np.ndarray
        Clean reference ground-truth infrared image.
    est : np.ndarray
        Estimated (corrected) infrared image.
    data_range : Optional[float]
        Peak dynamic range (default: 255.0 for [0, 255] or 1.0 for [0, 1]).

    Returns
    -------
    ssim : float
        SSIM value in [-1, 1], where 1.0 indicates identity.
    """
    r, e, auto_range = _check_and_prepare_images(ref, est)
    dr = data_range if data_range is not None else auto_range

    val = structural_similarity(r, e, data_range=dr)
    return float(val)


def compute_avge(orig: np.ndarray, proc: np.ndarray) -> float:
    """Compute Average Vertical Gradient Error (AVGE, Eq. 27).

    Equation 27:
        AVGE = (1 / N) * sum(|grad_y(proc) - grad_y(orig)|)

    Measures fidelity of vertical high-frequency edge and detail preservation.
    Values closer to 0 indicate superior vertical structure retention without
    unintended vertical over-smoothing.

    Parameters
    ----------
    orig : np.ndarray
        Original / reference image (or degraded input).
    proc : np.ndarray
        Processed / destriped image.

    Returns
    -------
    avge : float
        Average vertical gradient discrepancy (lower is better).
    """
    r, e, dr = _check_and_prepare_images(orig, proc)

    # Vertical gradient along axis=0 (row differences)
    grad_orig_y = np.diff(r, axis=0)
    grad_proc_y = np.diff(e, axis=0)

    diff = np.abs(grad_proc_y - grad_orig_y)
    raw_avge = float(np.mean(diff))
    if raw_avge <= 1e-12:
        return 0.0
    return raw_avge


def compute_roughness_rho(img: np.ndarray) -> float:
    """Compute Image Roughness Index rho(f) (Eq. 28 / calculateRoughness.m).

    Uses the 3x3 Sobel edge filter matching the authors' MATLAB calculateRoughness.m:
        h = fspecial('sobel')
        filteredEnergy = sum(|conv2(f, h)|) + sum(|conv2(f, h^T)|)
        filteredRoughness = filteredEnergy / sum(|f|)

    Parameters
    ----------
    img : np.ndarray
        2D infrared image.

    Returns
    -------
    rho : float
        Roughness index (lower is better for destriped scenes).
    """
    f = np.asarray(img, dtype=np.float64)

    # 3x3 Sobel kernels matching fspecial('sobel') in MATLAB
    h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    edge_h = np.abs(signal.convolve2d(f, h, mode="same", boundary="symm"))
    edge_v = np.abs(signal.convolve2d(f, h.T, mode="same", boundary="symm"))

    energy = float(np.sum(edge_h) + np.sum(edge_v))
    norm = float(np.sum(np.abs(f))) + 1e-8
    rho = float(energy / norm)

    return rho


def compute_icv(
    img: np.ndarray,
    roi_slice: Optional[
        Union[
            Tuple[slice, slice],
            Tuple[int, int, int, int],
            Sequence[Tuple[int, int, int, int]],
        ]
    ] = None,
    block_size: int = 16,
) -> float:
    """Compute Inverse Coefficient of Variation (ICV, Eq. 29 / calculateICV.m).

    Equation 29:
        ICV = mu / (sigma + 1e-8)
        computed over homogeneous infrared background regions.

    Parameters
    ----------
    img : np.ndarray
        2D infrared image.
    roi_slice : Optional[ROI]
        Optional homogeneous region specification.
    block_size : int
        Block size for automatic homogeneous region detection (default 16).

    Returns
    -------
    icv : float
        Inverse Coefficient of Variation (higher is better).
    """
    arr = np.asarray(img, dtype=np.float64)

    if roi_slice is not None:
        if isinstance(roi_slice, tuple) and len(roi_slice) == 2 and isinstance(roi_slice[0], slice):
            patch = arr[roi_slice]
            mu = float(np.mean(patch))
            sigma = float(np.std(patch))
            return float(mu / (sigma + 1e-8))

        if isinstance(roi_slice, (tuple, list)) and len(roi_slice) == 4 and isinstance(roi_slice[0], (int, np.integer)):
            ymin, xmin, h, w = [int(v) for v in roi_slice]
            patch = arr[ymin : ymin + h, xmin : xmin + w]
            mu = float(np.mean(patch))
            sigma = float(np.std(patch))
            return float(mu / (sigma + 1e-8))

        if isinstance(roi_slice, Sequence):
            icv_vals = []
            for roi in roi_slice:
                ymin, xmin, h, w = [int(v) for v in roi]
                patch = arr[ymin : ymin + h, xmin : xmin + w]
                mu = float(np.mean(patch))
                sigma = float(np.std(patch))
                icv_vals.append(mu / (sigma + 1e-8))
            return float(np.mean(icv_vals)) if icv_vals else 0.0

    # Autonomous homogeneous background region detection
    h_img, w_img = arr.shape[:2]
    blocks_stats = []

    for r in range(0, h_img - block_size + 1, block_size):
        for c in range(0, w_img - block_size + 1, block_size):
            blk = arr[r : r + block_size, c : c + block_size]
            mean_val = float(np.mean(blk))
            std_val = float(np.std(blk))
            if mean_val > 1.0 and std_val > 0.01:
                blocks_stats.append((std_val, mean_val))

    if not blocks_stats:
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        return float(mu / (sigma + 1e-8))

    # Sort ascending by std to locate homogeneous regions
    blocks_stats.sort(key=lambda item: item[0])
    k = max(1, len(blocks_stats) // 10)
    selected = blocks_stats[:k]

    icvs = [mean_v / (std_v + 1e-8) for std_v, mean_v in selected]
    return float(np.mean(icvs))


def compute_aad(orig: np.ndarray, proc: np.ndarray) -> float:
    """Compute Average Absolute Difference (AAD, Eq. 30).

    Equation 30:
        AAD = (1 / (m * n)) * sum(|proc(i, j) - orig(i, j)|)

    Parameters
    ----------
    orig : np.ndarray
        Input image before NUC (or clean ground truth).
    proc : np.ndarray
        Processed image after NUC.

    Returns
    -------
    aad : float
        Average Absolute Difference (lower is better).
    """
    r, e, dr = _check_and_prepare_images(orig, proc)
    diff = np.abs(e - r)
    raw_aad = float(np.mean(diff))
    if raw_aad <= 1e-12:
        return 0.0
    return raw_aad


def compute_mrd(orig: np.ndarray, proc: np.ndarray) -> float:
    """Compute Mean Relative Deviation (MRD, Eq. 31 / calculateMRD.m).

    Equation 31:
        MRD = (1 / (m * n)) * sum(|proc(i, j) - orig(i, j)| / max(orig(i, j), 1.0))

    Parameters
    ----------
    orig : np.ndarray
        Input image before NUC (or clean ground truth).
    proc : np.ndarray
        Processed image after NUC.

    Returns
    -------
    mrd : float
        Mean Relative Deviation (lower is better).
    """
    r, e, dr = _check_and_prepare_images(orig, proc)
    denom = np.maximum(r, 1.0 if dr > 1.0 else 0.01)
    raw_mrd = float(np.mean(np.abs(e - r) / denom))
    if raw_mrd <= 1e-12:
        return 0.0
    return raw_mrd


def evaluate_all_metrics(
    ref: np.ndarray,
    est: np.ndarray,
    degraded: Optional[np.ndarray] = None,
    roi_slice: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute all 7 evaluation metrics defined in Section IV-A2 of the paper.

    Parameters
    ----------
    ref : np.ndarray
        Clean reference ground-truth infrared image.
    est : np.ndarray
        Corrected/destriped infrared image.
    degraded : Optional[np.ndarray]
        Degraded input image before processing (used for Eqs. 27, 30, 31).
    roi_slice : Optional[Any]
        Optional homogeneous region specification for ICV.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing PSNR, SSIM, AVGE, Roughness_rho, ICV, AAD, MRD.
    """
    input_img = degraded if degraded is not None else ref

    results: Dict[str, float] = {
        "PSNR": compute_psnr(ref, est),
        "SSIM": compute_ssim(ref, est),
        "AVGE": compute_avge(input_img, est),
        "Roughness_rho": compute_roughness_rho(est),
        "ICV": compute_icv(est, roi_slice=roi_slice),
        "AAD": compute_aad(input_img, est),
        "MRD": compute_mrd(input_img, est),
    }

    # If degraded input is provided, also record full-reference vs clean ground truth
    if degraded is not None:
        deg = np.asarray(degraded, dtype=np.float64)
        results["Degraded_PSNR"] = compute_psnr(ref, deg)
        results["Degraded_SSIM"] = compute_ssim(ref, deg)
        results["Degraded_Roughness_rho"] = compute_roughness_rho(deg)
        results["Degraded_ICV"] = compute_icv(deg, roi_slice=roi_slice)
        results["PSNR_gt"] = compute_psnr(ref, est)
        results["SSIM_gt"] = compute_ssim(ref, est)
        results["AAD_gt"] = compute_aad(ref, est)
        results["MRD_gt"] = compute_mrd(ref, est)

    return results
