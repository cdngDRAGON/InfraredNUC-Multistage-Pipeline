"""mod03_stage2_patch_lowrank/src/patch_matching.py

Overlapping Patch Similarity Matching & Observation Matrix Construction (Eqs. 22-23, Figs. 7-8).

Mathematical Formulation:
- Given Stage 1 restored infrared image C of size (H, W).
- Divide into overlapping B x B patches (B = 11, vectorized dimension d = 121).
- For each reference patch y_j at position (r_j, c_j):
  Search within local spatial window of size W_s x W_s (default search radius R = 15).
  Compute Euclidean similarity distance (Eq. 22):
      H(y_j, y_c) = sum_{x, y} (y_j(x, y) - y_c(x, y))^2 = ||y_j - y_c||_2^2
- Sort candidate patches in ascending order of H and select the top L = 50 most similar patches.
- Vectorize each patch to form the observation matrix G in R^{121 x L} (Eq. 23):
      G = [y_j, y_{c,1}, y_{c,2}, ..., y_{c,L-1}]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PatchGroup:
    """Container holding grouped similar patches and spatial coordinate metadata."""

    ref_coord: Tuple[int, int]                  # (row, col) of reference patch
    patch_coords: List[Tuple[int, int]]         # List of L coordinates [(r_1, c_1), ...]
    observation_matrix: np.ndarray              # G of shape (d, L) where d = B*B (121), L = 50
    distances: np.ndarray                       # Array of Euclidean distances H(y_j, y_c) of length L


def extract_sliding_patches_view(img: np.ndarray, patch_size: int = 11) -> np.ndarray:
    """Return a zero-copy strided view of all overlapping B x B patches in img.

    Parameters
    ----------
    img : np.ndarray
        2D image of shape (H, W).
    patch_size : int
        Side length of square patch (default 11).

    Returns
    -------
    patches_view : np.ndarray
        4D strided view of shape (H - B + 1, W - B + 1, B, B).
    """
    return np.lib.stride_tricks.sliding_window_view(img, (patch_size, patch_size))


def get_reference_patch_grid(
    img_shape: Tuple[int, int],
    patch_size: int = 11,
    stride: int = 3,
) -> List[Tuple[int, int]]:
    """Generate top-left coordinate grid for overlapping reference patches.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        Dimensions (H, W) of image.
    patch_size : int
        Square patch size B (default 11).
    stride : int
        Step size between reference patches (default 3 for dense overlapping coverage).

    Returns
    -------
    coords : List[Tuple[int, int]]
        List of (row, col) coordinates for reference patches.
    """
    H, W = img_shape
    max_r = H - patch_size
    max_c = W - patch_size

    r_steps = list(range(0, max_r + 1, stride))
    if r_steps[-1] != max_r:
        r_steps.append(max_r)

    c_steps = list(range(0, max_c + 1, stride))
    if c_steps[-1] != max_c:
        c_steps.append(max_c)

    coords = [(r, c) for r in r_steps for c in c_steps]
    return coords


def find_similar_patches_for_ref(
    ref_coord: Tuple[int, int],
    all_patches_view: np.ndarray,
    patch_size: int = 11,
    search_radius: int = 15,
    num_similar_patches: int = 50,
) -> PatchGroup:
    """Find top-L similar patches for a reference patch within a local search window (Eq. 22-23).

    Parameters
    ----------
    ref_coord : Tuple[int, int]
        (r_j, c_j) top-left coordinate of reference patch.
    all_patches_view : np.ndarray
        4D strided view (H_p, W_p, B, B) of all patches.
    patch_size : int
        Square patch side B = 11.
    search_radius : int
        Local search radius R (window size (2R+1) x (2R+1)). Default 15.
    num_similar_patches : int
        Number of candidate patches L to group (default 50).

    Returns
    -------
    group : PatchGroup
        Container holding observation matrix G in R^{121 x L} and spatial coordinates.
    """
    r_j, c_j = ref_coord
    H_p, W_p = all_patches_view.shape[:2]

    # Reference patch y_j
    y_j = all_patches_view[r_j, c_j]  # Shape (B, B)

    # Search window bounds in patch-coordinate space
    r_min = max(0, r_j - search_radius)
    r_max = min(H_p, r_j + search_radius + 1)
    c_min = max(0, c_j - search_radius)
    c_max = min(W_p, c_j + search_radius + 1)

    # Extract all candidate patches in window
    window_patches = all_patches_view[r_min:r_max, c_min:c_max]  # Shape (n_r, n_c, B, B)
    n_r, n_c = window_patches.shape[:2]

    # Compute Euclidean distance H(y_j, y_c) (Eq. 22)
    diff = window_patches - y_j
    dist_map = np.sum(diff * diff, axis=(-2, -1))  # Shape (n_r, n_c)
    dist_flat = dist_map.ravel()

    # Determine L: clamp if search window contains fewer than L candidates
    total_candidates = len(dist_flat)
    L = min(num_similar_patches, total_candidates)

    # Ascending sort to find the L smallest distances
    if L < total_candidates:
        top_k_indices = np.argpartition(dist_flat, L)[:L]
        # Sort the top L
        sorted_order = np.argsort(dist_flat[top_k_indices])
        best_indices = top_k_indices[sorted_order]
    else:
        best_indices = np.argsort(dist_flat)

    # Convert linear indices back to global image patch coordinates
    r_offsets = best_indices // n_c
    c_offsets = best_indices % n_c

    global_coords = [
        (int(r_min + ro), int(c_min + co))
        for ro, co in zip(r_offsets, c_offsets)
    ]

    selected_distances = dist_flat[best_indices]

    # Construct observation matrix G in R^{(B*B) x L} = R^{121 x L} (Eq. 23)
    d = patch_size * patch_size
    flattened_window = window_patches.reshape(total_candidates, d)
    G = flattened_window[best_indices].T  # Shape (121, L)

    return PatchGroup(
        ref_coord=ref_coord,
        patch_coords=global_coords,
        observation_matrix=G,
        distances=selected_distances,
    )
