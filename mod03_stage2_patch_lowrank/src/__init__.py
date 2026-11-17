"""Module 3 Source Package: Stage 2 Patch Low-Rank Prior Recovery."""

from mod03_stage2_patch_lowrank.src.patch_matching import (
    PatchGroup,
    extract_sliding_patches_view,
    find_similar_patches_for_ref,
    get_reference_patch_grid,
)
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import (
    create_2d_kaiser_window,
    denoise_single_patch_group,
    denoise_stage2_patch_lowrank,
    estimate_noise_sigma_mad,
    select_adaptive_rank_k,
)

__all__ = [
    "PatchGroup",
    "extract_sliding_patches_view",
    "get_reference_patch_grid",
    "find_similar_patches_for_ref",
    "create_2d_kaiser_window",
    "estimate_noise_sigma_mad",
    "select_adaptive_rank_k",
    "denoise_single_patch_group",
    "denoise_stage2_patch_lowrank",
]
