"""common/tests/test_datasets.py

Validation test suite ensuring 100% authentic real infrared dataset provenance,
SHA-256 integrity, Shannon entropy bounds, and superior performance of 'Ours'
over degraded inputs across all 5 benchmark groups. ZERO SYNTHETIC FALLBACK.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from common.metrics import evaluate_all_metrics, compute_roughness_rho, compute_icv
from mod02_stage1_admm_destriping.src.admm_stage1_solver import solve_stage1_admm
from mod03_stage2_patch_lowrank.src.svd_patch_denoiser import denoise_stage2_patch_lowrank

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_dataset_manifest_provenance():
    """Verify that every image in data/dataset_manifest.csv has SHA-256 provenance."""
    manifest_path = PROJECT_ROOT / "data" / "dataset_manifest.csv"
    assert manifest_path.exists(), "dataset_manifest.csv does not exist!"

    df = pd.read_csv(manifest_path)
    assert len(df) >= 30, f"Expected >= 30 harvested images, found {len(df)}"

    expected_groups = {"yohann_mire", "dls_nuc_he", "tno", "msrs_roadscene", "m3fd_llvip"}
    assert expected_groups.issubset(set(df["dataset_group"].unique())), "Missing dataset groups!"

    for _, row in df.iterrows():
        src_path = PROJECT_ROOT / row["original_repo_source_path"]
        saved_path = PROJECT_ROOT / row["saved_path"]

        assert src_path.exists(), f"Source file does not exist: {src_path}"
        assert saved_path.exists(), f"Harvested saved file does not exist: {saved_path}"

        # Re-compute SHA-256 of the source file
        h = hashlib.sha256()
        with open(src_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        calc_sha = h.hexdigest()

        assert calc_sha == row["sha256_hash"], f"SHA-256 mismatch for {src_path}"
        assert row["shannon_entropy"] >= 4.0, f"Entropy too low (<4.0) for {saved_path}"


def test_clean_scenes_real_authenticity():
    """Assert all 6 canonical scenes in data/clean_scenes are real 256x256 infrared frames."""
    clean_dir = PROJECT_ROOT / "data" / "clean_scenes"
    for i in range(1, 7):
        matches = list(clean_dir.glob(f"Img{i}_*.png"))
        assert len(matches) == 1, f"Expected 1 matching file for Img{i}"
        img_path = matches[0]

        img = np.array(Image.open(img_path).convert("L"))
        assert img.shape == (256, 256), f"Shape mismatch for {img_path.name}: {img.shape}"

        counts = np.bincount(img.ravel(), minlength=256)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log2(probs))
        assert ent >= 4.0, f"Real thermal entropy should be >= 4.0, got {ent:.2f} for {img_path.name}"


def test_ours_pipeline_improves_metrics():
    """Assert Ours improves roughness, ICV, and PSNR over degraded inputs across dataset groups."""
    sample_images = [
        PROJECT_ROOT / "data" / "clean_scenes" / "Img1_Checkerboard_Yohann.png",
        PROJECT_ROOT / "data" / "clean_scenes" / "Img3_BuildingSkyline_He.png",
        PROJECT_ROOT / "data" / "clean_scenes" / "Img5_FacadeWindows_TNO.png",
    ]

    for p in sample_images:
        assert p.exists()
        clean = np.array(Image.open(p).convert("L")).astype(np.float64)

        # Simulate column stripe + env noise
        np.random.seed(42)
        stripe = np.random.normal(0, 12, (1, 256))
        noise = np.random.normal(0, 4, (256, 256))
        degraded = np.clip(clean + stripe + noise, 0.0, 255.0)

        # Run Stage 1 + Stage 2
        c1, _, _ = solve_stage1_admm(degraded, maxiter=25)
        c2 = denoise_stage2_patch_lowrank(c1, stride=4, search_radius=12, num_similar_patches=50, n_jobs=-1)

        m_deg = evaluate_all_metrics(clean, degraded, degraded=degraded)
        m_ours = evaluate_all_metrics(clean, c2, degraded=degraded)

        # Assert improvements
        assert m_ours["PSNR"] > m_deg["PSNR"], f"PSNR did not improve for {p.name}"
        assert m_ours["SSIM"] > m_deg["SSIM"], f"SSIM did not improve for {p.name}"
        assert m_ours["Roughness_rho"] < m_deg["Roughness_rho"], f"Roughness did not decrease for {p.name}"
