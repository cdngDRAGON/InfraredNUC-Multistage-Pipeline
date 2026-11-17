"""tests/test_web_and_mapping.py - Unit Tests for Web Workbench & Relational Mapping Pipeline."""

import io
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from common.io_mapping_manager import IOMappingManager
from common.matlab_port_engine import (
    divergence_xy,
    forward_diff_x,
    forward_diff_y,
    get_freq_coeff,
    psf2otf,
    run_matlab_pipeline,
    run_matlab_stage1_admm,
    run_matlab_stage2_lra_svd,
)
from web_workbench.server import WorkbenchRequestHandler


# ==============================================================================
# 1. Tests for MATLAB Port Engine Operators
# ==============================================================================

def test_psf2otf():
    """Verify psf2otf reproduces exact discrete frequency eigenvalues."""
    psf_x = np.array([[1.0, -1.0]])
    otf_x = psf2otf(psf_x, (16, 16))
    assert otf_x.shape == (16, 16)
    assert np.isclose(np.abs(otf_x[0, 0]) ** 2, 0.0)
    assert np.abs(otf_x[0, 1]) ** 2 > 0.0


def test_divergence_adjoint():
    """Verify divergenceXY is the exact adjoint of [forwardDiffX; forwardDiffY]."""
    rng = np.random.RandomState(42)
    A = rng.randn(12, 14)
    X = rng.randn(12, 14)
    Y = rng.randn(12, 14)

    grad_x = forward_diff_x(A)
    grad_y = forward_diff_y(A)
    div_xy = divergence_xy(X, Y)

    inner_forward = np.sum(grad_x * X) + np.sum(grad_y * Y)
    inner_adjoint = np.sum(A * div_xy)

    assert np.isclose(inner_forward, inner_adjoint, atol=1e-10)


def test_matlab_stage1_and_stage2_execution():
    """Verify Stage 1 ADMM and Stage 2 LRA-SVD execute cleanly without errors."""
    rng = np.random.RandomState(42)
    clean = np.full((32, 32), 128.0)
    stripe = np.tile(rng.normal(0, 15.0, (1, 32)), (32, 1))
    noise = rng.normal(0, 3.0, (32, 32))
    degraded = np.clip(clean + stripe + noise, 0.0, 255.0)

    C1, S1, t1 = run_matlab_stage1_admm(degraded, max_iter=10)
    assert C1.shape == (32, 32)
    assert S1.shape == (32, 32)
    assert t1["iterations"] > 0
    assert t1["time_sec"] > 0.0

    C2, N2, t2 = run_matlab_stage2_lra_svd(C1, block_size=7, step=4, num_blocks=10, n_jobs=1)
    assert C2.shape == (32, 32)
    assert N2.shape == (32, 32)
    assert t2["time_sec"] > 0.0


# ==============================================================================
# 2. Tests for IOMappingManager & Artifact Generation
# ==============================================================================

def test_io_mapping_lifecycle(tmp_path):
    """Verify end-to-end ingestion, artifact generation, and registry updates."""
    mgr = IOMappingManager(tmp_path)
    assert mgr.inputs_dir.exists()
    assert mgr.outputs_dir.exists()

    rng = np.random.RandomState(42)
    test_img = np.clip(rng.normal(128.0, 20.0, (48, 48)), 0.0, 255.0)

    # Ingest input
    run_id, in_path, img_arr = mgr.ingest_input(test_img, filename="test_sample.png", source_dataset="Test_Suite")
    assert in_path.exists()
    assert in_path.is_file()
    assert img_arr.shape == (48, 48)

    # Process and Map
    meta = mgr.process_and_map(
        run_id=run_id,
        input_arr=img_arr,
        clean_gt=test_img,
        source_dataset="Test_Suite",
        input_filename="test_sample.png",
        params={"engine": "proposed_ours", "fast_mode": True},
        run_baselines=False,
    )

    # Verify all 8 artifacts exist on disk
    run_dir = tmp_path / "uploaded_outputs" / run_id
    assert run_dir.exists()

    expected_artifacts = [
        "00_degraded_input.png",
        "01_stage1_stripe_noise.png",
        "01_stage1_stripe_noise.mat",
        "02_stage1_restored.png",
        "03_stage2_env_noise.png",
        "04_stage2_final_clean.png",
        "05_stage_progression_dashboard.png",
        "06_baseline_comparison_grid.png",
        "07_matlab_workspace.mat",
        "stage_metrics_report.csv",
        "run_metadata.json",
    ]

    for fname in expected_artifacts:
        fpath = run_dir / fname
        assert fpath.exists(), f"Missing artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"

    # Verify registry entries
    mappings = mgr.list_mappings()
    assert len(mappings) >= 1
    found = mgr.get_mapping(run_id)
    assert found is not None
    assert found["run_id"] == run_id
    assert "metrics" in found
    assert "timing" in found


# ==============================================================================
# 3. Tests for Server Endpoints
# ==============================================================================

def test_server_template_exists():
    """Verify HTML template exists and has MATLAB App Designer theme components."""
    tmpl_path = Path("web_workbench/templates/index.html")
    assert tmpl_path.exists()
    content = tmpl_path.read_text(encoding="utf-8")
    assert "MATLAB R2024a" in content
    assert "uploaded_inputs/" in content
    assert "uploaded_outputs/" in content
    assert "Stage-by-Stage Visual Pipeline" in content
    assert "Mapped Registry Explorer" in content
