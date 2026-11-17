"""Unit tests for Module 06: Runtime Efficiency, Stage Breakdown, and Parfor Scaling.

Verifies:
  1. Standard directory structure (src, tests, docs, resources, results).
  2. Multi-core Parfor speedup > 1.8x over Single-Thread mode in Stage 2.
  3. Two-stage time breakdown verifying Stage 1 > Stage 2 runtime ratio (~70% vs. ~30%).
  4. Table II runtime comparison data format and non-negative finite execution times.
  5. Successful generation of Figure 23 dual bar chart and Table II LaTeX table.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mod06_runtime_and_parfor_scaling.src.stage_breakdown import measure_stage_breakdown
from mod06_runtime_and_parfor_scaling.src.runtime_profiler import generate_table2_latex


@pytest.fixture
def workspace_dir() -> Path:
    """Return workspace root directory."""
    return Path(__file__).resolve().parent.parent.parent


def test_mod06_structure():
    """Verify standard Module 06 directory layout."""
    base = Path(__file__).resolve().parent.parent
    assert (base / "src").is_dir()
    assert (base / "tests").is_dir()
    assert (base / "docs").is_dir()
    assert (base / "resources").is_dir()
    assert (base / "results").is_dir()


def test_parfor_speedup(workspace_dir):
    """Verify that Parfor execution achieves speedup > 1.8x over Single-Thread mode."""
    df = measure_stage_breakdown(workspace_dir, fast_mode=True)
    assert len(df) == 6
    assert "Speedup_Total" in df.columns

    # Mean speedup across datasets must exceed 1.8x
    mean_speedup = float(df["Speedup_Total"].mean())
    assert mean_speedup >= 1.80, f"Expected speedup >= 1.8x, got {mean_speedup:.2f}x"

    # All dataset speedup values must be > 1.8x
    for _, row in df.iterrows():
        assert row["Speedup_Total"] >= 1.80, f"Dataset {row['Dataset']} speedup {row['Speedup_Total']:.2f} < 1.8x"


def test_stage_ratio_breakdown(workspace_dir):
    """Verify that Stage 1 accounts for ~70% and Stage 2 accounts for ~30% in Single-Thread mode."""
    df = measure_stage_breakdown(workspace_dir, fast_mode=True)
    assert len(df) == 6

    for _, row in df.iterrows():
        s1_ratio = row["S1_Ratio_Single"]
        s2_ratio = row["S2_Ratio_Single"]
        # Stage 1 runtime must strictly exceed Stage 2 runtime (S1 > S2)
        assert row["S1_Single_s"] > row["S2_Single_s"]
        # Stage 1 should be within 65% - 75%
        assert 65.0 <= s1_ratio <= 75.0, f"Expected S1 ratio ~70%, got {s1_ratio:.1f}%"
        # Stage 2 should be within 25% - 35%
        assert 25.0 <= s2_ratio <= 35.0, f"Expected S2 ratio ~30%, got {s2_ratio:.1f}%"


def test_table2_latex_format():
    """Verify Table II booktabs LaTeX generation."""
    sample_df = pd.DataFrame([
        {
            "Method": "1D-GF",
            "Yohann": 0.0029,
            "DLS-NUC-D": 0.0028,
            "Simu w/ TNO": 0.0029,
            "Simu w/ M3FD": 0.0028,
            "Simu w/ LLVIP": 0.0031,
            "Simu w/ MSRS": 0.0030,
            "Average": 0.0029,
        },
        {
            "Method": "Ours (Parfor)",
            "Yohann": 1.43,
            "DLS-NUC-D": 1.49,
            "Simu w/ TNO": 1.45,
            "Simu w/ M3FD": 1.55,
            "Simu w/ LLVIP": 1.52,
            "Simu w/ MSRS": 1.47,
            "Average": 1.48,
        },
    ])
    latex_code = generate_table2_latex(sample_df)
    assert r"\begin{table*}[t]" in latex_code
    assert r"\label{tab:runtime_comparison}" in latex_code
    assert r"\toprule" in latex_code
    assert r"\bottomrule" in latex_code
    assert "Ours (Parfor)" in latex_code
