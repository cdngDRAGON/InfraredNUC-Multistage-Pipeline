"""Unit tests for common/benchmark.py."""

import time
import numpy as np
import pytest
from common.benchmark import (
    BenchmarkRecord,
    BenchmarkSuite,
    profile_stage,
)


def test_profile_stage_decorator():
    """Verify @profile_stage captures positive execution time and peak memory."""
    suite = BenchmarkSuite("TestSuite")

    @profile_stage(name="TestStage", record_to_suite=True, suite=suite)
    def sample_workload():
        arr = np.ones((500, 500), dtype=np.float64)
        return arr.sum()

    res = sample_workload()
    assert res == 250000.0

    assert sample_workload.last_profile is not None
    record: BenchmarkRecord = sample_workload.last_profile
    assert record.stage_name == "TestStage"
    assert record.elapsed_sec > 0.0
    assert record.peak_memory_kb > 0.0

    assert len(suite.records) == 1
    df = suite.to_dataframe()
    assert len(df) == 1
    assert "stage_name" in df.columns

    table = suite.summary_table()
    assert "TestStage" in table


def test_bare_profile_stage_decorator():
    """Verify @profile_stage without arguments functions properly."""
    @profile_stage
    def bare_function():
        return 42

    val = bare_function()
    assert val == 42
    assert bare_function.last_profile.stage_name == "bare_function"
