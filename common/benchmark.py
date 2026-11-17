"""common/benchmark.py - Computational Profiling & Performance Benchmarking.

Provides a @profile_stage decorator measuring:
- Wall-clock execution time (seconds) via time.perf_counter()
- Peak resident memory allocation via tracemalloc (KB and MB)

Includes BenchmarkSuite and BenchmarkRecord for cross-method scaling and
reproducibility experiments matching IEEE Sensors Journal specifications.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import functools
import os
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
from tabulate import tabulate


@dataclass
class BenchmarkRecord:
    """Record containing timing and memory telemetry for an execution stage."""

    stage_name: str
    elapsed_sec: float
    peak_memory_kb: float
    peak_memory_mb: float
    current_memory_kb: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkSuite:
    """Registry and aggregator for multi-stage pipeline benchmarks."""

    _global_instance: Optional["BenchmarkSuite"] = None

    def __init__(self, name: str = "GlobalBenchmarkSuite") -> None:
        self.name = name
        self.records: List[BenchmarkRecord] = []

    @classmethod
    def get_global_suite(cls) -> "BenchmarkSuite":
        if cls._global_instance is None:
            cls._global_instance = BenchmarkSuite()
        return cls._global_instance

    def add_record(self, record: BenchmarkRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "stage_name",
                    "elapsed_sec",
                    "peak_memory_kb",
                    "peak_memory_mb",
                    "current_memory_kb",
                    "timestamp",
                ]
            )
        data = [r.to_dict() for r in self.records]
        return pd.DataFrame(data)

    def summary_table(self, table_format: str = "github") -> str:
        """Render a formatted ASCII/Markdown table of recorded benchmarks."""
        if not self.records:
            return "No benchmark records found."

        headers = ["Stage", "Time (s)", "Peak RAM (KB)", "Peak RAM (MB)"]
        rows = [
            [
                r.stage_name,
                f"{r.elapsed_sec:.4f}",
                f"{r.peak_memory_kb:.2f}",
                f"{r.peak_memory_mb:.4f}",
            ]
            for r in self.records
        ]
        return tabulate(rows, headers=headers, tablefmt=table_format)

    def export_csv(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)


def profile_stage(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    print_stats: bool = False,
    record_to_suite: bool = True,
    suite: Optional[BenchmarkSuite] = None,
) -> Callable[..., Any]:
    """Decorator to profile execution wall-clock time and peak memory.

    Parameters
    ----------
    _func : Optional[Callable]
        Function to wrap when called as bare @profile_stage.
    name : Optional[str]
        Custom stage label. Defaults to function name.
    print_stats : bool
        If True, prints profile telemetry immediately on stage completion.
    record_to_suite : bool
        If True, appends record to global or specified BenchmarkSuite.
    suite : Optional[BenchmarkSuite]
        Custom benchmark suite instance.

    Returns
    -------
    wrapper : Callable
        Wrapped function. Sets `wrapper.last_profile` attribute after execution.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        stage_label = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if tracemalloc was already active
            was_tracing = tracemalloc.is_tracing()
            if not was_tracing:
                tracemalloc.start()
            tracemalloc.reset_peak()

            start_wall = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                end_wall = time.perf_counter()
                current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                if not was_tracing:
                    tracemalloc.stop()

            elapsed = end_wall - start_wall
            peak_kb = peak_bytes / 1024.0
            peak_mb = peak_kb / 1024.0
            curr_kb = current_bytes / 1024.0

            record = BenchmarkRecord(
                stage_name=stage_label,
                elapsed_sec=elapsed,
                peak_memory_kb=peak_kb,
                peak_memory_mb=peak_mb,
                current_memory_kb=curr_kb,
            )

            # Attach last profile telemetry to wrapped function
            wrapper.last_profile = record  # type: ignore[attr-defined]

            if record_to_suite:
                target_suite = suite or BenchmarkSuite.get_global_suite()
                target_suite.add_record(record)

            if print_stats:
                print(
                    f"[PROFILE: {stage_label}] Time: {elapsed:.4f}s | "
                    f"Peak RAM: {peak_kb:.2f} KB ({peak_mb:.3f} MB)"
                )

            return result

        wrapper.last_profile = None  # type: ignore[attr-defined]
        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)
