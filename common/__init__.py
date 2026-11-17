"""Common utilities, metrics, plot styles, and dataset generation for IEEE Sensors Journal NUC."""

from typing import Any

__all__ = [
    "set_ieee_style",
    "plot_2d_with_zoom_and_3d_energy",
    "compute_psnr",
    "compute_ssim",
    "compute_avge",
    "compute_roughness_rho",
    "compute_icv",
    "compute_aad",
    "compute_mrd",
    "evaluate_all_metrics",
    "profile_stage",
    "BenchmarkRecord",
    "BenchmarkSuite",
    "generate_benchmark_scenes",
    "load_scene",
    "BENCHMARK_SCENES",
]


def __getattr__(name: str) -> Any:
    if name in ("set_ieee_style", "plot_2d_with_zoom_and_3d_energy"):
        from common import plot_style
        return getattr(plot_style, name)
    elif name in (
        "compute_psnr",
        "compute_ssim",
        "compute_avge",
        "compute_roughness_rho",
        "compute_icv",
        "compute_aad",
        "compute_mrd",
        "evaluate_all_metrics",
    ):
        from common import metrics
        return getattr(metrics, name)
    elif name in ("profile_stage", "BenchmarkRecord", "BenchmarkSuite"):
        from common import benchmark
        return getattr(benchmark, name)
    elif name in ("generate_benchmark_scenes", "load_scene", "BENCHMARK_SCENES"):
        from common import dataset_generator
        return getattr(dataset_generator, name)
    raise AttributeError(f"module 'common' has no attribute '{name}'")
