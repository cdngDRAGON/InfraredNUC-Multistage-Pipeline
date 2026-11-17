"""Module 1 Source Package: Priors and Noise Simulator."""

from mod01_priors_and_noise_simulator.src.noise_simulator import (
    SimulationResult,
    generate_column_stripe,
    simulate_case1_conventional,
    simulate_case2_periodic,
    simulate_case3_enhanced,
    simulate_degradation_by_case,
)
from mod01_priors_and_noise_simulator.src.gradient_priors import (
    compute_first_order_gradients,
    compute_second_order_derivatives,
    compute_gradient_anisotropy_ratio,
    plot_fig2_horizontal_gradient_energy,
    plot_fig3_vertical_gradient_energy,
    plot_fig5_first_and_second_order_gradients,
)
from mod01_priors_and_noise_simulator.src.lowrank_svd_analysis import (
    compute_svd,
    compute_rank_k_reconstruction,
    analyze_singular_value_decay,
    generate_svd_energy_table,
    plot_fig4_svd_singular_value_distribution,
    plot_fig9_simulation_cases_3d_energy,
)

__all__ = [
    "SimulationResult",
    "generate_column_stripe",
    "simulate_case1_conventional",
    "simulate_case2_periodic",
    "simulate_case3_enhanced",
    "simulate_degradation_by_case",
    "compute_first_order_gradients",
    "compute_second_order_derivatives",
    "compute_gradient_anisotropy_ratio",
    "plot_fig2_horizontal_gradient_energy",
    "plot_fig3_vertical_gradient_energy",
    "plot_fig5_first_and_second_order_gradients",
    "compute_svd",
    "compute_rank_k_reconstruction",
    "analyze_singular_value_decay",
    "generate_svd_energy_table",
    "plot_fig4_svd_singular_value_distribution",
    "plot_fig9_simulation_cases_3d_energy",
]
