"""mod01_priors_and_noise_simulator/src/noise_simulator.py

IRFPA Nonuniformity Degradation Engine (Eq. 5 & Fig. 9).

Mathematical Model:
    D(x, y) = C(x, y) + S(x, y) + N(x, y)

where:
    - C: Ground-truth clean thermal infrared image in [0, 255].
    - S: Column-correlated vertical stripe noise matrix.
         S(x, y) = s_col(x) + delta_S(x, y), with rank(S) <= 3.
    - N: Zero-mean Gaussian/environmental noise ~ N(0, sigma_env^2).
    - D: Degraded infrared observation matrix.

Provides three simulation scenarios corresponding to Fig. 9 of the paper:
    1. Case 1: Conventional FPN (aperiodic column gain/offset + ambient noise).
    2. Case 2: Periodic Stripes (harmonic readout crosstalk + random stripes + ambient noise).
    3. Case 3: Enhanced Stripes (high-intensity periodic + heavy random column stripes + strong ambient noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


@dataclass
class SimulationResult:
    """Structured container holding all degradation components of Eq. (5)."""

    case_name: str
    degraded: np.ndarray       # D in [0, 255]
    clean: np.ndarray          # C in [0, 255]
    stripe: np.ndarray         # S (column-correlated stripe component)
    noise: np.ndarray          # N (zero-mean ambient noise component)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "degraded_shape": self.degraded.shape,
            "stripe_norm": float(np.linalg.norm(self.stripe)),
            "noise_norm": float(np.linalg.norm(self.noise)),
            "params": self.params,
        }


def generate_column_stripe(
    shape: Tuple[int, int],
    stripe_profile: np.ndarray,
    vertical_drift_ratio: float = 0.02,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Broadcast 1D column stripe profile into 2D matrix with realistic subtle vertical drift.

    Parameters
    ----------
    shape : Tuple[int, int]
        Image dimensions (H, W).
    stripe_profile : np.ndarray
        1D column noise vector of length W.
    vertical_drift_ratio : float
        Magnitude of row-dependent dark-current fluctuation (keeps rank ~ 1-2).
    rng : Optional[np.random.RandomState]
        Random number generator.

    Returns
    -------
    S : np.ndarray
        2D stripe noise matrix of shape (H, W).
    """
    H, W = shape
    if len(stripe_profile) != W:
        raise ValueError(f"Stripe profile length {len(stripe_profile)} must match image width {W}")

    # Base column-correlated rank-1 structure
    S_base = np.tile(stripe_profile, (H, 1))

    # Subtle slow vertical modulation delta_S
    if vertical_drift_ratio > 0.0:
        if rng is None:
            rng = np.random.RandomState(42)
        drift_y = np.linspace(-1.0, 1.0, H)[:, None]
        delta_S = vertical_drift_ratio * np.std(stripe_profile) * drift_y * rng.normal(0, 1.0, (1, W))
        S = S_base + delta_S
    else:
        S = S_base

    return S


def simulate_case1_conventional(
    clean_image: np.ndarray,
    stripe_sigma: float = 12.0,
    env_sigma: float = 5.0,
    seed: int = 42,
    clip: bool = True,
) -> SimulationResult:
    """Case 1: Conventional FPN + Environmental Noise (Fig. 9 Case 1).

    Simulates standard IRFPA microbolometer column gain/offset disparities
    with Gaussian distribution plus ambient thermal noise.

    Parameters
    ----------
    clean_image : np.ndarray
        Clean infrared image C in [0, 255].
    stripe_sigma : float
        Standard deviation of random column stripe noise s_col(x). Default 12.0.
    env_sigma : float
        Standard deviation of zero-mean ambient noise N. Default 5.0.
    seed : int
        Deterministic random seed.
    clip : bool
        Whether to clip degraded image D to [0, 255].

    Returns
    -------
    result : SimulationResult
        Container holding (D, C, S, N).
    """
    C = np.asarray(clean_image, dtype=np.float64)
    H, W = C.shape[:2]
    rng = np.random.RandomState(seed)

    # 1. Random column FPN: s_col ~ N(0, stripe_sigma^2)
    s_col = rng.normal(0.0, stripe_sigma, W)
    S = generate_column_stripe((H, W), s_col, vertical_drift_ratio=0.015, rng=rng)

    # 2. Ambient environmental noise: N ~ N(0, env_sigma^2)
    N = rng.normal(0.0, env_sigma, (H, W))

    # 3. Observation synthesis: D = C + S + N
    D = C + S + N
    if clip:
        D = np.clip(D, 0.0, 255.0)

    return SimulationResult(
        case_name="Case 1 (Conventional FPN)",
        degraded=D,
        clean=C,
        stripe=S,
        noise=N,
        params={"stripe_sigma": stripe_sigma, "env_sigma": env_sigma, "seed": seed},
    )


def simulate_case2_periodic(
    clean_image: np.ndarray,
    stripe_amp: float = 15.0,
    freq: float = 0.18,
    env_sigma: float = 6.0,
    seed: int = 42,
    clip: bool = True,
) -> SimulationResult:
    """Case 2: Periodic Stripe + Ambient Noise (Fig. 9 Case 2).

    Simulates ROIC multiplexer crosstalk, chopper synchronization harmonics,
    and column readout pattern with mixed periodic and random variations.

    Parameters
    ----------
    clean_image : np.ndarray
        Clean infrared image C in [0, 255].
    stripe_amp : float
        Amplitude of harmonic periodic column stripe pattern. Default 15.0.
    freq : float
        Spatial frequency of periodic stripe (cycles/column). Default 0.18.
    env_sigma : float
        Standard deviation of ambient Gaussian noise. Default 6.0.
    seed : int
        Deterministic random seed.
    clip : bool
        Whether to clip degraded image D to [0, 255].

    Returns
    -------
    result : SimulationResult
        Container holding (D, C, S, N).
    """
    C = np.asarray(clean_image, dtype=np.float64)
    H, W = C.shape[:2]
    rng = np.random.RandomState(seed)

    # 1. Harmonic periodic component + mild random FPN
    x_idx = np.arange(W, dtype=np.float64)
    s_periodic = stripe_amp * np.sin(2.0 * np.pi * freq * x_idx + 0.35)
    s_random = rng.normal(0.0, 0.35 * stripe_amp, W)
    s_col = s_periodic + s_random
    S = generate_column_stripe((H, W), s_col, vertical_drift_ratio=0.02, rng=rng)

    # 2. Ambient environmental noise
    N = rng.normal(0.0, env_sigma, (H, W))

    # 3. Observation synthesis
    D = C + S + N
    if clip:
        D = np.clip(D, 0.0, 255.0)

    return SimulationResult(
        case_name="Case 2 (Periodic Stripe)",
        degraded=D,
        clean=C,
        stripe=S,
        noise=N,
        params={"stripe_amp": stripe_amp, "freq": freq, "env_sigma": env_sigma, "seed": seed},
    )


def simulate_case3_enhanced(
    clean_image: np.ndarray,
    stripe_amp: float = 28.0,
    freq: float = 0.22,
    env_sigma: float = 10.0,
    seed: int = 42,
    clip: bool = True,
) -> SimulationResult:
    """Case 3: Enhanced Severe Stripe + Strong Ambient Noise (Fig. 9 Case 3).

    Simulates extreme sensor degradation: high-amplitude multi-harmonic stripes,
    severe random column offset fluctuations, occasional high-intensity defective lines,
    and elevated environmental noise.

    Parameters
    ----------
    clean_image : np.ndarray
        Clean infrared image C in [0, 255].
    stripe_amp : float
        Amplitude of primary harmonic stripe. Default 28.0.
    freq : float
        Spatial frequency of primary harmonic. Default 0.22.
    env_sigma : float
        Standard deviation of ambient Gaussian noise. Default 10.0.
    seed : int
        Deterministic random seed.
    clip : bool
        Whether to clip degraded image D to [0, 255].

    Returns
    -------
    result : SimulationResult
        Container holding (D, C, S, N).
    """
    C = np.asarray(clean_image, dtype=np.float64)
    H, W = C.shape[:2]
    rng = np.random.RandomState(seed)

    # 1. Multi-harmonic periodic + heavy random column noise
    x_idx = np.arange(W, dtype=np.float64)
    s_harm1 = stripe_amp * np.sin(2.0 * np.pi * freq * x_idx)
    s_harm2 = 0.40 * stripe_amp * np.cos(4.0 * np.pi * freq * x_idx + 0.5)
    s_rand = rng.normal(0.0, 0.55 * stripe_amp, W)

    # Defective/dead column impulses (sparse severe spikes)
    defective_indices = rng.choice(W, size=max(2, W // 40), replace=False)
    s_impulses = np.zeros(W, dtype=np.float64)
    s_impulses[defective_indices] = rng.choice([-1.2, 1.2], size=len(defective_indices)) * stripe_amp

    s_col = s_harm1 + s_harm2 + s_rand + s_impulses
    S = generate_column_stripe((H, W), s_col, vertical_drift_ratio=0.03, rng=rng)

    # 2. Strong ambient environmental noise
    N = rng.normal(0.0, env_sigma, (H, W))

    # 3. Observation synthesis
    D = C + S + N
    if clip:
        D = np.clip(D, 0.0, 255.0)

    return SimulationResult(
        case_name="Case 3 (Enhanced Stripe)",
        degraded=D,
        clean=C,
        stripe=S,
        noise=N,
        params={"stripe_amp": stripe_amp, "freq": freq, "env_sigma": env_sigma, "seed": seed},
    )


def simulate_degradation_by_case(
    clean_image: np.ndarray,
    case_number: int,
    seed: int = 42,
    clip: bool = True,
) -> SimulationResult:
    """Convenience dispatcher to simulate Case 1, 2, or 3."""
    if case_number == 1:
        return simulate_case1_conventional(clean_image, seed=seed, clip=clip)
    elif case_number == 2:
        return simulate_case2_periodic(clean_image, seed=seed, clip=clip)
    elif case_number == 3:
        return simulate_case3_enhanced(clean_image, seed=seed, clip=clip)
    else:
        raise ValueError(f"Unknown case number {case_number}. Expected 1, 2, or 3.")
