# Module 1: IRFPA Nonuniformity Degradation Model & Prior Information Verification

[![Module](https://img.shields.io/badge/Module-01--Priors%20%26%20Noise%20Simulator-00629B.svg)]()
[![Status](https://img.shields.io/badge/Status-Verified-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-10%2F10%20Passing-brightgreen.svg)]()

---

## 1. Overview & Objectives

Module 1 establishes the mathematical formulation and physical foundations for the entire multistage nonuniformity correction (NUC) pipeline. Specifically, it implements:
1. **The IRFPA Degradation Model (Eq. 5)**:
   $$D(x, y) = C(x, y) + S(x, y) + N(x, y)$$
2. **Three Real-World Sensor Degradation Regimes (Fig. 9)**:
   - Case 1: Conventional aperiodic column gain/offset FPN + environmental noise.
   - Case 2: Periodic harmonic readout crosstalk + random column FPN + environmental noise.
   - Case 3: Enhanced multi-harmonic severe stripes + defective column spikes + elevated noise.
3. **Directional Gradient Anisotropy Analysis (Figs. 2 & 3)**:
   - Quantitative and qualitative proof that horizontal gradients $\nabla_x D$ oscillate wildly due to vertical stripes, whereas vertical gradients $\nabla_y D$ preserve the underlying structural details of $C$.
4. **High-Order Directional Derivative Analysis (Fig. 5)**:
   - Verification that $\frac{\partial^2}{\partial y^2}$ is uncorrupted by vertical stripes, providing the theoretical justification for Hybrid High-Order Directional Total Variation (HODTV).
5. **SVD Low-Rank Verification & Eckart-Young-Mirsky Theorem (Eqs. 1–4, Fig. 4)**:
   - Demonstration that the stripe matrix $S$ has an intrinsic rank $\le 3$, and infrared scenes concentrate over 98% of their Frobenius energy in the top singular values.

---

## 2. Directory Layout

```
mod01_priors_and_noise_simulator/
├── src/
│   ├── noise_simulator.py          # Eq. 5 & Fig. 9: Cases 1, 2, and 3 simulation
│   ├── gradient_priors.py          # Figs. 2, 3 & 5: 1st/2nd-order directional gradients
│   ├── lowrank_svd_analysis.py     # Eqs. 1-4 & Fig. 4: SVD rank-k decay & Fig. 9 3D energy
│   └── run_mod01.py                # Master CLI runner
├── tests/
│   └── test_mod01.py               # 10 rigorous unit tests
├── docs/
│   ├── README.md                   # This execution guide
│   └── ieee_section_mod01.tex      # IEEE publication LaTeX manuscript section
├── resources/
│   ├── theoretical_guide.md        # Mathematical proofs & ROIC physics
│   └── viva_prep.md                # 10 comprehensive Viva Voce questions & answers
└── results/                        # Generated 300-DPI publication figures & CSVs
    ├── fig2_horizontal_gradient_energy.png
    ├── fig3_vertical_gradient_energy.png
    ├── fig4_svd_singular_value_distribution.png
    ├── fig5_first_and_second_order_gradients.png
    ├── fig9_simulation_cases_3d_energy.png
    ├── gradient_anisotropy_ratio.csv
    └── svd_energy_distribution.csv
```

---

## 3. Mathematical Foundations

### 3.1 Degradation Model (Eq. 5)
$$D = C + S + N$$
where $C \in \mathbb{R}^{M \times N}$ is the true scene radiant flux map, $S \in \mathbb{R}^{M \times N}$ is the column-correlated stripe noise matrix characterized by $S(y, x) = s_{\text{col}}(x) + \delta_S(y, x)$ with $\text{rank}(S) \le 3$, and $N \sim \mathcal{N}(0, \sigma^2)$ is zero-mean random ambient detector noise.

### 3.2 SVD and Low-Rank Energy Concentration (Eqs. 1–4)
By the **Eckart-Young-Mirsky Theorem**, any matrix $J \in \mathbb{R}^{M \times N}$ with SVD $J = U \Sigma V^T = \sum_{i=1}^r \sigma_i u_i v_i^T$ has an optimal rank-$k$ approximation:
$$\hat{J}_k = U_k \Sigma_k V_k^T = \arg\min_{\text{rank}(X) \le k} \|J - X\|_F$$
with exact residual Frobenius error:
$$\|J - \hat{J}_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$
and energy capture ratio:
$$\eta_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$$

### 3.3 Directional Derivative Anisotropy
For vertical stripe noise $S(y, x) = s(x)$:
- $\nabla_y S = 0 \implies \nabla_y D \approx \nabla_y C + \nabla_y N$ (Vertical detail is preserved).
- $\nabla_x S = s'(x) \ne 0 \implies \nabla_x D$ is corrupted by high-frequency spikes.
- $\frac{\partial^2 S}{\partial y^2} = 0 \implies$ High-order vertical curvature is completely immune to vertical stripe noise.

---

## 4. Execution & Verification

### Running the Master Pipeline
```bash
python3 -m mod01_priors_and_noise_simulator.src.run_mod01
```

### Running Module Tests
```bash
pytest mod01_priors_and_noise_simulator/tests/ -v
```
All 10 unit tests verify mathematical bounds, Eckart-Young-Mirsky errors, directional gradient ratios ($>10^3$), and noise simulation cases.
