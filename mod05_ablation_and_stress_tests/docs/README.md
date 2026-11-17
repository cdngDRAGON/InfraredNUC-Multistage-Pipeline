# Module 5: Section IV-C3 Ablation Studies and Stress Tests

This module implements all four comprehensive ablation investigations and operational stress tests documented in Section IV-C3 of the IEEE Sensors Journal paper:
> *"Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images Based on Hybrid High-Order Directional and Low-Rank Prior Information"* (Feb 2026).

---

## 1. Overview of Ablation Experiments & Figures

| Experiment | Code File | Output Figures | Output Data Table | Scientific Focus |
| :--- | :--- | :--- | :--- | :--- |
| **Ablation 1** | `ablation_iterations.py` | `fig16_iterations_visual.png`<br>`fig17_iterations_psnr_ssim.png` | `ablation_iterations.csv` | **Maximum ADMM Iterations $K \in \{150, 200, 250, 300\}$**<br>Demonstrates decoupling of scene semantics from estimated stripe noise $\mathbf{S}$. At $K < 200$, incomplete primal-dual convergence causes checkerboard edges to leak into $\mathbf{S}$. At $K=250$, complete decoupling is achieved, reaching peak PSNR ($43.65\text{ dB}$). |
| **Ablation 2** | `ablation_lambdas.py` | `fig18_regularization_lambdas.png` | `ablation_lambdas.csv` | **Regularization Weights $\lambda_1, \lambda_2, \lambda_3 \in \{10^{-2}, 10^{-3}, 10^{-4}\}$**<br>Analyzes parametric sensitivity across 9 configurations. Confirms that $\lambda_1 = \lambda_2 = \lambda_3 = 10^{-3}$ achieves optimal balance between stripe removal, edge sharpness, and texture preservation. |
| **Ablation 3** | `ablation_noise_stress.py` | `fig19_stripe_intensity_stress.png`<br>`fig20_env_noise_intensity_stress.png` | `ablation_noise_stress.csv` | **Nonuniform Noise Stress Tests**<br>- **Fig. 19**: 5 escalating stripe levels ($\sigma_s \in \{8, 16, 24, 32, 40\}$) under constant ambient noise ($\sigma_n = 5$).<br>- **Fig. 20**: 4 escalating ambient noise levels ($\sigma_n \in \{4, 8, 14, 20\}$) under constant stripe noise ($\sigma_s = 15$). |
| **Ablation 4** | `ablation_patch_count_L.py` | `fig21_patch_count_visual.png`<br>`fig22_patch_count_psnr_ssim.png` | `ablation_patch_count_L.csv` | **Stage 2 Patch Count $L \in \{20, 25, 30, 35, 40, 45, 50, 55, 60\}$**<br>Evaluates Stage 1 alone ("w/o Stage 2", $36.67\text{ dB}$) vs. Stage 2. Proves $L=50$ is optimal; SSIM plateaus at $\approx 0.934$ for $L \ge 45$. |

---

## 2. Theoretical Insights & Empirical Findings

### 2.1 Decoupling Across Iteration Count $K$ (Figs. 16 & 17)
The Stage 1 energy minimization problem balances four competing terms via ADMM:
$$\min_{\mathbf{C}, \mathbf{S}} \frac{1}{2}\|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2 + \lambda_1 \|\mathbf{S}\|_{S_{1/2}}^{1/2} + \lambda_2 \|\nabla \mathbf{C}\|_1 + \lambda_3 \left\|\frac{\partial^2 \mathbf{C}}{\partial x^2}\right\|_1$$
- **Incomplete Convergence ($K=150, 200$)**:
  The dual multiplier matrices $\mathbf{\Lambda}_G, \mathbf{\Lambda}_H$ have not achieved full consensus with the primal gradients. Consequently, high-contrast thermal gradients from the scene project into the low-rank singular value decomposition of $\mathbf{D} - \mathbf{C}^k$. In Figure 16, object contours and checkerboard boundaries are visibly apparent within the estimated stripe matrix $\mathbf{S}$.
- **Optimal Decoupling ($K=250$)**:
  Primal and dual residuals satisfy $\|\mathbf{C}^{k+1} - \mathbf{C}^k\|_F / \|\mathbf{C}^k\|_F < 10^{-4}$. The estimated stripe matrix $\mathbf{S}$ becomes purely column-correlated with zero scene texture leakage ($\partial \mathbf{S} / \partial y \approx 0$). The restored thermal image attains peak PSNR ($43.65\text{ dB}$) and SSIM ($0.9820$).
- **Asymptotic Plateau ($K=300$)**:
  Additional iterations yield marginal improvements ($< 0.05\text{ dB}$) while increasing computation time by $20\%$.

### 2.2 Sensitivity to Regularization Parameters (Fig. 18)
- **$\lambda_1$ (Schatten-$1/2$ Low-Rank Weight)**:
  - $\lambda_1 = 10^{-2}$ (Over-regularization): Suppresses the rank of $\mathbf{S}$ excessively, causing residual vertical stripes to remain in $\mathbf{C}$.
  - $\lambda_1 = 10^{-4}$ (Under-regularization): Insufficient low-rank constraint allows $\mathbf{S}$ to capture native thermal radiance, attenuating scene contrast.
- **$\lambda_2$ (First-Order TV Weight $\|\nabla \mathbf{C}\|_1$)**:
  - $\lambda_2 = 10^{-2}$: Induces piecewise-constant "cartooning" artifacts, destroying subtle infrared micro-textures.
  - $\lambda_2 = 10^{-4}$: Fails to suppress residual isotropic noise.
- **$\lambda_3$ (Second-Order Directional TV $\|\partial^2 \mathbf{C} / \partial x^2\|_1$)**:
  - $\lambda_3 = 10^{-2}$: Blurs legitimate vertical thermal edges (e.g. building columns, pedestrian silhouettes).
  - $\lambda_3 = 10^{-4}$: Fails to damp high-frequency horizontal ripples across detector columns, leaving visible vertical streakiness.
- **Optimal Balance**: $\lambda_1 = \lambda_2 = \lambda_3 = 10^{-3}$ achieves minimal stripe error MAE and maximal structural preservation.

### 2.3 Noise Stress Tests (Figs. 19 & 20)
- **Stripe Stress ($\sigma_s \in [8, 40]$)**:
  Even under extreme corruption where stripe amplitude exceeds native thermal contrast ($\sigma_s = 40$), the Schatten-$1/2$ subspace projection cleanly extracts the column component, sustaining a net PSNR gain $\Delta\text{PSNR} > +6.5\text{ dB}$.
- **Ambient Noise Stress ($\sigma_n \in [4, 20]$)**:
  While Stage 1 isolates structured column disparities, Stage 2's adaptive SVD patch approximation effectively eliminates heavy Gaussian thermal noise, delivering smooth surfaces with sharp edge retention.

### 2.4 Patch Manifold Cardinality $L$ in Stage 2 (Figs. 21 & 22)
Stage 2 constructs an observation matrix $\mathbf{G} \in \mathbb{R}^{d \times L}$ from $L$ non-local similar patches ($d = 11 \times 11 = 121$):
$$\mathbf{G} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T, \quad \sum_{i=k}^n \sigma_i^2 > \tau^2 \ge \sum_{i=k+1}^n \sigma_i^2$$
- **$L < 35$**: Insufficient statistical redundancy leads to noisy patch estimates; singular values of noise and scene overlap.
- **$L = 50$**: Provides optimal redundancy. PSNR surges from $36.67\text{ dB}$ (Stage 1 alone) to $> 41.25\text{ dB}$, and SSIM plateaus at $0.934$.
- **$L > 50$**: Gathering increasingly distant and dissimilar patches introduces slight edge blurring without further noise reduction.

---

## 3. Directory Layout

```
mod05_ablation_and_stress_tests/
├── src/
│   ├── ablation_iterations.py      # Figs. 16-17: K in {150, 200, 250, 300} ADMM iteration analysis
│   ├── ablation_lambdas.py         # Fig. 18: Regularization parameter sensitivity (9 configs)
│   ├── ablation_noise_stress.py    # Figs. 19-20: Stripe intensity and environmental noise stress tests
│   ├── ablation_patch_count_L.py   # Figs. 21-22: Stage 2 patch count L sweep vs. w/o Stage 2
│   └── run_mod05.py                # Master CLI runner executing all 4 ablation studies
├── tests/
│   └── test_mod05.py               # Pytest suite verifying K=250, lambda=1e-3, L=50 optimality
├── docs/
│   ├── README.md                   # This documentation guide
│   └── ieee_section_mod05.tex      # Publication-ready IEEE LaTeX section for Section IV-C3
├── resources/
│   ├── theoretical_guide.md        # Mathematical proofs of convergence, Schatten-1/2, and rank bounds
│   └── viva_prep.md                # 10 Viva Voce questions & detailed answers
└── results/                        # Generated 300-DPI publication figures & CSV tables
    ├── fig16_iterations_visual.png
    ├── fig17_iterations_psnr_ssim.png
    ├── fig18_regularization_lambdas.png
    ├── fig19_stripe_intensity_stress.png
    ├── fig20_env_noise_intensity_stress.png
    ├── fig21_patch_count_visual.png
    ├── fig22_patch_count_psnr_ssim.png
    ├── ablation_iterations.csv
    ├── ablation_lambdas.csv
    ├── ablation_noise_stress.csv
    └── ablation_patch_count_L.csv
```

---

## 4. Execution & Verification

### Running the Complete Module 5 Pipeline
```bash
export MPLCONFIGDIR=/tmp/matplotlib_cache
python3 -m mod05_ablation_and_stress_tests.src.run_mod05
```

### Running the Unit Test Suite
```bash
export MPLCONFIGDIR=/tmp/matplotlib_cache
pytest mod05_ablation_and_stress_tests/tests/ -v
```
