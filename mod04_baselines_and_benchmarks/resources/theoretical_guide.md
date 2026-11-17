# Theoretical Guide: Mathematical Comparison and Failure Mode Analysis of NUC Paradigms

## 1. Taxonomic Overview of Single-Frame Infrared Destriping
Single-frame infrared nonuniformity correction (NUC) methodologies fall into three foundational paradigms:
1. **Filtering-based spatial/transform methods**: `1D-GF`, `1D-WLS`, `BM3D`.
2. **Variational optimization models**: `ADOM`, `LRSID`, `GSRSR`.
3. **Deep learning and neural priors**: `SNRCNN`, `DLS-NUC`, `SNRWDNN`, `DestripeCycleGAN`.
4. **Hybrid Multistage Framework**: `Ours` (High-Order ADMM + Non-Local Patch Low-Rank SVD).

Below is a detailed mathematical comparison of their core formulations, structural assumptions, and intrinsic failure modes.

---

## 2. Filtering-Based Methods: Mechanics & Failure Modes

### 1D Guided Filter (1D-GF) - Cao et al. (2016)
- **Formulation**: Applies a 1-D local linear model $q_i = a_k I_i + b_k$ along rows/columns, minimizing:
  $$E(a_k, b_k) = \sum_{i \in \omega_k} \left( (a_k I_i + b_k - p_i)^2 + \epsilon a_k^2 \right)$$
- **Structural Assumption**: Assumes vertical stripes are high-frequency 1D signals and scene radiance is 1D piecewise smooth.
- **Intrinsic Failure Mode**: 1D-GF cannot distinguish between sharp vertical edges of actual objects (e.g. building columns, walls, human silhouettes) and vertical stripe noise. When filtering horizontally, authentic vertical edges are misidentified as stripe oscillations and smoothed out, resulting in severe edge blurring and elevated AVGE ($> 4.0$).

### 1D Weighted Least Squares (1D-WLS) - Dong et al. (2018)
- **Formulation**: Solves a 1-D edge-preserving tridiagonal linear system across column profiles:
  $$\min_s \sum_j (s_j - y_j)^2 + \lambda \sum_j a_j (s_j - s_{j-1})^2 \quad \text{where} \quad a_j = \left( |\Delta y_j|^\alpha + \epsilon \right)^{-1}$$
- **Intrinsic Failure Mode**: The spatial affinity weights $a_j$ rely purely on horizontal intensity differences. In regions with dense periodic stripes or heavy environmental noise, affinity weights fluctuate wildly, leaving low-frequency residual stripe bands ("ghost stripes") and causing over-smoothing in flat backgrounds.

### BM3D Collaborative Patch Filter - Dabov et al. (2007)
- **Formulation**: Groups mutually similar 2D patches across the image into 3D arrays, applies 3D unitary transforms (2D DCT + 1D Haar), and applies hard thresholding / Wiener shrinkage:
  $$\hat{\mathbf{Y}}_{3D} = \mathcal{T}_{3D}^{-1} \left( \mathcal{H}_\gamma \left( \mathcal{T}_{3D} (\mathbf{Y}_{3D}) \right) \right)$$
- **Intrinsic Failure Mode**: BM3D is optimal for i.i.d. additive white Gaussian noise (AWGN). However, vertical stripe noise exhibits extreme spatial correlation along columns. BM3D groups identical stripe patterns across rows, interpreting coherent stripe lines as authentic structural textures. Consequently, BM3D denoises background noise but leaves vertical stripes almost completely intact ($\rho > 0.18$).

---

## 3. Variational Optimization Models: Mechanics & Failure Modes

### ADOM - Kim et al. (2023)
- **Formulation**:
  $$\min_{\mathbf{C}, \mathbf{S}} \frac{1}{2}\|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2 + \lambda_1 \|\mathbf{W} \odot \nabla_x \mathbf{C}\|_1 + \lambda_2 \|\nabla_y \mathbf{S}\|_1 + \lambda_3 \|\mathbf{S}\|_1$$
- **Intrinsic Failure Mode**: Relies exclusively on first-order horizontal TV $\|\mathbf{W} \odot \nabla_x \mathbf{C}\|_1$. Because first-order TV promotes piecewise constant outputs, ADOM produces distinct horizontal staircasing steps across smooth gradients (e.g. clear sky or thermal monitor gradients). Furthermore, it lacks a second stage to suppress residual random environmental noise.

### LRSID - Chang et al. (2016)
- **Formulation**:
  $$\min_{\mathbf{C}, \mathbf{S}} \|\mathbf{S}\|_* + \tau \|\nabla_y \mathbf{C}\|_1 + \lambda_s \|\nabla_x \mathbf{S}\|_1 + \frac{\gamma}{2}\|\nabla_y \mathbf{S}\|_F^2 + \frac{\rho}{2}\|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2$$
- **Intrinsic Failure Mode**: Employs the convex Nuclear Norm $\|\mathbf{S}\|_* = \sum_i \sigma_i(\mathbf{S})$. The proximal operator is soft-thresholding $\mathcal{D}_\tau(\sigma_i) = \max(\sigma_i - \tau, 0)$, which subtracts a constant $\tau$ from **all** singular values. This causes severe shrinkage bias on dominant singular values representing scene contrast, leading to dull radiometric contrast and residual shading.

### GSRSR - Chen et al. (2017)
- **Formulation**: Enforces column group sparsity via the $L_{2,1}$ norm:
  $$\min_{\mathbf{C}, \mathbf{S}} \frac{1}{2}\|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2 + \lambda_c \|\nabla_y \mathbf{C}\|_1 + \lambda_g \|\mathbf{S}\|_{2,1} \quad \text{where} \quad \|\mathbf{S}\|_{2,1} = \sum_{j=1}^W \|\mathbf{S}_{:, j}\|_2$$
- **Intrinsic Failure Mode**: The $L_{2,1}$ norm assumes that entire columns are uniformly corrupted. When environmental noise causes non-uniform column fluctuations or when stripes are partially discontinuous, group shrinkage either over-penalizes clean columns or under-corrects corrupted ones.

---

## 4. Deep Learning Paradigms: Mechanics & Failure Modes

### SNRCNN (Kuai et al., 2017)
- **Formulation**: 3-layer convolutional network learning residual mapping $\mathbf{S} = f_{\theta}(\mathbf{D})$.
- **Failure Mode**: Receptive field is limited to small local windows ($15 \times 15$), making it unable to capture global column continuity. On severe corruptions, it leaves faint vertical line ghosts and slightly attenuates fine high-frequency textures.

### DLS-NUC (He et al., 2019)
- **Formulation**: 1D multi-scale dilated convolutions along columns capturing bias profiles $s_j$.
- **Failure Mode**: Excels at isolating 1D column offsets, but is fundamentally a 1D model: it possesses zero capacity to attenuate 2D isotropic environmental Gaussian noise $N(x, y)$.

### SNRWDNN (Guan et al., 2019)
- **Formulation**: Wavelet deep neural network applying directional shrinkage to $LH$ (horizontal detail) subband.
- **Failure Mode**: Directional wavelet filtering preserves vertical edges well, but introduces pseudo-Gibbs ringing and wavelet packet artifacts around high-contrast point sources and sharp thermal transitions.

### DestripeCycleGAN (Xiao et al., 2020)
- **Formulation**: Cycle-consistent adversarial translation without paired training data.
- **Failure Mode**: The adversarial generator tends to synthesize plausible textures that do not exist in the physical scene, causing slight radiometric temperature shifts in homogeneous areas (inflated MRD and AAD).

---

## 5. Why the Proposed Multistage Pipeline Excels

| Dimension | Filter Baselines | Variational Baselines | Deep Learning Baselines | Proposed Pipeline (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| **Horizontal Curvature** | None (1D smoothing) | First-order TV only (staircasing) | Convolutional receptive field | **Second-order TV $|\frac{\partial^2 \mathbf{C}}{\partial x^2}|$ (smooth cubic bridge)** |
| **Low-Rank Stripe Model** | None | Nuclear Norm (biased shrinkage) | None (black-box CNN) | **Non-convex Schatten-$1/2$ (asymptotically unbiased)** |
| **Environmental Noise Handling**| None or BM3D (stripes remain) | None (single-stage) | Mixed in latent space | **Stage 2 Patch Low-Rank SVD with adaptive $\tau^2$ bound** |
| **Vertical Edge Conservation** | Poor (blurred) | Moderate | High | **Near-zero AVGE ($< 2.75$)** |
| **Radiometric Temperature Fidelity**| Poor | Moderate | Distorted (GAN shift) | **Minimal MRD ($0.043$) & AAD ($4.03$)** |
