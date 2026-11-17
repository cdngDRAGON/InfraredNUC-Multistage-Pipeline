# Theoretical Guide: Optimization Landscape, Convergence Guarantees, and Parameter Mechanics (Section IV-C3)

This resource provides deep mathematical foundations, convergence proofs, and parametric sensitivity derivations for the four ablation studies in Section IV-C3 of the paper:
> *"Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images Based on Hybrid High-Order Directional and Low-Rank Prior Information"* (IEEE Sensors Journal, Feb 2026).

---

## 1. Stage 1 ADMM Formulation and KKT Stationarity Conditions

### 1.1 Composite Optimization Problem
Stage 1 formulates stripe removal as a constrained composite optimization problem on the latent radiant scene $\mathbf{C} \in \mathbb{R}^{M \times N}$ and stripe matrix $\mathbf{S} \in \mathbb{R}^{M \times N}$:
$$\min_{\mathbf{C}, \mathbf{S}} \mathcal{F}(\mathbf{C}, \mathbf{S}) = \frac{1}{2} \|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2 + \lambda_1 \|\mathbf{S}\|_{S_{1/2}}^{1/2} + \lambda_2 \|\nabla \mathbf{C}\|_1 + \lambda_3 \left\|\frac{\partial^2 \mathbf{C}}{\partial x^2}\right\|_1$$

Introducing auxiliary splitting variables $\mathbf{G} = [\mathbf{G}_x, \mathbf{G}_y] = \nabla \mathbf{C}$ and $\mathbf{H} = \nabla_{xx} \mathbf{C} = \frac{\partial^2 \mathbf{C}}{\partial x^2}$, the Augmented Lagrangian is:
$$\begin{aligned}
\mathcal{L}_\rho(\mathbf{C}, \mathbf{S}, \mathbf{G}, \mathbf{H}, \mathbf{\Lambda}_G, \mathbf{\Lambda}_H) &= \frac{1}{2} \|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2 + \lambda_1 \|\mathbf{S}\|_{S_{1/2}}^{1/2} + \lambda_2 \|\mathbf{G}\|_1 + \lambda_3 \|\mathbf{H}\|_1 \\
&+ \frac{\rho}{2} \left\|\nabla \mathbf{C} - \mathbf{G} + \frac{\mathbf{\Lambda}_G}{\rho}\right\|_F^2 + \frac{\rho}{2} \left\|\nabla_{xx} \mathbf{C} - \mathbf{H} + \frac{\mathbf{\Lambda}_H}{\rho}\right\|_F^2
\end{aligned}$$

### 1.2 First-Order Stationarity (KKT Conditions)
At an optimal saddle point $(\mathbf{C}^*, \mathbf{S}^*, \mathbf{G}^*, \mathbf{H}^*, \mathbf{\Lambda}_G^*, \mathbf{\Lambda}_H^*)$, the subdifferential inclusions satisfy:
1. **Primal Feasibility**:
   $$\nabla \mathbf{C}^* - \mathbf{G}^* = \mathbf{0}, \quad \nabla_{xx} \mathbf{C}^* - \mathbf{H}^* = \mathbf{0}$$
2. **Dual Multiplier Subdifferentials**:
   $$\mathbf{\Lambda}_G^* \in \lambda_2 \, \partial \|\mathbf{G}^*\|_1, \quad \mathbf{\Lambda}_H^* \in \lambda_3 \, \partial \|\mathbf{H}^*\|_1$$
3. **Primal Stationarity with respect to $\mathbf{C}$**:
   $$\mathbf{C}^* + \mathbf{S}^* - \mathbf{D} + \nabla^T \mathbf{\Lambda}_G^* + \nabla_{xx}^T \mathbf{\Lambda}_H^* = \mathbf{0}$$
4. **Primal Stationarity with respect to $\mathbf{S}$**:
   $$\mathbf{C}^* + \mathbf{S}^* - \mathbf{D} + \lambda_1 \, \partial \|\mathbf{S}^*\|_{S_{1/2}}^{1/2} \ni \mathbf{0}$$

---

## 2. Decoupling Dynamics Across Iteration Count $K$ (Figs. 16 & 17)

### 2.1 Why Incomplete Convergence Causes Scene Semantic Leakage ($K < 200$)
At any intermediate iteration $k < 200$:
$$\mathbf{Y}^k = \mathbf{D} - \mathbf{C}^k = \mathbf{S}_{\mathrm{true}} + (\mathbf{C}_{\mathrm{true}} - \mathbf{C}^k) + \mathbf{N}$$
Because the dual variables $\mathbf{\Lambda}_G^k, \mathbf{\Lambda}_H^k$ have not converged, the reconstructed scene $\mathbf{C}^k$ possesses significant reconstruction error $\mathbf{E}_C^k = \mathbf{C}_{\mathrm{true}} - \mathbf{C}^k$ along high-gradient boundaries (e.g. the checkerboard edges in Fig. 16).

When the Schatten-$1/2$ shrinkage operator $\mathcal{H}_{1/2}(\mathbf{Y}^k, \tau)$ is applied, its singular value decomposition:
$$\mathbf{Y}^k = \sum_{i=1}^{\min(M, N)} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$
receives dominant energy contributions not only from the true column stripes (which form a rank-1 or rank-2 subspace), but also from the coherent geometric patterns in $\mathbf{E}_C^k$. Consequently:
$$\mathbf{S}^k = \sum_{i=1}^r \hat{\sigma}_i \mathbf{u}_i \mathbf{v}_i^T$$
inevitably incorporates structural contours from the underlying scene. In Fig. 16, this appears as rectangular outlines of checkerboard squares inside $\mathbf{S}^{(150)}$ and $\mathbf{S}^{(200)}$.

### 2.2 Proof of Complete Separation at $K=250$
Under the standard ADMM step-size and penalty parameter $\rho = 0.5$, the sequence of primal residuals $\mathbf{r}_k = [\nabla \mathbf{C}^k - \mathbf{G}^k, \nabla_{xx} \mathbf{C}^k - \mathbf{H}^k]$ and dual residuals $\mathbf{s}_k = \rho \nabla^T (\mathbf{G}^k - \mathbf{G}^{k-1}) + \rho \nabla_{xx}^T (\mathbf{H}^k - \mathbf{H}^{k-1})$ satisfies:
$$\lim_{k \to \infty} \|\mathbf{r}_k\|_F = 0, \quad \lim_{k \to \infty} \|\mathbf{s}_k\|_F = 0$$
Empirically, for $256 \times 256$ thermal scenes, the relative primal step:
$$\frac{\|\mathbf{C}^{k+1} - \mathbf{C}^k\|_F}{\|\mathbf{C}^k\|_F + 10^{-10}} < 10^{-4}$$
is achieved at $k \approx 230 - 250$. At this point:
$$\|\mathbf{E}_C^k\|_F \approx 0 \implies \mathbf{Y}^k \to \mathbf{S}_{\mathrm{true}} + \mathbf{N}$$
The column-correlated subspace projection $\mathbf{P}_{\mathrm{col}}(\mathbf{S}_{\mathrm{svd}}) = \frac{1}{M} \mathbf{1}_M \mathbf{1}_M^T \mathbf{S}_{\mathrm{svd}}$ perfectly eliminates any residual non-stripe component, yielding pure vertical lines ($\partial \mathbf{S} / \partial y = 0$).

---

## 3. Regularization Parameter Mechanics: Trade-Off Analysis of $(\lambda_1, \lambda_2, \lambda_3)$ (Fig. 18)

| Parameter | Physical Role | Over-Regularization Failure Mode ($\lambda = 10^{-2}$) | Under-Regularization Failure Mode ($\lambda = 10^{-4}$) | Optimal Setting |
| :--- | :--- | :--- | :--- | :--- |
| **$\lambda_1$** | Promotes sparsity of the singular value spectrum of $\mathbf{S}$ (enforcing rank $\le 2$). | $\tau = \lambda_1/\rho$ is too large. Valid stripe singular values are shrunk to zero, leaving residual stripes in $\mathbf{C}$. | Low-rank penalty is too weak. Non-stripe thermal structures are absorbed into $\mathbf{S}$, attenuating image contrast. | **$\lambda_1 = 10^{-3}$** |
| **$\lambda_2$** | Promotes sparsity of 1st-order spatial gradients $\|\nabla \mathbf{C}\|_1$. | Gradient shrinkage over-flattens the scene, inducing piecewise-constant "cartooning" and destroying subtle thermal texture. | Gradient penalty is insufficient to suppress high-frequency isotropic noise residuals. | **$\lambda_2 = 10^{-3}$** |
| **$\lambda_3$** | Penalizes 2nd-order horizontal curvature $\|\partial^2 \mathbf{C}/\partial x^2\|_1$. | Over-regularization blurs legitimate vertical edges (e.g. wall pillars, pedestrian contours). | Horizontal curvature is insufficiently regularized; high-frequency cross-column ripples persist. | **$\lambda_3 = 10^{-3}$** |

---

## 4. Stage 2 Non-Local Low-Rank Patch Manifold Capacity (Figs. 21 & 22)

### 4.1 Observation Matrix Construction and Redundancy
For each reference patch $\mathbf{y}_i \in \mathbb{R}^{d}$ ($d = B \times B = 11 \times 11 = 121$), Stage 2 gathers the $L$ most similar patches within search window $\Omega_i$ based on Euclidean distance:
$$\mathbf{G}_i = [\mathbf{y}_{i, 1}, \mathbf{y}_{i, 2}, \dots, \mathbf{y}_{i, L}] \in \mathbb{R}^{d \times L}$$
Because natural infrared scenes exhibit high non-local self-similarity, the noise-free patch matrix $\mathbf{G}_{0, i}$ lies on a low-dimensional manifold:
$$\mathrm{rank}(\mathbf{G}_{0, i}) = k^* \ll \min(d, L)$$

### 4.2 Noise Energy Truncation Bound (Eq. 26)
Under additive Gaussian noise $\mathbf{N} \sim \mathcal{N}(0, \sigma_n^2 \mathbf{I})$, the singular values of $\mathbf{G}_i = \mathbf{G}_{0, i} + \mathbf{N}$ exhibit a distinct separation:
$$\sigma_1 \ge \dots \ge \sigma_{k^*} > \sigma_{k^*+1} \approx \dots \approx \sigma_n \approx \sqrt{d L} \, \sigma_n$$
By the Eckart-Young-Mirsky theorem, the optimal rank-$k$ approximation in the Frobenius norm is given by the truncated SVD:
$$\mathbf{G}_{k, i} = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
Equation (26) selects the adaptive rank $k$ satisfying:
$$\sum_{i=k}^n \sigma_i^2 > \tau^2 \ge \sum_{i=k+1}^n \sigma_i^2, \quad \text{where } \tau^2 = \gamma \, d L \, \sigma_n^2$$
- **When $L < 35$**: The sample size $L$ is too small; empirical noise eigenvalues fluctuate widely, and genuine structural singular values bleed into the noise tail, leading to under-filtering.
- **When $L = 50$**: The empirical distribution of noise singular values concentrates tightly around the Marchenko-Pastur bound, allowing $\tau^2$ to cleanly excise noise energy without disturbing patch semantics.
- **When $L > 50$**: In non-homogeneous regions, the search window $\Omega_i$ exhausts truly similar patches and admits dissimilar patches, which corrupts the low-rank subspace and degrades edge sharpness.
