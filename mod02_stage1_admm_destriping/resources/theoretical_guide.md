# Theoretical Guide: Stage 1 ADMM Formulation and Subproblem Derivations

---

## 1. Mathematical Formulation & Sign Convention Analysis

### 1.1 Physical Degradation Model (Eq. 5 & Eq. 7)
In an infrared focal plane array (IRFPA), the degraded observation matrix $\mathbf{D} \in \mathbb{R}^{M \times N}$ is physically the sum of the latent scene radiant flux $\mathbf{C}$, the column-correlated stripe noise $\mathbf{S}$, and temporal/environmental noise $\mathbf{N}$:
$$\mathbf{D} = \mathbf{C} + \mathbf{S} + \mathbf{N}$$
Therefore, the true physical data fidelity term must measure the discrepancy between the observation $\mathbf{D}$ and the reconstructed signal plus stripe component $(\mathbf{C} + \mathbf{S})$:
$$\Phi_{\text{fidelity}}(\mathbf{C}, \mathbf{S}) = \frac{1}{2} \|\mathbf{D} - (\mathbf{C} + \mathbf{S})\|_F^2 = \frac{1}{2} \|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2$$
Notice that when $\mathbf{S}$ is subtracted from $\mathbf{D}$, we obtain the restored estimate:
$$\mathbf{C} = \mathbf{D} - \mathbf{S} \quad (\text{as explicitly diagrammed in Fig. 6 of the paper})$$

### 1.2 Note on the Typographical Discrepancy in Eq. (11)
In Section III-C of the paper, Eq. (7) correctly introduces the objective with residual $\mathbf{D} - (\mathbf{C} + \mathbf{S})$. However, in Eq. (11), the published text contains a typographical expression printing $\frac{1}{2} \|\mathbf{D} - \mathbf{S} + \mathbf{C}\|_F^2$.
- If taken literally as $\mathbf{D} - \mathbf{S} + \mathbf{C} = 0$, then $\mathbf{C} = \mathbf{S} - \mathbf{D} = -(\mathbf{D} - \mathbf{S})$, producing an unphysical inverted negative image.
- Consequently, our implementation strictly adheres to the mathematically grounded data fidelity term $\frac{1}{2} \|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2$, consistent with Eq. (5), Eq. (7), Fig. 6, and all physical principles of infrared radiometric imaging.

---

## 2. Derivation of the Augmented Lagrangian (Eq. 14)

The constrained optimization problem is:
$$\min_{\mathbf{C}, \mathbf{S}, \mathbf{G}, \mathbf{H}} \frac{1}{2} \|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2 + \lambda_1 \|\mathbf{S}\|_{S_{1/2}}^{1/2} + \lambda_2 \|\mathbf{G}\|_1 + \lambda_3 \|\mathbf{H}\|_1$$
$$\text{subject to } \quad \nabla \mathbf{C} - \mathbf{G} = \mathbf{0}, \quad \nabla_{xx} \mathbf{C} - \mathbf{H} = \mathbf{0}$$
where $\nabla \mathbf{C} = [\nabla_x \mathbf{C}, \nabla_y \mathbf{C}] \in \mathbb{R}^{M \times N \times 2}$ and $\nabla_{xx} \mathbf{C} = \frac{\partial^2 \mathbf{C}}{\partial x^2} \in \mathbb{R}^{M \times N}$.

The Augmented Lagrangian $\mathcal{L}_\rho$ combines the unconstrained objective with linear multiplier terms and quadratic penalty terms:
$$\mathcal{L}_\rho(\mathbf{C}, \mathbf{S}, \mathbf{G}, \mathbf{H}, \mathbf{\Lambda}_G, \mathbf{\Lambda}_H) = \frac{1}{2} \|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2 + \lambda_1 \|\mathbf{S}\|_{S_{1/2}}^{1/2} + \lambda_2 \|\mathbf{G}\|_1 + \lambda_3 \|\mathbf{H}\|_1$$
$$+ \langle \mathbf{\Lambda}_G, \nabla \mathbf{C} - \mathbf{G} \rangle + \frac{\rho}{2} \|\nabla \mathbf{C} - \mathbf{G}\|_F^2 + \langle \mathbf{\Lambda}_H, \nabla_{xx} \mathbf{C} - \mathbf{H} \rangle + \frac{\rho}{2} \|\nabla_{xx} \mathbf{C} - \mathbf{H}\|_F^2$$
Completing the squares yields the compact scaled form (Eq. 14):
$$\mathcal{L}_\rho = \frac{1}{2} \|\mathbf{D} - \mathbf{C} - \mathbf{S}\|_F^2 + \lambda_1 \|\mathbf{S}\|_{S_{1/2}}^{1/2} + \lambda_2 \|\mathbf{G}\|_1 + \lambda_3 \|\mathbf{H}\|_1$$
$$+ \frac{\rho}{2} \left\| \nabla \mathbf{C} - \mathbf{G} + \frac{\mathbf{\Lambda}_G}{\rho} \right\|_F^2 + \frac{\rho}{2} \left\| \nabla_{xx} \mathbf{C} - \mathbf{H} + \frac{\mathbf{\Lambda}_H}{\rho} \right\|_F^2$$

---

## 3. Subproblem Derivations

### 3.1 Subproblem 1: Clean Scene $\mathbf{C}^{k+1}$ via FFT (Eq. 15)
Fixing all other variables, the objective for $\mathbf{C}$ is purely quadratic:
$$\min_{\mathbf{C}} \frac{1}{2} \|\mathbf{D} - \mathbf{S}^k - \mathbf{C}\|_F^2 + \frac{\rho}{2} \left\| \nabla_x \mathbf{C} - \mathbf{G}_x^k + \frac{\mathbf{\Lambda}_{Gx}^k}{\rho} \right\|_F^2 + \frac{\rho}{2} \left\| \nabla_y \mathbf{C} - \mathbf{G}_y^k + \frac{\mathbf{\Lambda}_{Gy}^k}{\rho} \right\|_F^2 + \frac{\rho}{2} \left\| \nabla_{xx} \mathbf{C} - \mathbf{H}^k + \frac{\mathbf{\Lambda}_H^k}{\rho} \right\|_F^2$$
Differentiating with respect to $\mathbf{C}$ and setting the gradient to zero:
$$\mathbf{C} - (\mathbf{D} - \mathbf{S}^k) + \rho \nabla_x^T \left( \nabla_x \mathbf{C} - \mathbf{G}_x^k + \frac{\mathbf{\Lambda}_{Gx}^k}{\rho} \right) + \rho \nabla_y^T \left( \nabla_y \mathbf{C} - \mathbf{G}_y^k + \frac{\mathbf{\Lambda}_{Gy}^k}{\rho} \right) + \rho \nabla_{xx}^T \left( \nabla_{xx} \mathbf{C} - \mathbf{H}^k + \frac{\mathbf{\Lambda}_H^k}{\rho} \right) = \mathbf{0}$$
Rearranging into the normal equation:
$$\left[ \mathbf{I} + \rho (\nabla_x^T \nabla_x + \nabla_y^T \nabla_y) + \rho \nabla_{xx}^T \nabla_{xx} \right] \mathbf{C} = \mathbf{RHS}^k$$
where:
$$\mathbf{RHS}^k = (\mathbf{D} - \mathbf{S}^k) + \rho \nabla_x^T \left(\mathbf{G}_x^k - \frac{\mathbf{\Lambda}_{Gx}^k}{\rho}\right) + \rho \nabla_y^T \left(\mathbf{G}_y^k - \frac{\mathbf{\Lambda}_{Gy}^k}{\rho}\right) + \rho \nabla_{xx}^T \left(\mathbf{H}^k - \frac{\mathbf{\Lambda}_H^k}{\rho}\right)$$
Under periodic boundary conditions, the differential matrices are Block Circulant with Circulant Blocks (BCCC), diagonalized by the 2D Discrete Fourier Transform ($\mathcal{F}$):
$$\mathcal{F}\left( \nabla_x^T \nabla_x \mathbf{C} \right) = |\mathcal{F}(\nabla_x)|^2 \odot \mathcal{F}(\mathbf{C})$$
$$\mathcal{F}\left( \nabla_y^T \nabla_y \mathbf{C} \right) = |\mathcal{F}(\nabla_y)|^2 \odot \mathcal{F}(\mathbf{C})$$
$$\mathcal{F}\left( \nabla_{xx}^T \nabla_{xx} \mathbf{C} \right) = |\mathcal{F}(\nabla_{xx})|^2 \odot \mathcal{F}(\mathbf{C})$$
Applying the Fourier transform to both sides:
$$\left[ \mathbf{1} + \rho (|\mathcal{F}(\nabla_x)|^2 + |\mathcal{F}(\nabla_y)|^2) + \rho |\mathcal{F}(\nabla_{xx})|^2 \right] \odot \mathcal{F}(\mathbf{C}) = \mathcal{F}(\mathbf{RHS}^k)$$
Since the denominator is strictly positive everywhere ($\ge 1.0$), inverting in closed form yields:
$$\mathbf{C}^{k+1} = \operatorname{Re}\left( \mathcal{F}^{-1} \left( \frac{\mathcal{F}(\mathbf{RHS}^k)}{\mathbf{1} + \rho (|\mathcal{F}(\nabla_x)|^2 + |\mathcal{F}(\nabla_y)|^2) + \rho |\mathcal{F}(\nabla_{xx})|^2} \right) \right) \quad \blacksquare$$

---

### 3.2 Subproblem 2: Stripe Noise $\mathbf{S}^{k+1}$ via Schatten 1/2 Shrinkage (Eqs. 16–17)
Given residual $\mathbf{Y} = \mathbf{D} - \mathbf{C}^{k+1}$:
$$\min_{\mathbf{S}} \frac{1}{2} \|\mathbf{Y} - \mathbf{S}\|_F^2 + \lambda_1 \|\mathbf{S}\|_{S_{1/2}}^{1/2}$$
By von Neumann's trace inequality and the unitary invariance of the Schatten-$p$ norm, the optimal $\mathbf{S}$ shares the left and right singular vectors of $\mathbf{Y}$:
$$\mathbf{Y} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \implies \mathbf{S}^* = \mathbf{U} \widehat{\mathbf{\Sigma}} \mathbf{V}^T$$
where each singular value $\hat{\sigma}_i$ solves the scalar half-thresholding problem:
$$\min_{\sigma \ge 0} \frac{1}{2} (\sigma_i - \sigma)^2 + \tau \sqrt{\sigma}, \quad \tau = \frac{\lambda_1}{\rho}$$

#### Derivation of the Half-Thresholding Formula
The first-order necessary condition for non-zero $\sigma > 0$ is:
$$\sigma - \sigma_i + \frac{\tau}{2 \sqrt{\sigma}} = 0 \implies \sigma^{3/2} - \sigma_i \sigma^{1/2} + \frac{\tau}{2} = 0$$
Let $t = \sqrt{\sigma} > 0$. The equation becomes the depressed cubic:
$$t^3 - \sigma_i t + \frac{\tau}{2} = 0$$
Using Cardano's trigonometric formula for $t^3 - p t + q = 0$ with $p = \sigma_i$ and $q = \tau / 2$:
The discriminant is:
$$\Delta = \left(\frac{q}{2}\right)^2 - \left(\frac{p}{3}\right)^3 = \frac{\tau^2}{16} - \frac{\sigma_i^3}{27}$$
Real roots exist when $\Delta \le 0$, which defines the critical threshold:
$$\sigma_i \ge \frac{3}{4} (2\tau)^{2/3} = T$$
When $\sigma_i > T$, the largest positive real root is:
$$t = 2 \sqrt{\frac{\sigma_i}{3}} \cos\left( \frac{\pi}{3} - \frac{1}{3} \arccos\left( \frac{3q}{2p} \sqrt{\frac{3}{p}} \right) \right)$$
Squaring $t$ gives the closed-form half-thresholding operator $\mathcal{H}_{1/2}(\sigma_i, \tau)$:
$$\mathcal{H}_{1/2}(\sigma_i, \tau) = \frac{2}{3} \sigma_i \left( 1 + \cos\left( \frac{2\pi}{3} - \frac{2}{3} \phi(\sigma_i) \right) \right)$$
where $\phi(\sigma_i) = \arccos\left( \frac{\tau}{8} \left( \frac{\sigma_i}{3} \right)^{-1.5} \right)$ and $\mathcal{H}_{1/2}(\sigma_i, \tau) = 0$ for $\sigma_i \le T$.

---

### 3.3 Subproblem 3: Auxiliary Variables $\mathbf{G}^{k+1}, \mathbf{H}^{k+1}$ via ISTA (Eqs. 18–19)
The auxiliary variables decouple pixel-wise into unconstrained proximal problems:
$$\min_{\mathbf{G}} \lambda_2 \|\mathbf{G}\|_1 + \frac{\rho}{2} \left\| \mathbf{G} - \left( \nabla \mathbf{C}^{k+1} + \frac{\mathbf{\Lambda}_G^k}{\rho} \right) \right\|_F^2$$
$$\min_{\mathbf{H}} \lambda_3 \|\mathbf{H}\|_1 + \frac{\rho}{2} \left\| \mathbf{H} - \left( \nabla_{xx} \mathbf{C}^{k+1} + \frac{\mathbf{\Lambda}_H^k}{\rho} \right) \right\|_F^2$$
These have the standard closed-form ISTA soft-thresholding solution:
$$\mathbf{G}^{k+1} = \operatorname{shrink}\left( \nabla \mathbf{C}^{k+1} + \frac{\mathbf{\Lambda}_G^k}{\rho}, \frac{\lambda_2}{\rho} \right)$$
$$\mathbf{H}^{k+1} = \operatorname{shrink}\left( \nabla_{xx} \mathbf{C}^{k+1} + \frac{\mathbf{\Lambda}_H^k}{\rho}, \frac{\lambda_3}{\rho} \right)$$
where $\operatorname{shrink}(v, \theta) = \operatorname{sign}(v) \max(|v| - \theta, 0)$.
