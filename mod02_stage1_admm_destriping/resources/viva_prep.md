# Viva Voce Preparation Guide: Stage 1 ADMM, Schatten 1/2 Norm & Variable Splitting

---

### Q1: Why does the non-convex Schatten $1/2$ quasi-norm outperform the convex Nuclear Norm ($p=1$) in infrared stripe matrix recovery?
**Answer:**
The Nuclear Norm $\|\mathbf{S}\|_* = \sum_i \sigma_i$ is the convex envelope of matrix rank, and its proximal operator is Singular Value Thresholding (SVT), which applies a uniform constant shift:
$$\hat{\sigma}_i = \max(\sigma_i - \tau, 0)$$
This uniform shrinkage introduces **severe estimation bias**:
1. Large singular values corresponding to the true column stripe structure are over-penalized and underestimated by $\tau$.
2. To retain the dominant stripe component, $\tau$ must be kept small, which fails to zero out small trailing singular values caused by noise and background image leakage.

In contrast, the **Schatten $1/2$ quasi-norm** $\|\mathbf{S}\|_{S_{1/2}}^{1/2} = \sum_i \sqrt{\sigma_i}$ has an aggressive, non-linear subgradient $\propto \frac{1}{2\sqrt{\sigma}}$. As $\sigma \to 0$, the penalty slope approaches infinity, forcing small singular values to zero strictly (sparsity). Conversely, for large singular values $\sigma \gg \tau$, the penalty gradient decays to zero, eliminating shrinkage bias on the dominant column stripe modes.

---

### Q2: How does the Alternating Direction Method of Multipliers (ADMM) handle non-smooth and non-convex terms like $L_1$ and Schatten-$1/2$?
**Answer:**
Direct gradient-based optimization cannot handle the objective because:
- The differential operators ($\nabla, \nabla_{xx}$) are coupled inside non-smooth $L_1$ norm functions.
- The Schatten-$1/2$ norm on $\mathbf{S}$ is non-convex and non-Lipschitz at the origin.

ADMM resolves this via **variable splitting**:
By defining auxiliary variables $\mathbf{G} = \nabla \mathbf{C}$ and $\mathbf{H} = \nabla_{xx} \mathbf{C}$, the linear operators are transferred into smooth quadratic penalty terms in the Augmented Lagrangian:
$$\frac{\rho}{2} \left\| \nabla \mathbf{C} - \mathbf{G} + \frac{\mathbf{\Lambda}_G}{\rho} \right\|_F^2$$
This decouples the global coupled problem into three simple, isolated subproblems:
1. A smooth quadratic subproblem for $\mathbf{C}$ (solved in closed form via 2D FFT).
2. Decoupled pixel-wise $L_1$ proximal problems for $\mathbf{G}$ and $\mathbf{H}$ (solved via ISTA soft-thresholding).
3. A low-rank matrix proximal problem for $\mathbf{S}$ (solved via SVD half-thresholding).

---

### Q3: What is the role of the ADMM penalty parameter $\rho$, and how does it balance primal vs. dual convergence?
**Answer:**
The parameter $\rho > 0$ controls the weight of the augmented quadratic penalty terms enforcing the linear constraints $\nabla \mathbf{C} = \mathbf{G}$ and $\nabla_{xx} \mathbf{C} = \mathbf{H}$:
- **If $\rho$ is too small**: The constraint violations (primal residuals $r_{\text{prim}} = \nabla \mathbf{C} - \mathbf{G}$) are weakly penalized. The auxiliary variables $\mathbf{G}$ and $\mathbf{H}$ shrink aggressively, but $\mathbf{C}$ is slow to satisfy the constraint, leading to sluggish convergence and residual stripe artifacts.
- **If $\rho$ is too large**: The algorithm prioritizes feasibility over objective minimization. The thresholds $\lambda_2 / \rho$ and $\lambda_3 / \rho$ become tiny, turning off shrinkage and slowing down the dual variables (dual residuals $s_{\text{dual}} = \rho \nabla^T (G^{k+1} - G^k)$).

In our implementation, $\rho = 0.5$ balances primal and dual residuals, providing stable convergence within 50–75 iterations.

---

### Q4: How is the $\mathbf{C}$-subproblem (Eq. 15) solved in $O(MN \log MN)$ operations via the 2D Fast Fourier Transform?
**Answer:**
The normal equation for $\mathbf{C}$ is:
$$\left[ \mathbf{I} + \rho (\nabla_x^T \nabla_x + \nabla_y^T \nabla_y) + \rho \nabla_{xx}^T \nabla_{xx} \right] \mathbf{C} = \mathbf{RHS}$$
For an $M \times N$ image, $\mathbf{C}$ has $MN$ pixels, so the system matrix has dimension $MN \times MN$. Direct matrix inversion would require prohibitive $O((MN)^3)$ operations ($\approx 10^{14}$ FLOPS for $256 \times 256$).

However, under periodic boundary conditions, discrete finite difference operators are spatial circular convolutions. Convolution matrices are **Block Circulant with Circulant Blocks (BCCC)**, which are all simultaneously diagonalized by the 2D Discrete Fourier Transform matrix $\mathbf{F}_{2D}$:
$$\nabla_x^T \nabla_x = \mathbf{F}_{2D}^H \mathbf{\Lambda}_{\nabla_x} \mathbf{F}_{2D}$$
Thus, in the frequency domain, the differential operators become point-wise multiplication by their eigenvalues:
$$\mathcal{F}(\mathbf{C}) = \frac{\mathcal{F}(\mathbf{RHS})}{\mathbf{1} + \rho (|\mathcal{F}(\nabla_x)|^2 + |\mathcal{F}(\nabla_y)|^2) + \rho |\mathcal{F}(\nabla_{xx})|^2}$$
Using the 2D FFT, this requires two forward FFTs and one inverse FFT, executing in $O(MN \log MN)$ time ($\approx 15$ ms on modern CPU).

---

### Q5: Why is the second-order horizontal directional Total Variation $\left\|\frac{\partial^2 \mathbf{C}}{\partial x^2}\right\|_1$ essential in Stage 1?
**Answer:**
Vertical stripes run along columns, causing sharp, high-frequency intensity jumps across the horizontal $x$-direction:
- First-order derivative $\nabla_x \mathbf{C}$ penalizes both true vertical object edges (e.g., building walls, door frames) and stripe noise equally.
- Second-order horizontal derivative $\nabla_{xx} \mathbf{C} = \mathbf{C}(x+1) - 2\mathbf{C}(x) + \mathbf{C}(x-1)$ penalizes horizontal **curvature** (acceleration).

Stripe noise manifests as high-frequency oscillating ripples across columns with large second derivatives. Natural objects exhibit smooth intensity changes or isolated step edges with localized second derivatives. Incorporating $\left\|\frac{\partial^2 \mathbf{C}}{\partial x^2}\right\|_1$ specifically drives out cross-column periodic and random stripe fluctuations while leaving genuine scene contours intact and eliminating staircasing.

---

### Q6: What is the physical justification for the column-mean structural projection $\mathbf{S}_{\text{col}} = \operatorname{tile}(\operatorname{mean}(\mathbf{S}, \text{axis}=0))$?
**Answer:**
In an IRFPA, vertical stripes arise from individual column readout circuits (CTIAs and ADCs). Therefore, all $M$ pixels in column $j$ share the same physical amplifier bias $s_j$.
While SVD Schatten-$1/2$ shrinkage extracts low-rank components mathematically, unconstrained SVD modes can occasionally capture horizontal scene textures (e.g. horizontal road edges).
Projecting onto the column-mean subspace:
$$\mathbf{S}_{\text{col}}(i, j) = \frac{1}{M} \sum_{i'=1}^M \mathbf{S}(i', j)$$
enforces the rigid physical reality that the stripe noise is uniform down each column, completely preventing horizontal thermal scene features from leaking into the estimated noise map.

---

### Q7: How does ISTA soft-thresholding relate to the proximal operator of the $L_1$ norm?
**Answer:**
For the objective $f(x) = \lambda \|x\|_1 + \frac{\rho}{2} \|x - v\|_2^2$, completing the square gives the proximal operator definition:
$$\operatorname{prox}_{\frac{\lambda}{\rho} \|\cdot\|_1}(v) = \arg\min_x \frac{1}{2} \|x - v\|_2^2 + \frac{\lambda}{\rho} \|x\|_1$$
Because the $L_1$ norm is separable across pixels:
$$\min_{x_i} \frac{1}{2} (x_i - v_i)^2 + \theta |x_i|, \quad \theta = \frac{\lambda}{\rho}$$
Setting the subgradient to zero:
$$x_i - v_i + \theta \partial |x_i| = 0$$
- If $|v_i| \le \theta$, $0 \in [v_i - \theta, v_i + \theta]$, so $x_i = 0$.
- If $v_i > \theta$, $x_i = v_i - \theta$.
- If $v_i < -\theta$, $x_i = v_i + \theta$.
Thus, $\operatorname{prox}_{\theta \|\cdot\|_1}(v) = \operatorname{shrink}(v, \theta) = \operatorname{sign}(v) \max(|v| - \theta, 0)$, which is the fundamental shrinkage step of the Iterative Shrinkage-Thresholding Algorithm (ISTA).

---

### Q8: What are the primal and dual residuals in ADMM, and how do they guarantee convergence?
**Answer:**
For the constraint $\mathbf{A} x + \mathbf{B} z = c$:
1. **Primal Residual**: Measures feasibility of the linear constraint:
   $$r_{\text{prim}}^{k+1} = \mathbf{A} x^{k+1} + \mathbf{B} z^{k+1} - c = \begin{bmatrix} \nabla \mathbf{C}^{k+1} - \mathbf{G}^{k+1} \\ \nabla_{xx} \mathbf{C}^{k+1} - \mathbf{H}^{k+1} \end{bmatrix}$$
   As $k \to \infty$, $r_{\text{prim}}^{k} \to \mathbf{0}$, ensuring the constraints are satisfied.
2. **Dual Residual**: Measures stationarity of the Lagrangian with respect to the primal variable:
   $$s_{\text{dual}}^{k+1} = \rho \mathbf{A}^T \mathbf{B} (z^{k+1} - z^k) = \begin{bmatrix} \rho \nabla^T (\mathbf{G}^{k+1} - \mathbf{G}^k) \\ \rho \nabla_{xx}^T (\mathbf{H}^{k+1} - \mathbf{H}^k) \end{bmatrix}$$
   As $k \to \infty$, $s_{\text{dual}}^{k} \to \mathbf{0}$, ensuring the first-order optimality condition is satisfied.
By the Boyd et al. ADMM convergence theorem, if both $\|r_{\text{prim}}\|_F \to 0$ and $\|s_{\text{dual}}\|_F \to 0$, the sequence $(C^k, S^k)$ converges to the optimal solution.

---

### Q9: Why is the half-thresholding operator closed-form despite the non-convexity of the $L_{1/2}$ quasi-norm?
**Answer:**
The 1D scalar $L_{1/2}$ minimization is:
$$\min_x g(x) = \frac{1}{2} (y - x)^2 + \tau |x|^{1/2}$$
Although $g(x)$ is non-convex and non-Lipschitz at $x=0$, for $x > 0$, $g'(x) = x - y + \frac{\tau}{2\sqrt{x}} = 0$.
Substituting $t = \sqrt{x} > 0$ transforms this into a depressed cubic equation:
$$t^3 - y t + \frac{\tau}{2} = 0$$
Every cubic equation has an exact algebraic solution via Cardano's trigonometric formula. Evaluating the local minimum against the boundary value $g(0) = \frac{1}{2} y^2$ yields the exact threshold $T = \frac{3}{4} (2\tau)^{2/3}$.
Below $T$, $g(0)$ is the global minimum. Above $T$, the largest real root of the cubic equation is the global minimum. Thus, despite non-convexity, the global minimizer is known in exact closed form without numerical search.

---

### Q10: If Stage 1 ADMM destriping successfully removes the stripe noise $\mathbf{S}$, why is Stage 2 (Patch Low-Rank Tensor Recovery) still necessary in the pipeline?
**Answer:**
While Stage 1 effectively separates the structured, column-correlated stripe noise $\mathbf{S}$ (increasing PSNR by $+8$ to $+10$ dB), two challenges remain:
1. **Residual Gaussian Ambient Noise**: The degradation model contains $N \sim \mathcal{N}(0, \sigma_{\text{env}}^2)$. Because ambient noise is 2D white noise with full rank and isotropic gradients, Stage 1's low-rank stripe prior and directional prior do not remove it completely.
2. **Subtle Residual Nonuniformity**: In high-noise scenarios (e.g. Case 3), weak residual line noise can remain embedded in the clean estimate.

Stage 2 exploits **nonlocal patch self-similarity**: grouping similar 2D thermal patches across the image into high-order patch matrices/tensors. Because natural thermal textures share identical structural manifolds, low-rank tensor decomposition eliminates residual random noise $\mathbf{N}$ and polishes edges without blurring fine textures.
