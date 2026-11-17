# Viva Voce Preparation: Module 5 Ablation Studies and Stress Tests (Section IV-C3)

This resource contains 10 rigorous, publication-level Viva Voce examination questions and model answers on the ablation experiments, hyperparameter selection, and stress testing of the IEEE Sensors Journal paper (Feb 2026).

---

### Q1: In Figure 16, why does an insufficient number of iterations ($K < 200$) cause genuine image semantics to leak into the estimated stripe matrix $\mathbf{S}$?
**Answer**:
In Stage 1 ADMM, the objective is to decompose the degraded observation $\mathbf{D}$ into a piecewise-smooth thermal scene $\mathbf{C}$ and a low-rank stripe matrix $\mathbf{S}$. At early iterations ($K = 150, 200$), the primal-dual variables have not reached consensus; the residual $\mathbf{D} - \mathbf{C}^k$ still contains high-contrast scene boundaries (such as the checkerboard edges in `Img1`). When the Schatten-$1/2$ singular value thresholding operator $\mathcal{H}_{1/2}$ is computed on this residual, these prominent scene edges possess high spectral energy and project into the dominant singular vectors. Consequently, structural outlines leak into $\mathbf{S}^k$. Only when iterations reach $K = 250$ does $\mathbf{C}^k$ accurately recover scene edges, leaving the residual $\mathbf{D} - \mathbf{C}^k$ composed solely of true vertical column disparities, achieving complete, semantic-free decoupling.

---

### Q2: Why is $K = 250$ selected as the optimal iteration threshold in Figure 17, and what happens if $K$ is increased to 300 or 500?
**Answer**:
Figure 17 demonstrates that the restored image PSNR rises monotonically from $40.85\text{ dB}$ at $K=150$ to $43.65\text{ dB}$ at $K=250$, with SSIM reaching a sharp plateau at $0.9820$. Beyond $K=250$, the relative Frobenius change $\|\mathbf{C}^{k+1} - \mathbf{C}^k\|_F / \|\mathbf{C}^k\|_F$ drops well below the convergence tolerance ($10^{-4}$). Advancing from $K=250$ to $K=300$ yields an insignificant PSNR gain of $< 0.02\text{ dB}$ while incurring a $20\%$ computational time penalty. Running to $K=500$ provides zero perceptual improvement and risks slight over-smoothing of low-contrast thermal boundaries. Hence, $K=250$ represents the optimal Pareto trade-off between restoration fidelity and operational latency.

---

### Q3: What is the failure mode if the Schatten-1/2 regularizer weight $\lambda_1$ is set too high ($10^{-2}$) or too low ($10^{-4}$) in Figure 18?
**Answer**:
- **When $\lambda_1 = 10^{-2}$ (Over-regularization)**: The threshold $\tau = \lambda_1 / \rho$ in the half-thresholding operator $\mathcal{H}_{1/2}$ is excessively large. This heavily penalizes the singular values of the stripe matrix, shrinking subtle and moderate column disparities to zero. As a result, $\mathbf{S}$ under-estimates the true non-uniformity, and noticeable vertical stripes remain embedded in the restored scene $\mathbf{C}$.
- **When $\lambda_1 = 10^{-4}$ (Under-regularization)**: The low-rank penalty on $\mathbf{S}$ is too weak. The matrix $\mathbf{S}$ gains unnecessary degrees of freedom and begins absorbing broad thermal gradients and radiant background energy from the scene, causing significant contrast attenuation in $\mathbf{C}$ ($\mathrm{MAE} = 4.58$).

---

### Q4: Why does the second-order horizontal TV prior ($\lambda_3 \|\partial^2 \mathbf{C} / \partial x^2\|_1$) preserve vertical edges, whereas standard first-order horizontal TV ($\nabla_x$) blurs them?
**Answer**:
Standard first-order horizontal Total Variation $\|\nabla_x \mathbf{C}\|_1 = \sum |\mathbf{C}(x+1, y) - \mathbf{C}(x, y)|$ penalizes the existence of any horizontal gradient. Because a genuine vertical object edge (such as a building facade or pole) exhibits a sharp step change along the horizontal direction, minimizing $\|\nabla_x \mathbf{C}\|_1$ directly penalizes and blurs that edge.
In contrast, second-order horizontal Total Variation $\|\frac{\partial^2 \mathbf{C}}{\partial x^2}\|_1 = \sum |\mathbf{C}(x+1, y) - 2\mathbf{C}(x, y) + \mathbf{C}(x-1, y)|$ penalizes *horizontal curvature* (rapid oscillations or zigzag patterns across adjacent columns). For a sharp vertical step edge or a smooth linear ramp, the second derivative is zero everywhere except at the single inflection column, meaning genuine edges are preserved with high fidelity while alternating high-frequency column stripes are heavily suppressed.

---

### Q5: What happens when $\lambda_3$ is set to $10^{-4}$ in Figure 18?
**Answer**:
When $\lambda_3 = 10^{-4}$, the second-order horizontal directional penalty is severely under-weighted. The algorithm loses its primary mechanism for penalizing cross-column intensity fluctuations. As seen in Figure 18 (bottom row, right), high-frequency vertical streaks persist in the restored image, causing the stripe estimation error to surge to its worst value ($\mathrm{MAE} = 6.56$) and degrading PSNR to $29.01\text{ dB}$.

---

### Q6: Why can't first-order TV ($\lambda_2$) alone solve the destriping problem without $\lambda_3$?
**Answer**:
First-order TV is isotropic; it penalizes gradients equally in the horizontal ($\nabla_x$) and vertical ($\nabla_y$) directions. Thermal stripe noise is inherently anisotropic—it is constant or slowly drifting along columns ($\nabla_y \mathbf{S} \approx 0$) but oscillates aggressively across rows ($\nabla_x \mathbf{S} \gg 0$). If only $\lambda_2$ is used, increasing $\lambda_2$ to eliminate horizontal stripe ripples simultaneously over-smooths vertical scene variations, destroying authentic horizontal edges (e.g. road markings, horizon lines, window sills) and causing severe piecewise-constant "cartoon" artifacts.

---

### Q7: In Stage 2, why is the patch size chosen as $11 \times 11$ rather than $7 \times 7$ or $15 \times 15$?
**Answer**:
- If $B = 7 \times 7$ ($d = 49$): The patch vector dimension is too small. Natural infrared image textures cannot be uniquely characterized, resulting in false patch matchings during the non-local Euclidean search and inadequate statistical distinction between signal and noise.
- If $B = 15 \times 15$ ($d = 225$): The vector dimension is excessively large. It becomes difficult to find $L=50$ truly similar patches within the search radius $\Omega_i$ unless the scene is entirely uniform. Furthermore, SVD on a $225 \times 50$ matrix increases computation time quadratically without yielding any measurable PSNR improvement over $B=11$.
Thus, $B = 11 \times 11$ ($d = 121$) represents the optimal balance between spatial representation capacity and computational efficiency.

---

### Q8: What is the physical and mathematical justification for the adaptive rank criterion in Eq. (26)?
**Answer**:
Equation (26) defines the adaptive truncation rank $k$ via the cumulative tail energy of the singular values:
$$\sum_{i=k}^n \sigma_i^2 > \tau^2 \ge \sum_{i=k+1}^n \sigma_i^2, \quad \text{where } \tau^2 = \gamma \, d L \, \sigma_n^2$$
Under the assumption of additive zero-mean Gaussian noise with variance $\sigma_n^2$, an observation matrix of dimension $d \times L$ composed purely of noise has an expected total Frobenius energy $\mathbb{E}[\|\mathbf{N}\|_F^2] = d L \sigma_n^2$. By the properties of SVD, the total energy equals the sum of squared singular values $\sum \sigma_i^2$. Setting $\tau^2 = \gamma d L \sigma_n^2$ ensures that the discarded tail singular values $\sum_{i=k+1}^n \sigma_i^2$ account precisely for the noise energy. In smooth regions where all patches are identical, the signal rank is 1 and $k$ automatically drops to 1. In complex textured regions, $k$ automatically increases to retain multiple structural components, avoiding arbitrary manual rank tuning.

---

### Q9: In Figures 21 and 22, why does increasing $L$ beyond 50 ($L=55, 60$) cause performance to plateau or slightly degrade?
**Answer**:
The non-local patch grouping algorithm searches for the $L$ closest candidate patches within a local spatial window ($31 \times 31$). In structured regions (such as edges, corners, or isolated targets), the number of genuinely similar patches in the neighborhood is finite. Once $L$ exceeds this natural redundancy limit ($L > 50$), the algorithm is forced to include dissimilar patches with larger Euclidean distances. These dissimilar patches introduce outlier vectors into the observation matrix $\mathbf{G}$, diluting the dominant low-rank subspace and causing subtle edge blurring upon back-projection. Additionally, SVD latency scales with $L$, making $L > 50$ counterproductive.

---

### Q10: How does the pipeline maintain operational stability under severe stress conditions ($\sigma_s = 40, \sigma_n = 20$) as shown in Figures 19 and 20?
**Answer**:
The stability arises from the decoupling architecture between Stage 1 and Stage 2:
1. **Stage 1 (Stripe Isolation)**: Under extreme stripe corruption ($\sigma_s = 40$), the column disparities completely dominate scene variations. The Schatten-$1/2$ shrinkage operator specifically targets the low-rank singular value subspace of the columns, while the second-order horizontal TV prior acts as an adaptive smoothing filter that prevents stripe energy from entering $\mathbf{C}$.
2. **Stage 2 (Ambient Denoising)**: The intermediate image $\mathbf{C}_1$ from Stage 1 is stripe-free but contains residual ambient noise. Stage 2's non-local patch SVD aggregates self-similar patches across the image, exploiting spatial redundancy that isotropic filters cannot capture.
By decoupling directional low-rank destriping from isotropic patch denoising, each stage solves a well-conditioned subproblem, preventing error compounding even under catastrophic initial degradation.
