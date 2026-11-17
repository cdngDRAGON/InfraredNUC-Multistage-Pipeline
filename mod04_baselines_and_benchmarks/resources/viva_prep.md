# Viva Voce Preparation: Module 4 Baselines and Benchmarks

## Question 1: Why does 1D Guided Filtering (1D-GF) severely blur genuine vertical edges in thermal imagery?
**Answer:** 1D-GF operates row-by-row along the horizontal axis, assuming that horizontal gradient fluctuations with high spatial frequency represent stripe noise, while low-frequency components represent the true scene. However, sharp vertical structures (such as edges of buildings, telephone poles, or pedestrian silhouettes) also generate step discontinuities with high horizontal frequencies. Because 1D-GF lacks 2D context or vertical derivative awareness, it treats genuine vertical edges identically to vertical stripes, smoothing them out and inflating the Average Vertical Gradient Error (AVGE).

---

## Question 2: Why does BM3D achieve excellent Gaussian denoising but fail catastrophically on infrared stripe noise?
**Answer:** BM3D relies on 3D transform-domain collaborative filtering: it searches the entire image for mutually similar 2D patches and groups them into a 3D volume. For independent Gaussian noise, coefficients in the 3D transform domain are small and sparsely distributed, making hard thresholding highly effective. However, vertical stripe noise is rigidly column-correlated—it repeats identical vertical line patterns across adjacent image rows. BM3D groups these matching stripe patterns together, interpreting them as authentic repeating textures. Consequently, BM3D smooths random background noise but leaves vertical stripes almost completely untouched, resulting in an unacceptably high roughness index ($\rho > 0.18$).

---

## Question 3: What is the fundamental mathematical reason why ADOM produces horizontal staircasing artifacts?
**Answer:** ADOM employs first-order horizontal Total Variation: $\|\mathbf{W} \odot \nabla_x \mathbf{C}\|_1$. In $L_1$ total variation minimization, the optimal solution for a gradient-penalized variable is piecewise constant. When reconstructing intensity across corrupted vertical columns, first-order TV forces adjacent pixels to collapse to identical flat values, forming discrete stair-step plateaus across smooth gradients (e.g. across clear sky or thermal monitor gradients). Our proposed framework resolves this by penalizing the **second-order directional horizontal derivative** $|\frac{\partial^2 \mathbf{C}}{\partial x^2}|$, which forces the horizontal curvature to be continuous, creating a smooth cubic interpolation bridge across stripes without flat terraces.

---

## Question 4: How does LRSID's Nuclear Norm prior on the stripe matrix differ from our Schatten-$1/2$ prior?
**Answer:**
- LRSID uses the convex Nuclear Norm $\|\mathbf{S}\|_* = \sum_i \sigma_i(\mathbf{S})$. The corresponding proximal operator is singular value soft-thresholding $\mathcal{D}_\tau(\sigma) = \max(\sigma - \tau, 0)$, which subtracts a constant threshold $\tau$ from every singular value. Because dominant singular values represent major scene contrast, subtracting $\tau$ introduces linear shrinkage bias, pulling true scene contrast into the stripe matrix and dulling image contrast.
- In contrast, our Schatten-$1/2$ quasi-norm $\|\mathbf{S}\|_{S_{1/2}}^{1/2} = \sum_i \sigma_i^{1/2}$ is non-convex. Its derivative $\frac{d}{d\sigma} \sigma^{1/2} = \frac{1}{2\sqrt{\sigma}} \to 0$ as $\sigma \to \infty$. Consequently, large singular values experience virtually zero shrinkage bias, while small noise-induced singular values are aggressively truncated to exact zero. This produces sharper rank separation and authentic radiometric contrast preservation.

---

## Question 5: What is the group sparsity regularizer in GSRSR, and where does it break down?
**Answer:** GSRSR enforces column-wise group sparsity via the $L_{2,1}$ norm: $\|\mathbf{S}\|_{2,1} = \sum_{j=1}^W \|\mathbf{S}_{:, j}\|_2$. Its proximal operator applies block-thresholding to entire columns: $\operatorname{prox}_{\tau}(S_{:, j}) = \max(1 - \tau / \|S_{:, j}\|_2, 0) S_{:, j}$.
It breaks down in two scenarios:
1. When environmental noise creates local fluctuations along a column, the assumption of uniform column energy fails.
2. In scenes with high-contrast localized thermal hot-spots, the $L_2$ norm of the column is inflated by the true object, preventing the threshold from triggering and leaving prominent stripes directly above and below the hot object.

---

## Question 6: What are the distinct failure modes of deep learning models like SNRWDNN and DestripeCycleGAN?
**Answer:**
- **SNRWDNN** (Wavelet Deep Neural Network) filters the horizontal detail subband ($LH$) of a 2D wavelet transform. While it preserves vertical edges well, hard directional subband thresholding produces high-frequency wavelet packet oscillations and pseudo-Gibbs ringing around sharp point sources.
- **DestripeCycleGAN** uses unpaired adversarial translation. While the discriminator forces the generator to produce stripe-free imagery, the adversarial loss frequently hallucinates fine non-existent surface micro-textures and shifts the mean global luminance in flat sky or pavement regions, degrading radiometric temperature calibration (elevated AAD and MRD).

---

## Question 7: Why is a two-stage pipeline necessary instead of solving a single unified optimization problem?
**Answer:** Stripe noise and environmental noise occupy fundamentally different mathematical spaces:
1. Stripe noise $\mathbf{S}$ is **globally structured**, **unidirectional**, and **low-rank** ($\operatorname{rank} \le 2$). It is best resolved in the full-image spatial/Fourier domain via directional operators and matrix factorization.
2. Environmental noise $\mathbf{N}$ is **locally isotropic**, **full-rank**, and **random**. It is best resolved in the non-local patch domain using natural thermal self-similarity.
Attempting to solve both simultaneously in a single objective leads to conflicting regularization penalties (e.g. high TV smoothing blurs textures, while weak TV leaves stripes). Decoupling into Stage 1 (ADMM destriping) and Stage 2 (Patch Low-Rank SVD) achieves optimal separation without compromise.

---

## Question 8: How do we interpret the 3D Global Energy surface plots in Figs. 10–15?
**Answer:** The 3D surface plot renders the spatial pixel intensity topography $Z = \mathbf{C}(X, Y)$ using the `jet` colormap:
- In the degraded input, stripe noise appears as severe, high-frequency, knife-edge ridges and troughs running along the vertical direction.
- In poorly corrected methods (e.g. 1D-GF, BM3D), residual corrugated waves remain visible across flat plateau regions.
- In our method, the 3D surface is pristine: flat backgrounds form perfectly smooth, level plains, while authentic structural peaks (e.g. pedestrian or vehicle hot-spots) stand out crisply with steep, well-defined slopes.

---

## Question 9: In Table I, why does our method achieve both minimal AVGE and minimal Roughness $\rho$?
**Answer:**
- **Roughness $\rho$** (Eq. 28) measures total high-frequency 1D differences. Our method minimizes $\rho$ because Stage 1 eliminates column-to-column intensity oscillations, while Stage 2 suppresses residual pixel variance.
- **AVGE** (Eq. 27) measures the preservation of vertical gradients $|\nabla_y \mathbf{C}_{\mathrm{est}} - \nabla_y \mathbf{C}_{\mathrm{orig}}|$. Because our Stage 1 objective explicitly isolates the second-order horizontal derivative $|\frac{\partial^2 \mathbf{C}}{\partial x^2}|$ and places zero penalty on $\nabla_y \mathbf{C}$, genuine vertical structures undergo zero horizontal smearing, keeping AVGE near zero.

---

## Question 10: How does our method ensure that radiometric temperature measurements are trustworthy?
**Answer:** Radiometric temperature measurement depends on two factors:
1. **Low MRD (Mean Relative Deviation)**: Ensuring that the average pixel temperature does not systematically shift up or down.
2. **High ICV (Inverse Coefficient of Variation)**: Ensuring that flat calibration targets or homogeneous backgrounds have minimal spatial temperature variance ($\mu / \sigma$).
Our method preserves local column averages in Stage 1 and uses adaptive noise energy tail truncation ($\tau^2$) in Stage 2, achieving an average $\mathrm{MRD} = 0.043$ and the highest $\mathrm{ICV} = 27.67$, ensuring radiometric accuracy for critical thermal inspection tasks.
