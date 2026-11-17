# Theoretical Guide: Non-Local Patch Self-Similarity & Adaptive Low-Rank Truncation

---

## 1. Non-Local Self-Similarity (NSS) in Thermal Imaging

Natural thermal infrared radiant fields exhibit high spatial redundancy. Because thermal radiation obeys heat diffusion and conduction equations across homogeneous materials (e.g. walls, sky, pavements, vehicle surfaces), local temperature variations form smooth geometric manifolds:

```
    Thermal Infrared Image C(x, y)
    ┌───────────────────────────┐
    │  [Patch y_j]              │   <-- Reference patch (e.g. wall texture)
    │     ...                   │
    │         [Patch y_c1]      │   <-- Similar patch elsewhere
    │     ...                   │
    │              [Patch y_c2] │   <-- Similar patch with identical edge angle
    └───────────────────────────┘
                 │
                 ▼ Vectorization (11 x 11 -> 121 x 1)
    Observation Matrix G in R^{121 x 50}
    ┌───────┬─────────┬─────────┬─────┐
    │  y_j  │  y_c1   │  y_c2   │ ... │  <-- Strongly correlated columns!
    └───────┴─────────┴─────────┴─────┘
                 │
                 ▼ SVD
    Singular values: sigma_1 >> sigma_2 > sigma_3 >> ... -> 0
    (Clean patch manifold has rank k ~ 1-4)
```

### 1.1 Patch Vectorization and Matrix Stacking (Eqs. 22–23)
For a reference patch $\mathbf{y}_j \in \mathbb{R}^{B \times B}$ ($B = 11$, $d = 121$ pixels):
1. Candidate patches $\mathbf{y}_c$ within local radius $R = 15$ are compared using Euclidean distance:
   $$H(\mathbf{y}_j, \mathbf{y}_c) = \|\mathbf{y}_j - \mathbf{y}_c\|_2^2$$
2. Selecting the $L = 50$ smallest distances forms the observation matrix:
   $$\mathbf{G} = [\mathbf{y}_j, \mathbf{y}_{c,1}, \dots, \mathbf{y}_{c,L-1}] \in \mathbb{R}^{d \times L}$$

### 1.2 Mathematical Low-Rankness of $\mathbf{G}$
Under the manifold hypothesis, all $L$ patches in $\mathbf{G}$ are noisy observations of an underlying clean patch manifold $\mathbf{X}_G$:
$$\mathbf{G} = \mathbf{X}_G + \mathbf{N}_G$$
Since the patches in the cluster share nearly identical edge orientation, phase, and intensity gradients:
- $\mathbf{X}_G$ is spanned by a very small number of basis vectors: $\operatorname{rank}(\mathbf{X}_G) = r \ll \min(d, L)$.
- In contrast, the noise $\mathbf{N}_G$ has independent components $N_{ij} \sim \mathcal{N}(0, \sigma^2)$, which disperse energy uniformly across all $n = \min(d, L) = 50$ singular modes.

---

## 2. Derivation of the Adaptive SVD Rank Threshold (Eq. 26)

### 2.1 The Eckart-Young-Mirsky Low-Rank Approximation
Let the SVD of $\mathbf{G}$ be:
$$\mathbf{G} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \sum_{i=1}^n \sigma_i \mathbf{u}_i \mathbf{v}_i^T, \quad \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_n \ge 0$$
The rank-$k$ approximation $\widehat{\mathbf{G}}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ discards the tail singular components $\{k+1, \dots, n\}$.
By the Eckart-Young-Mirsky Theorem, the discarded energy equals the residual Frobenius error:
$$\|\mathbf{G} - \widehat{\mathbf{G}}_k\|_F^2 = \sum_{i=k+1}^n \sigma_i^2$$

### 2.2 Noise Energy in the Discarded Subspace
If the discarded subspace $\{k+1, \dots, n\}$ contains purely noise, its expected energy is:
$$\mathbb{E}\left[ \sum_{i=k+1}^n \sigma_i^2 \right] = \sum_{i=k+1}^n \mathbb{E}[\sigma_i^2]$$
For a pure Gaussian noise matrix $\mathbf{N}_G \in \mathbb{R}^{d \times L}$ with zero mean and variance $\sigma^2$:
$$\mathbb{E}[\|\mathbf{N}_G\|_F^2] = d \cdot L \cdot \sigma^2$$
After mean-centering (removing the DC component), the degrees of freedom become $(d - 1)(L - 1)$, giving the expected noise energy:
$$\tau^2 = \gamma (d - 1)(L - 1) \sigma^2$$
where $\gamma \approx 0.95 - 1.0$ is an empirical calibration factor.

### 2.3 The Adaptive Energy Criterion (Eq. 26)
To separate genuine signal components from noise:
1. As rank $k$ decreases, more energy is assigned to the residual: the tail sum $\sum_{i=k+1}^n \sigma_i^2$ increases monotonically.
2. The optimal truncation rank $k$ is the smallest integer such that the residual error does not exceed the expected noise energy $\tau^2$:
   $$\sum_{i=k+1}^n \sigma_i^2 \le \tau^2$$
3. Simultaneously, retaining one fewer component ($k-1$) would cause the residual error to exceed the noise energy:
   $$\sum_{i=k}^n \sigma_i^2 > \tau^2$$
Combining both inequalities yields **Equation (26)**:
$$\sum_{i=k}^n \sigma_i^2 > \tau^2 \ge \sum_{i=k+1}^n \sigma_i^2 \quad \blacksquare$$

---

## 3. Kaiser Window Aggregation & Partition of Unity

### 3.1 Eliminating Boundary Discontinuities
When overlapping patches are averaged uniformly, high-frequency errors at patch boundaries create grid-like seam artifacts.
To enforce smooth continuity, we weight each patch by a 2D separable Kaiser window:
$$w_{\text{kaiser}}(x) = \frac{I_0\left( \beta \sqrt{1 - \left(\frac{2x}{B-1} - 1\right)^2} \right)}{I_0(\beta)}, \quad 0 \le x < B$$
$$W_{\text{kaiser}}(x, y) = w_{\text{kaiser}}(x) \cdot w_{\text{kaiser}}(y)$$
where $I_0(\cdot)$ is the zeroth-order modified Bessel function of the first kind, and $\beta = 2.0$ provides smooth attenuation towards the patch boundaries.

### 3.2 Normalized Back-Projection
For each pixel $(x, y)$, the final pixel intensity is normalized by the accumulated sum of spatial weights:
$$\mathbf{C}^{(2)}(x, y) = \frac{\sum_p w_p W_{\text{kaiser}}(x - x_p, y - y_p) \widehat{\mathbf{y}}_p(x - x_p, y - y_p)}{\sum_p w_p W_{\text{kaiser}}(x - x_p, y - y_p)}$$
This satisfies the partition of unity property, eliminating boundary seams while preserving sharp edge transitions.
