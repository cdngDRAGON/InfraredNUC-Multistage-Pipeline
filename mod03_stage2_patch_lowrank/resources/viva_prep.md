# Viva Voce Preparation Guide: Stage 2 Patch Low-Rank Prior Recovery

---

### Q1: Why was the patch size chosen as $11 \times 11$ ($B = 11$) rather than smaller ($5 \times 5, 7 \times 7$) or larger ($21 \times 21$)?
**Answer:**
The patch size $B = 11$ balances spatial discriminability, noise averaging power, and computational tractability:
1. **If too small ($5 \times 5$ or $7 \times 7$)**: The vectorized dimensionality ($d = 25$ or $49$) is too low. In noisy infrared images, tiny patches lack sufficient geometric context; noise fluctuations easily fool Euclidean distance matching, grouping geometrically dissimilar patches together and causing edge hallucination.
2. **If too large ($21 \times 21$)**: Vectorized dimension $d = 441$. Finding $L = 50$ mutually similar patches within a local search window becomes statistically impossible due to natural scene heterogeneity (the "curse of dimensionality"). SVD on $441 \times 50$ matrices is also computationally heavy.
3. **Optimality of $11 \times 11$ ($d = 121$)**: An $11 \times 11$ patch spans typical infrared structural features (e.g. pedestrian widths, window mullions, car tires) while containing enough pixels ($121$) to average out Gaussian noise effectively and ensure robust Euclidean similarity matching.

---

### Q2: What governs the selection of cluster size $L = 50$ similar patches? What happens if $L$ is too small or too large?
**Answer:**
1. **If $L$ is too small (e.g. $L < 15$)**: The observation matrix $\mathbf{G} \in \mathbb{R}^{121 \times L}$ has too few columns. The statistical sample size is insufficient to separate random noise variance from true structural singular modes. Noise averaging power scales as $\approx 1/\sqrt{L}$, so low $L$ leaves visible graininess.
2. **If $L$ is too large (e.g. $L > 150$)**: The algorithm is forced to include candidate patches with large Euclidean distances that are geometrically dissimilar to the reference patch. This "cluster contamination" violates the low-rank assumption, artificially inflating the rank $k$ and blurring fine thermal textures.
3. **Selection of $L = 50$**: Provides a high noise-suppression ratio ($\approx 1/\sqrt{50} \approx 0.14$) while ensuring that all 50 patches within a $31 \times 31$ search window share genuine geometric alignment.

---

### Q3: How does the Eckart-Young-Mirsky Theorem guarantee optimality for the rank-$k$ truncated matrix $\widehat{\mathbf{G}}_k$?
**Answer:**
The Eckart-Young-Mirsky Theorem proves that for any matrix $\mathbf{G} \in \mathbb{R}^{d \times L}$ with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_n \ge 0$, the truncated SVD expansion:
$$\widehat{\mathbf{G}}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$
is the unique global minimizer of the distance to the manifold of matrices of rank at most $k$:
$$\widehat{\mathbf{G}}_k = \arg\min_{\operatorname{rank}(\mathbf{X}) \le k} \|\mathbf{G} - \mathbf{X}\|_F$$
Furthermore, the residual error is bounded by the tail singular values:
$$\|\mathbf{G} - \widehat{\mathbf{G}}_k\|_F = \sqrt{\sum_{i=k+1}^n \sigma_i^2}$$
In Stage 2, setting $k$ according to the noise energy criterion guarantees that the reconstruction is the optimal $L_2$ approximation that discards purely noise energy without discarding structural signal modes.

---

### Q4: What is the physical meaning of the noise energy criterion in Eq. (26): $\sum_{i=k}^n \sigma_i^2 > \tau^2 \ge \sum_{i=k+1}^n \sigma_i^2$?
**Answer:**
In an observation matrix $\mathbf{G} = \mathbf{X}_G + \mathbf{N}_G$, the Frobenius norm of the residual error when approximating $\mathbf{G}$ by a rank-$k$ matrix is precisely the sum of the discarded tail singular values:
$$\|\mathbf{G} - \widehat{\mathbf{G}}_k\|_F^2 = \sum_{i=k+1}^n \sigma_i^2$$
The quantity $\tau^2 = \gamma (d - 1)(L - 1) \sigma^2$ is the expected total energy of pure Gaussian noise in the subspace.
- If we discard $n - k$ singular values, the discarded energy should not exceed the noise energy: $\sum_{i=k+1}^n \sigma_i^2 \le \tau^2$.
- At the same time, if we were to discard $n - (k - 1)$ singular values, the discarded energy would exceed the noise energy ($\sum_{i=k}^n \sigma_i^2 > \tau^2$), meaning we would inadvertently discard genuine structural signal energy.
Thus, Eq. (26) identifies the unique transition index $k$ where the signal subspace ends and the noise subspace begins.

---

### Q5: Why is mean-centering (removing column means) performed on the observation matrix $\mathbf{G}$ before SVD?
**Answer:**
Each patch $\mathbf{y}_c$ in $\mathbf{G}$ has a DC offset (mean intensity) representing local radiant temperature:
$$\mathbf{y}_c = \bar{y}_c \mathbf{1} + \tilde{\mathbf{y}}_c$$
If SVD is performed on uncentered $\mathbf{G}$, the first singular value $\sigma_1$ is dominated by the shared average DC brightness, rather than textural structure:
$$\sigma_1 \approx \sqrt{d L} \cdot \mu_{\text{patch}}$$
By subtracting column means $\mathbf{G}_{\text{centered}} = \mathbf{G} - \operatorname{mean}(\mathbf{G}, \text{axis}=0)$, the DC component is preserved separately. The SVD spectrum then reflects purely zero-mean geometric variations (edges, gradients, textures). The DC offsets are added back after low-rank reconstruction, preventing global brightness shifts.

---

### Q6: How does Stage 2 prevent image blurring around fine structures such as pedestrian contours and thin window mullions?
**Answer:**
1. **Adaptive Rank Allocation**: In smooth background regions (e.g. sky, road), all 50 patches are nearly flat, so the singular values decay immediately; the adaptive criterion selects $k = 1$ or $2$, aggressively filtering noise. In complex structural regions (e.g. pedestrian edges, window frames), patches contain sharp geometric transitions; the singular values decay more slowly, and the adaptive criterion selects a higher rank ($k = 8 - 14$), preserving edge sharpness.
2. **Cluster Weighting**: Groups with high rank complexity receive lower aggregation weight $w = 1 / (k + 0.1)$, preventing noisy clusters from polluting the reconstructed canvas.
3. **Kaiser Window Tapering**: Center pixels receive maximum weight while patch borders taper smoothly, avoiding edge smearing.

---

### Q7: Why is a 2D Kaiser window preferred over uniform boxcar averaging during patch aggregation?
**Answer:**
Uniform boxcar averaging gives equal weight to all $B^2 = 121$ pixels in a patch. However, pixels near patch boundaries suffer from boundary effects, truncated spatial context, and minor Euclidean matching misalignments. When boxcar patches are overlapped, these boundary discrepancies create noticeable grid-like seam artifacts.
The 2D separable Kaiser window:
$$W_{\text{kaiser}}(x, y) = w_{\text{kaiser}}(x) \cdot w_{\text{kaiser}}(y), \quad \beta = 2.0$$
places high weight on the reliable center of the patch and smoothly attenuates to near-zero at the outer perimeter. Aggregation satisfies the partition of unity, ensuring continuous, seam-free reconstructed transitions.

---

### Q8: How does multi-threading via `joblib.Parallel` mirror MATLAB's `Parfor` described in Table II and Fig. 23 of the paper?
**Answer:**
In Stage 2, the operations for each reference patch $\mathbf{y}_j$ (local search, Euclidean distance calculation, sorting, SVD, and adaptive rank truncation) depend only on the static input image $\mathbf{C}^{(1)}$ and are completely independent across reference patch indices $j$.
In MATLAB, this "embarrassingly parallel" loop is executed via `parfor j = 1:N_patches`.
In Python, `joblib.Parallel(n_jobs=-1, prefer="threads")` shards the reference patch coordinates across available CPU cores using work-stealing thread pools. This achieves linear scaling with core count, reducing execution time from $\approx 15$ seconds to $< 1.8$ seconds on a modern multi-core processor.

---

### Q9: What is the computational complexity of Stage 2 as a function of image size $(H, W)$, patch size $B$, and cluster size $L$?
**Answer:**
Let $N_{\text{ref}} = \frac{H - B}{\text{stride}} \times \frac{W - B}{\text{stride}}$ be the number of reference patches.
For each reference patch:
1. **Patch Search**: $(2R + 1)^2$ candidate distance calculations, each requiring $O(B^2)$ operations $\implies O(R^2 B^2)$.
2. **Top-$L$ Selection**: Quickselect / partial sort $\implies O(R^2 + L \log L)$.
3. **SVD of $\mathbf{G} \in \mathbb{R}^{d \times L}$**: $O(d L^2) = O(B^2 L^2)$ operations.
4. **Reconstruction**: $O(d L k) = O(B^2 L k)$ operations.

Total computational complexity is:
$$\mathcal{O}\left( N_{\text{ref}} \cdot \left[ R^2 B^2 + B^2 L^2 \right] \right)$$
For $B = 11, L = 50, R = 15$, $B^2 L^2 = 121 \times 2500 \approx 3 \times 10^5$ FLOPS per reference patch. With $N_{\text{ref}} \approx 6400$ patches and multi-core parallelism, total execution finishes in under 2 seconds.

---

### Q10: Can Stage 2 replace Stage 1 entirely? What happens if you apply Stage 2 directly to a raw striped infrared image?
**Answer:**
**No, Stage 2 cannot replace Stage 1.**
If Stage 2 is applied directly to a raw striped image $\mathbf{D} = \mathbf{C} + \mathbf{S} + \mathbf{N}$:
1. **Stripe Matching Failure**: The vertical stripe noise is correlated across the entire column. In an $11 \times 11$ patch, the vertical stripes appear as fixed vertical bands.
2. When searching for similar patches in the search window, candidate patches that happen to align with the same vertical column pattern will have low Euclidean distance, grouping the stripe pattern into the observation matrix $\mathbf{G}$.
3. Because the stripes are identical across grouped patches, the stripe energy projects onto the dominant singular values ($\sigma_1, \sigma_2$) of $\mathbf{G}$. The low-rank approximation $\widehat{\mathbf{G}}_k$ faithfully **retains and preserves the stripe noise**, failing to remove it.

Stage 1 is essential because its global directional prior and Schatten-$1/2$ column-projection strip out the column-wide line noise $\mathbf{S}$. Stage 2 then operates on the destriped image to clean up the residual isotropic noise $\mathbf{N}$.
