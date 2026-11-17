# Viva Voce Preparation Guide: IRFPA Nonuniformity, SVD Low-Rankness & Directional Priors

---

### Q1: What is the physical origin of column-correlated stripe noise in modern IRFPAs, and why is it not observed in typical visible-light CMOS sensors?
**Answer:**
Infrared Focal Plane Arrays (IRFPAs)—particularly uncooled microbolometers ($\text{VO}_x$, $\alpha\text{-Si}$) and cooled MCT/QWIP detectors—operate in high-emittance, low-contrast infrared regimes where the background radiant flux dominates the incident signal. To accommodate high frame rates and minimize power consumption, the Read-Out Integrated Circuit (ROIC) utilizes a **column-parallel readout architecture** where an entire column of microbolometer pixels shares a single readout line, pre-amplifier (CTIA), and sample-and-hold circuit.

Due to lithographic and material fabrication tolerances, subtle threshold voltage ($\Delta V_{\text{th}}$) and transconductance ($\Delta g_m$) mismatches occur between adjacent column amplifiers. Furthermore, $1/f$ low-frequency flicker noise and ambient thermal drift cause column biases to wander over time independently. In visible-light CMOS sensors, the high signal-to-noise ratio, correlated double sampling (CDS) circuits, and low temperature-sensitivity make column FPN negligible, whereas in infrared detectors, column FPN often exceeds true target radiometric temperature variations.

---

### Q2: Why does conventional isotropic Total Variation (TV) destriping lead to severe "staircase artifacts" and horizontal edge blurring?
**Answer:**
Isotropic Total Variation regularizes the image by penalizing the L1-norm of the spatial gradient:
$$\mathcal{R}_{\text{TV}}(C) = \int_\Omega \|\nabla C\|_2 \, dx \, dy$$
1. **Staircasing**: Because the L1-norm penalizes non-zero gradients equally regardless of their magnitude (subgradient is $\pm 1$), it favors piecewise-constant solutions over smooth ramps. Consequently, natural continuous thermal gradients (such as sky atmospheric lapse rates or conduction ramps across warm bodies) are discretized into artificial flat plateaus separated by spurious jumps ("staircases").
2. **Horizontal Edge Blurring**: Isotropic TV treats horizontal ($\nabla_x$) and vertical ($\nabla_y$) gradients symmetrically. Because vertical stripes create enormous cross-column gradients $\nabla_x S$, an isotropic regularizer penalizes all horizontal variation heavily, effectively smoothing out genuine vertical boundaries of buildings, pedestrians, and vehicles along with the noise.

---

### Q3: State and explain the Eckart-Young-Mirsky Theorem in the context of infrared image decomposition (Eqs. 1–4).
**Answer:**
The Eckart-Young-Mirsky Theorem establishes that for any matrix $\mathbf{J} \in \mathbb{R}^{M \times N}$ with Singular Value Decomposition $\mathbf{J} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$, the closest matrix of rank at most $k$ ($k < r$) in the Frobenius norm is the truncated SVD expansion:
$$\widehat{\mathbf{J}}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \arg\min_{\text{rank}(\mathbf{X}) \le k} \|\mathbf{J} - \mathbf{X}\|_F$$
The minimal approximation error equals the square root of the sum of discarded squared singular values:
$$\|\mathbf{J} - \widehat{\mathbf{J}}_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$
In infrared image destriping, this theorem proves that both the clean thermal scene $\mathbf{C}$ and the stripe component $\mathbf{S}$ can be represented by low-dimensional subspace projections. Specifically, because the singular values $\sigma_i$ of $\mathbf{C}$ decay exponentially, truncating to rank $k$ preserves the principal structural energy while attenuating residual unstructured random noise.

---

### Q4: Why is the stripe noise matrix $\mathbf{S}$ strictly low-rank ($\text{rank}(\mathbf{S}) \le 3$), and how does this relate to sensor hardware?
**Answer:**
In an ideal sensor without row drift, the stripe noise on column $j$ is identical across all rows $i \in \{1, \dots, M\}$:
$$S(i, j) = s_j \implies \mathbf{S} = \mathbf{1}_M \mathbf{s}_{\text{col}}^T$$
where $\mathbf{1}_M = [1, 1, \dots, 1]^T \in \mathbb{R}^M$ and $\mathbf{s}_{\text{col}} = [s_1, \dots, s_N]^T \in \mathbb{R}^N$. This outer product is an exact **rank-1 matrix** with only one non-zero singular value:
$$\sigma_1 = \sqrt{M} \|\mathbf{s}_{\text{col}}\|_2, \quad \sigma_2 = \dots = \sigma_N = 0$$
In practical hardware, minor thermal gradients or timing shifts introduce subtle row-dependent modulations $\delta_S(i, j) \approx y_i \cdot \Delta s_j$, which adds at most a linear or quadratic trend across rows, introducing at most 1 or 2 additional singular values. Hence, $\text{rank}(\mathbf{S}) \le 3$, and its top 2 singular values capture $>99.99\%$ of its Frobenius energy.

---

### Q5: Mathematically and physically, why does the vertical gradient $\nabla_y \mathbf{D}$ preserve genuine scene structures, while $\nabla_x \mathbf{D}$ is heavily corrupted?
**Answer:**
Given the degradation model $\mathbf{D} = \mathbf{C} + \mathbf{S} + \mathbf{N}$, consider the spatial difference operators along the columns ($x$) and rows ($y$):
- Along columns ($x$): Since each column has a different gain/offset $s_j \neq s_{j+1}$, the horizontal difference of the stripe field is non-zero:
  $$\nabla_x S(i, j) = s_{j+1} - s_j \neq 0$$
  This produces high-frequency, large-magnitude spikes in $\nabla_x \mathbf{D}$.
- Along rows ($y$): Because pixels in the same column share the same readout amplifier and bias $s_j$, the vertical difference of the stripe field vanishes:
  $$\nabla_y S(i, j) = S(i+1, j) - S(i, j) = s_j - s_j = 0$$
  Therefore:
  $$\nabla_y \mathbf{D} = \nabla_y \mathbf{C} + \nabla_y \mathbf{N}$$
  Aside from zero-mean Gaussian noise $\nabla_y \mathbf{N}$, the vertical gradient of the degraded image directly reflects the true vertical edges and textures of the clean thermal scene.

---

### Q6: What is the theoretical advantage of utilizing 2nd-order directional derivatives ($\frac{\partial^2}{\partial y^2}$) over 1st-order derivatives ($\frac{\partial}{\partial y}$)?
**Answer:**
1. **Preservation of Affine/Smooth Ramps**: The kernel of the first derivative operator $\nabla_y f = 0$ is the set of constant functions $f(y) = c$. The kernel of the second derivative operator $\frac{\partial^2 f}{\partial y^2} = 0$ is the set of affine functions $f(y) = a \cdot y + b$. Thus, regularizing with $\frac{\partial^2}{\partial y^2}$ penalizes curvature (acceleration) rather than slope (velocity), naturally preserving smooth linear thermal gradients without converting them into piecewise flat stairs.
2. **Insulation from Vertical Stripes**: Since $\nabla_y S \approx 0$, differentiating again along $y$ yields:
   $$\frac{\partial^2 S}{\partial y^2} = \frac{\partial}{\partial y}\left(\frac{\partial S}{\partial y}\right) \approx 0$$
   This ensures that the high-order vertical prior provides an uncorrupted channel for enforcing structural continuity in the restoration objective.

---

### Q7: Explain the three degradation cases defined in Figure 9 of the paper. Why are all three necessary to evaluate an infrared NUC algorithm?
**Answer:**
1. **Case 1 (Conventional FPN)**: Simulates standard microbolometer aperiodic column gain/offset variations ($\sigma_{\text{stripe}} = 12.0, \sigma_{\text{env}} = 5.0$). This represents typical lab/field environments with ambient drift.
2. **Case 2 (Periodic Stripe Noise)**: Simulates harmonic interference ($A = 15.0, f = 0.18, \sigma_{\text{env}} = 6.0$) caused by ROIC multiplexer clock crosstalk, chopper wheel phase synchronization, or multi-channel ADC interleaving.
3. **Case 3 (Enhanced Severe Stripe Noise)**: Simulates extreme environmental stress ($A = 28.0, f = 0.22, \sigma_{\text{env}} = 10.0$) including defective, dead, or saturated columns with large impulse spikes ($\pm 35.0$).

Evaluating all three is essential because an algorithm that performs well on stationary Gaussian stripes (Case 1) often fails completely on periodic resonance (Case 2) due to Fourier harmonic leakage, or hallucinates artifacts when encountering saturated defective lines (Case 3).

---

### Q8: What is the fundamental difference between additive fixed-pattern noise (FPN) and multiplicative gain nonuniformity?
**Answer:**
At the detector level, a microbolometer pixel's response to incident scene irradiance $\Phi(i, j)$ is governed by a linear model:
$$Y(i, j) = g(i, j) \cdot \Phi(i, j) + o(i, j) + n(i, j)$$
where $g(i, j)$ is the detector responsivity (gain), $o(i, j)$ is the dark-current offset, and $n$ is temporal noise.
- **Additive FPN**: Represents variations in the offset term $o(i, j)$, which is independent of incident irradiance.
- **Multiplicative Nonuniformity**: Variations in gain $g(i, j)$, where stripe intensity scales proportionally with scene brightness.
In single-frame destriping, when scene dynamic range is moderately uniform or when taking the logarithm/first-order Taylor expansion around the mean irradiance $\mu_\Phi$:
$$g(i, j) \Phi(i, j) \approx \bar{g} \Phi(i, j) + \Delta g(i, j) \mu_\Phi$$
the multiplicative component is effectively mapped into an equivalent additive column stripe field $S(i, j) \approx o(i, j) + \Delta g(i, j) \mu_\Phi$, making the additive model $D = C + S + N$ mathematically and practically effective.

---

### Q9: How can an algorithm distinguish between genuine high-frequency vertical scene structures (e.g., window frames, fence pickets, architectural columns) and vertical stripe noise?
**Answer:**
Stripe noise and vertical scene structures differ in three fundamental geometric and statistical properties:
1. **Column Continuity**: Stripe noise spans the entire height of the detector array ($M$ rows) continuously because it originates in the physical column readout bus. In contrast, natural objects (window mullions, fence pickets) terminate at building or ground boundaries.
2. **Low-Rank Subspace**: Stripe noise lives in an extremely low-rank column subspace ($\text{rank} \le 2$). Genuine scene structures exhibit 2D variations and high local rank when analyzed across overlapping spatial patches.
3. **Nonlocal Patch Self-Similarity**: Natural scene textures exhibit repetitive 2D similarity across spatial neighborhoods, whereas column stripes do not have horizontal translational replicas at matching offsets. Exploiting patch-level low-rank tensor decomposition (Stage 2) separates genuine textures from global sensor stripes.

---

### Q10: Why is single-frame nonuniformity correction (NUC) preferred over multi-frame scene-based NUC in dynamic or resource-constrained platforms?
**Answer:**
1. **No Ghosting Artifacts**: Multi-frame scene-based NUC algorithms (such as LMS, Kalman filters, or inter-frame motion estimation) require significant camera or scene motion to average out stationary features. If an object remains stationary (e.g. a parked car, human sentry, or horizon), multi-frame algorithms burn the object into the offset map, creating persistent **ghosting artifacts** when the camera pans. Single-frame NUC operates purely on the instantaneous frame and is immune to ghosting.
2. **Zero Latency and Minimal Memory**: Multi-frame methods require storing historical frame buffers, motion vectors, and recursive state matrices, which imposes severe memory and bandwidth bottlenecks on SWaP-constrained (Size, Weight, and Power) edge hardware. Single-frame NUC processes frames independently in real-time.
3. **Robustness to Rapid Thermal Shocks**: When a sensor undergoes rapid temperature swings (e.g. missile seeker launch, UAV altitude change), the detector offsets drift faster than multi-frame recursive filters can adapt, whereas single-frame optimization instantly recalibrates to the current frame's noise state.
