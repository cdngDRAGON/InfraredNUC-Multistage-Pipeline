# Theoretical and Mathematical Foundations: Nonuniformity Modeling & Prior Information

---

## 1. Physical Origins of IRFPA Nonuniformity and Stripe Noise

Infrared Focal Plane Arrays (IRFPAs) operate in either the Mid-Wave Infrared (MWIR: 3–5 $\mu$m) or Long-Wave Infrared (LWIR: 8–14 $\mu$m) spectral regimes. Unlike visible-light silicon CMOS sensors, thermal detectors—predominantly uncooled vanadium oxide ($\text{VO}_x$) / amorphous silicon ($\alpha\text{-Si}$) microbolometers and cooled Mercury Cadmium Telluride ($\text{HgCdTe}$) / Quantum Well Infrared Photodetectors (QWIP)—exhibit substantial pixel-to-pixel response heterogeneity.

```
       Microbolometer Pixel Array (M rows x N columns)
       ┌─────┬─────┬─────┬─────┬─────┬─────┐
Row 0  │ P00 │ P01 │ P02 │ P03 │ P04 │ ... │
       ├─────┼─────┼─────┼─────┼─────┼─────┤
Row 1  │ P10 │ P11 │ P12 │ P13 │ P14 │ ... │
       ├─────┼─────┼─────┼─────┼─────┼─────┤
...    │ ... │ ... │ ... │ ... │ ... │ ... │
       └─────┴─────┴─────┴─────┴─────┴─────┘
          │     │     │     │     │
          ▼     ▼     ▼     ▼     ▼
       ┌─────────────────────────────────┐
       │ Column-Parallel Readout (ROIC)  │
       │ Pre-Amps / CTIAs / Sample & Hold│  <-- Column-to-column gain & offset disparity
       └─────────────────────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │  High-Speed A/D Converters      │  <-- Multi-channel multiplexing harmonics
       └─────────────────────────────────┘
```

### 1.1 Read-Out Integrated Circuit (ROIC) Architecture
To manage high frame rates while limiting power dissipation, IRFPAs employ a **column-parallel readout topology**:
1. All $M$ pixels in column $j$ share common column buslines, column pre-amplifiers (Capacitive Transimpedance Amplifiers, CTIA), and sample-and-hold stages.
2. Even with high-precision fabrication, lithographic variations induce threshold voltage mismatch ($\Delta V_{\text{th}}$) and transconductance mismatch ($\Delta g_m$) between adjacent column amplifiers.
3. This manifests as column-dependent bias and gain discrepancies across columns $j \in \{1, \dots, N\}$.

### 1.2 Thermal Drift & $1/f$ Low-Frequency Fluctuation
Microbolometers measure temperature changes via resistance variation:
$$R(T) = R_0 \exp\left(\frac{B}{T}\right)$$
Because the thermal time constant $\tau_{\text{th}} \approx 8 - 15$ ms and ambient temperature fluctuates continuously, detector bias drifts over time with a power spectral density $S_v(f) \propto 1/f^\alpha$ ($\alpha \approx 1 - 2$). This low-frequency flicker noise causes column offsets to drift independently, generating persistent vertical stripes that degrade standard static two-point calibration within minutes of sensor operation.

---

## 2. Rigorous Proof of the Eckart-Young-Mirsky Theorem (Eqs. 1–4)

### Theorem Statement
Let $\mathbf{J} \in \mathbb{R}^{M \times N}$ be a matrix of rank $r \le \min(M, N)$ with Singular Value Decomposition (SVD):
$$\mathbf{J} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T, \quad \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$$
Then for any $k < r$, the optimal rank-$k$ approximation under the Frobenius norm:
$$\widehat{\mathbf{J}}_k = \arg\min_{\text{rank}(\mathbf{X}) \le k} \|\mathbf{J} - \mathbf{X}\|_F$$
is uniquely achieved by truncating the SVD to the top $k$ components:
$$\widehat{\mathbf{J}}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$
and the minimum reconstruction error is given exactly by:
$$\min_{\text{rank}(\mathbf{X}) \le k} \|\mathbf{J} - \mathbf{X}\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

### Proof
**Step 1: Unitary Invariance of the Frobenius Norm**
Recall that for any orthogonal matrices $\mathbf{U} \in \mathbb{R}^{M \times M}$ and $\mathbf{V} \in \mathbb{R}^{N \times N}$, the Frobenius norm is unitarily invariant:
$$\|\mathbf{J} - \mathbf{X}\|_F^2 = \|\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T - \mathbf{X}\|_F^2 = \|\mathbf{\Sigma} - \mathbf{U}^T \mathbf{X} \mathbf{V}\|_F^2$$
Define $\mathbf{Y} = \mathbf{U}^T \mathbf{X} \mathbf{V}$. Since $\mathbf{U}$ and $\mathbf{V}$ are non-singular, $\text{rank}(\mathbf{Y}) = \text{rank}(\mathbf{X}) \le k$.
Thus, the problem is equivalent to minimizing:
$$\min_{\text{rank}(\mathbf{Y}) \le k} \|\mathbf{\Sigma} - \mathbf{Y}\|_F^2 = \min_{\text{rank}(\mathbf{Y}) \le k} \left( \sum_{i=1}^r (\sigma_i - Y_{ii})^2 + \sum_{i \neq j} Y_{ij}^2 \right)$$

**Step 2: Off-Diagonal Elements**
To minimize the sum, all off-diagonal entries must vanish: $Y_{ij} = 0$ for all $i \neq j$.
Furthermore, for all diagonal elements $i > r$, $\sigma_i = 0$, so $Y_{ii} = 0$.
Thus, $\mathbf{Y}$ must be a diagonal matrix with at most $k$ non-zero entries on its diagonal:
$$\|\mathbf{\Sigma} - \mathbf{Y}\|_F^2 = \sum_{i=1}^r (\sigma_i - Y_{ii})^2$$

**Step 3: Optimal Choice of Diagonal Entries**
To minimize $\sum_{i=1}^r (\sigma_i - Y_{ii})^2$ subject to having at most $k$ non-zero $Y_{ii}$:
- For the $k$ chosen indices, setting $Y_{ii} = \sigma_i$ achieves zero error $(\sigma_i - Y_{ii})^2 = 0$.
- For the remaining $r - k$ unchosen indices, $Y_{ii} = 0$, incurring an error contribution of $\sigma_i^2$.
To minimize the total residual error $\sum_{i \notin \mathcal{K}} \sigma_i^2$, we must choose the indices with the largest values of $\sigma_i^2$.
Because $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$, the optimal index set is precisely the first $k$ singular values:
$$\mathcal{K}^* = \{1, 2, \dots, k\}$$
Therefore, the optimal diagonal matrix is $\mathbf{Y}^* = \mathbf{\Sigma}_k$, and transforming back yields:
$$\mathbf{X}^* = \mathbf{U} \mathbf{Y}^* \mathbf{V}^T = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T = \widehat{\mathbf{J}}_k$$
and the exact minimal Frobenius error is:
$$\|\mathbf{J} - \widehat{\mathbf{J}}_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2} \quad \blacksquare$$

---

## 3. Directional Derivative Anisotropy of Vertical Stripe Fields

Let the stripe noise be represented as a 2D continuous field:
$$S(x, y) = s(x) + \delta(x, y)$$
where $s(x)$ is a 1D function representing column variations, and $\delta(x, y)$ is a negligible vertical perturbation ($\|\delta\| \ll \|s\|$).

### 3.1 First-Order Differential Behavior
Taking partial derivatives along the two orthogonal image axes:
$$\frac{\partial S}{\partial x} = s'(x) \neq 0$$
$$\frac{\partial S}{\partial y} = \frac{\partial \delta(x, y)}{\partial y} \approx 0$$
In the degraded infrared image $D(x, y) = C(x, y) + S(x, y) + N(x, y)$:
$$\frac{\partial D}{\partial x} = \frac{\partial C}{\partial x} + s'(x) + \frac{\partial N}{\partial x}$$
$$\frac{\partial D}{\partial y} = \frac{\partial C}{\partial y} + \frac{\partial \delta}{\partial y} + \frac{\partial N}{\partial y} \approx \frac{\partial C}{\partial y} + \frac{\partial N}{\partial y}$$
Because $s'(x)$ has large high-frequency amplitude due to column-to-column jumps, the horizontal gradient $\frac{\partial D}{\partial x}$ is severely corrupted. In contrast, along the vertical direction, the stripe component disappears:
$$\nabla_y D \approx \nabla_y C + \nabla_y N$$

### 3.2 Second-Order Differential Behavior
Evaluating higher-order directional derivatives:
$$\frac{\partial^2 S}{\partial x^2} = s''(x) \gg 0$$
$$\frac{\partial^2 S}{\partial y^2} = \frac{\partial^2 \delta(x, y)}{\partial y^2} \approx 0$$
$$\frac{\partial^2 S}{\partial x \partial y} = \frac{\partial^2 \delta(x, y)}{\partial x \partial y} \approx 0$$
Hence, in the degraded observation:
$$\frac{\partial^2 D}{\partial y^2} \approx \frac{\partial^2 C}{\partial y^2} + \frac{\partial^2 N}{\partial y^2}$$
This demonstrates that the second vertical derivative $\frac{\partial^2 D}{\partial y^2}$ is **completely insulated from vertical stripe noise**.

### 3.3 Theoretical Elimination of Staircasing
Standard 1st-order Total Variation (TV) models penalize the L1-norm of the first gradient:
$$\min_C \|\nabla C\|_1$$
Because the subgradient of the L1 norm is constant in flat regions, 1st-order TV transforms smooth intensity gradients (such as sky backgrounds or thermal conduction ramps) into piecewise-constant planar patches—the notorious **staircase artifact**.

By introducing second-order directional total variation along the vertical axis:
$$\mathcal{R}_{\text{high-order}}(C) = \left\| \frac{\partial^2 C}{\partial y^2} \right\|_1$$
the optimization penalizes high curvature rather than slope, allowing linear and smooth thermal gradients to be preserved perfectly without staircasing, while exploiting the pristine vertical channel of the sensor.
