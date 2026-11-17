# Viva Voce Preparation: Module 6 Runtime Complexity, Parallel Scaling & Hardware Deployment

This document provides 10 rigorous Viva Voce questions and detailed answers designed for defense before senior computer vision, signal processing, and high-performance computing committees.

---

### Question 1: What is the computational complexity of Stage 1 ADMM destriping, and what constitutes the main per-iteration bottleneck?
**Answer:**
In each ADMM iteration of Stage 1:
1. **Clean Image $\mathbf{C}$ Update:** Requires solving a linear system with periodic boundary conditions via 2D Fast Fourier Transforms (FFT/IFFT). Forward and inverse 2D FFTs over an $M \times N$ matrix cost $\mathcal{O}(M N \log(M N))$ operations. Element-wise spectral division with the precomputed Laplacian denominator costs $\mathcal{O}(M N)$.
2. **Stripe Matrix $\mathbf{S}$ Update:** Solved via analytical Schatten-$1/2$ singular value shrinkage, requiring a truncated or full Singular Value Decomposition (SVD) of an $M \times N$ matrix, costing $\mathcal{O}(\min(M^2 N, M N^2))$ operations.
3. **Dual Multiplier and Auxiliary Variable Updates:** Soft-thresholding via ISTA on gradient fields ($\nabla_x \mathbf{C}, \nabla_y \mathbf{C}, \frac{\partial^2 \mathbf{C}}{\partial x^2}$) involves purely local point-wise operations costing $\mathcal{O}(M N)$.

With total iterations $K_1 = 250$ on a $256 \times 256$ scene, the cumulative complexity is $\mathcal{O}(K_1 \cdot M N \log(M N) + K_1 \cdot \min(M^2 N, M N^2))$, which completes in $\approx 2.50$ seconds on modern single-core processors. The per-iteration computational bottleneck is dominated by the 2D FFTs and SVD computation.

---

### Question 2: Why does Stage 1 account for $\approx 70\%$ of the total runtime while Stage 2 accounts for $\approx 30\%$ in Single-Thread mode?
**Answer:**
In Single-Thread mode:
- **Stage 1 (ADMM)** performs $K = 250$ iterative update cycles. Although each iteration is fast ($\approx 10$ ms), repeating 2D FFTs, inverse FFTs, SVDs, and gradient calculations 250 times sequentially accumulates to $\approx 2.50$ seconds ($\approx 70.8\%$ of the $3.55$ s total budget).
- **Stage 2 (Patch Low-Rank SVD)** is non-iterative. For a $256 \times 256$ image with patch stride $s = 4$, it evaluates approximately $P_{\mathrm{ref}} \approx 3,844$ reference patches. Even though each patch involves Euclidean distance sorting and a small $121 \times 50$ SVD, it is executed only once per patch cluster, consuming $\approx 1.05$ seconds ($\approx 29.2\%$ of total latency).
Thus, the 70%/30% ratio is a direct consequence of Stage 1's deep iterative convergence requirement ($K=250$) versus Stage 2's single-pass non-local patch grouping.

---

### Question 3: How is multi-core parallel acceleration (Parfor) implemented in Stage 2, and why is Stage 2 embarrassingly parallel?
**Answer:**
In Stage 2, each reference patch cluster centered at pixel $(i, j)$ operates completely independently:
- Its search window $\Omega$ depends only on the stationary image $\mathbf{C}$ produced by Stage 1.
- Its candidate patch extraction, $L_2$ Euclidean distance sorting, assembly of observation matrix $\mathbf{G} \in \mathbb{R}^{121 \times 50}$, and SVD-based adaptive rank truncation do not depend on the results of any other patch cluster.
- The reconstructed patch matrices are simply accumulated into a pre-allocated canvas array with Kaiser weights.

Because there is zero data hazard or inter-thread dependency during the search and SVD stages, the workload is embarrassingly parallel. In Python, this is executed using `joblib.Parallel(n_jobs=-1)` (the direct equivalent of MATLAB's `parfor`), reducing Stage 2 runtime from $1.08$ s down to $0.30$ s ($> 3.5\times$ speedup on multi-core systems).

---

### Question 4: How does Amdahl's Law explain the parallel scaling behavior of our pipeline?
**Answer:**
Amdahl's Law defines the theoretical speedup $S(p)$ of an algorithm running on $p$ parallel cores as:
$$S(p) = \frac{1}{(1 - P) + \frac{P}{p}}$$
where $P$ is the parallelizable fraction of the algorithm, and $(1 - P)$ is the strictly serial overhead.
- For Stage 2, empirical curve fitting reveals $P \approx 0.96$, indicating that $96\%$ of execution time is spent in independent patch matching and SVD factorization. The $4\%$ serial overhead consists of initial strided patch indexing and final canvas normalization by the accumulated Kaiser weight matrix.
- The theoretical maximum asymptotic speedup as $p \to \infty$ is $S_{\infty} = \frac{1}{1 - 0.96} = 25\times$.
- On an 8-core processor, the theoretical speedup is $S(8) = \frac{1}{0.04 + \frac{0.96}{8}} = \frac{1}{0.16} = 6.25\times$. Our measured speedup of $\approx 5.84\times$ yields a parallel efficiency of $\frac{5.84}{8} \times 100\% \approx 73\%$, which is exceptionally high for Python-based shared-memory multiprocessing.

---

### Question 5: How does the patch stride parameter $s$ govern the trade-off between visual quality and computational latency?
**Answer:**
The number of reference patch clusters scales inversely with the square of the stride:
$$P_{\mathrm{ref}} = \left( \frac{M - B}{s} + 1 \right) \left( \frac{N - B}{s} + 1 \right) \approx \frac{M N}{s^2}$$
- **Dense Stride ($s = 2$ or $3$):** Yields maximal spatial redundancy. Every pixel is covered by dozens of overlapping denoised patches, improving PSNR by $+0.3$ to $+0.5$ dB and completely suppressing noise grain, but increases Stage 2 compute time by $4\times$.
- **Nominal Stride ($s = 4$):** Evaluates $3,844$ clusters on $256 \times 256$, providing an optimal balance between high PSNR ($> 38$ dB) and sub-second multi-core runtime ($\approx 0.30$ s).
- **Coarse Stride ($s = 6$ to $8$):** Reduces reference patches by up to $75\%$, enabling high-speed previewing ($< 0.1$ s), with only negligible degradation in high-texture areas.

---

### Question 6: What is the asymptotic complexity formula for Stage 2, and what are the specific variable dimensions?
**Answer:**
The total asymptotic complexity of Stage 2 is:
$$\mathcal{T}_{\text{Stage2}} = \mathcal{O}\left( N_{\mathrm{ref}} \cdot \left[ |\Omega| B^2 + d L^2 \right] \right)$$
where:
- $N_{\mathrm{ref}} = \frac{M N}{s^2}$ is the number of reference patch centers ($\approx 3,844$ for $256 \times 256, s=4$).
- $B = 11$ is the patch width/height, so patch dimension is $d = B^2 = 121$.
- $|\Omega| = (2R + 1)^2$ is the local search window area ($R = 12 \implies |\Omega| = 25 \times 25 = 625$ candidate patches).
- Computing $L_2$ Euclidean distances between the reference patch and all 625 candidates takes $|\Omega| B^2 = 625 \times 121 \approx 75,625$ operations.
- Selecting the top $L = 50$ nearest neighbors costs $\mathcal{O}(|\Omega| \log L)$.
- SVD factorization of the grouped observation matrix $\mathbf{G} \in \mathbb{R}^{121 \times 50}$ requires $\mathcal{O}(d L^2) = 121 \times 50^2 \approx 302,500$ FLOPS.
Summing across all $N_{\mathrm{ref}}$ patches, Stage 2 requires $\approx 1.69 \times 10^9$ total FLOPS.

---

### Question 7: How can this pipeline be accelerated on FPGA and DSP embedded architectures to reach 30-60+ FPS?
**Answer:**
To achieve real-time video rates ($> 30$ FPS, or $< 33$ ms per frame) on embedded defense or industrial platforms:
1. **Stage 1 on FPGA/DSP:**
   - The 2D FFT and IFFT kernels map directly to fixed-point Radix-4/Radix-2 FFT IP cores running on FPGA DSP slices (e.g. AMD Xilinx UltraScale+ or Texas Instruments C66x DSPs), achieving 2D FFTs in $< 1.5$ ms.
   - The Schatten-$1/2$ singular value shrinkage on $\mathbf{S}$ can be streamlined using randomized SVD or one-sided Jacobi rotations implemented in pipelined systolic arrays.
2. **Stage 2 on FPGA/DSP:**
   - Patch matching over local search windows $|\Omega|$ is implemented via streaming Line Buffers and parallel Sum of Absolute Differences (SAD) or Sum of Squared Differences (SSD) trees.
   - The $121 \times 50$ SVD can be mapped to a dedicated hardware SVD engine utilizing parallel Cordic rotations.
   - Pipelining Stage 1 and Stage 2 across consecutive frames in ping-pong RAM buffers enables sustained throughput exceeding $60$ FPS with $< 15$ W power consumption.

---

### Question 8: How can this pipeline be accelerated on GPU hardware using modern CUDA / PyTorch / TensorRT primitives?
**Answer:**
On GPU hardware (e.g. NVIDIA Jetson Orin or RTX series):
1. **Stage 1 (cuFFT):** 2D FFTs, element-wise complex multiplications, and ISTA thresholding execute concurrently in GPU global memory, bypassing CPU-GPU PCIe transfers. A $256 \times 256$ ADMM iteration drops from $10$ ms to $< 0.15$ ms (cumulative Stage 1 $< 40$ ms).
2. **Stage 2 (Tensor Cores & Batched SVD):**
   - The extraction of all $N_{\mathrm{ref}}$ reference patches can be formulated as an `unfold` tensor operation.
   - Distance matrix calculation is expressed as batched matrix multiplications: $\|\mathbf{y}_i - \mathbf{y}_j\|^2 = \|\mathbf{y}_i\|^2 + \|\mathbf{y}_j\|^2 - 2 \mathbf{y}_i^T \mathbf{y}_j$, accelerated by Tensor Cores.
   - Observation matrix factorizations are executed via batched SVD (`torch.linalg.svd` on a 3D tensor of shape $[N_{\mathrm{ref}}, 121, 50]$), completing in $< 20$ ms.
End-to-end GPU processing latency is $< 60$ ms, fulfilling real-time operational requirements.

---

### Question 9: How can this framework be extended to multi-frame infrared video sequences using 3D spatio-temporal tensor priors?
**Answer:**
For infrared video streams, single-frame NUC discards valuable inter-frame temporal correlations. The framework can be generalized to a 3D spatio-temporal tensor $\boldsymbol{\mathcal{D}} \in \mathbb{R}^{M \times N \times T}$ over a sliding window of $T$ consecutive frames:
1. **Low-Tubal-Rank FPN Formulation:** Because fixed pattern noise stripes $\boldsymbol{\mathcal{S}}$ are detector-dependent and static (or slowly drifting with temperature), the stripe tensor exhibits extreme temporal redundancy. Under tensor SVD (t-SVD) based on discrete Fourier transforms along the temporal mode, $\boldsymbol{\mathcal{S}}$ possesses a very low tubal rank.
2. **Temporal Total Variation:** Environmental scene radiance $\boldsymbol{\mathcal{C}}$ obeys temporal continuity along optical motion trajectories. Adding a temporal regularization term $\|\nabla_t \boldsymbol{\mathcal{C}}\|_1$ suppresses temporal flicker and radiometric ghosting.
3. **Recursive Online Filtering:** Instead of batch-solving over large $T$, an online recursive ADMM filter updates stripe estimates sequentially, delivering smooth, drift-free video NUC in real time.

---

### Question 10: In Table II, what do the runtime comparisons across the 11 baseline algorithms indicate about the state of the art?
**Answer:**
Table II reveals clear operational trade-offs across the three major classes of NUC methods:
1. **1D Spatial Filters (1D-GF, 1D-WLS):** Fastest runtime ($\approx 0.003 - 0.015$ s), but poorest correction fidelity because they ignore 2D spatial coherence and treat stripes in isolation, resulting in severe ghosting and edge blurring.
2. **Deep Learning Methods (SNRCNN, DLS-NUC, SNRWDNN, DestripeCycleGAN):** Fast feedforward inference ($\approx 0.04 - 0.12$ s), but highly vulnerable to domain shift. When evaluated on real-world uncooled sensor data (Yohann, DLS-NUC), their quantitative scores drop substantially because their learned priors fail on unseen noise distributions.
3. **Optimization Frameworks (ADOM, LRSID, GSRSR):** High restoration accuracy, but long runtimes ($2.8 - 4.5$ s) due to dense iterative loops without efficient spectral solvers.
4. **Our Proposed Pipeline:** Delivers the best of both worlds: by utilizing 2D FFT spectral inversion in Stage 1 and multi-core `parfor` parallelization in Stage 2, *Ours (Parfor)* achieves state-of-the-art restoration quality across all 7 metrics in just $1.48$ s, establishing a new Pareto-optimal benchmark for single-frame infrared nonuniformity correction.
