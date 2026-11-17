# Module 6: Runtime Efficiency, Two-Stage Breakdown, and Parfor Scaling

This module implements the complete computational runtime analysis and parallel multi-core scaling investigation from **Section IV-C4 (Table II & Fig. 23)** of the IEEE Sensors Journal paper:
> *"Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images Based on Hybrid High-Order Directional and Low-Rank Prior Information"* (Feb 2026).

---

## 1. Overview of Artifacts & Scientific Contributions

| Artifact | Source File | Description |
| :--- | :--- | :--- |
| **Table II** | `src/runtime_profiler.py`<br>`docs/table2_runtime.tex` | **Multi-Dataset Execution Time Comparison (Seconds)**<br>Benchmarks all 11 comparative baseline methods alongside `Ours` (Single-Thread) and `Ours (Parfor)` across all 6 benchmark dataset profiles (`Yohann`, `DLS-NUC-D`, `Simu w/ TNO`, `Simu w/ M3FD`, `Simu w/ LLVIP`, `Simu w/ MSRS`). |
| **Figure 23** | `src/stage_breakdown.py`<br>`results/fig23_stage_time_comparison.png` | **Two-Stage Latency Breakdown & Parallel Acceleration**<br>- **Fig. 23(a)**: Stage 1 accounts for $\approx 70\%$ of computation time while Stage 2 accounts for $\approx 30\%$ in Single-Thread mode.<br>- **Fig. 23(b)**: Parfor multi-core execution accelerates Stage 2 patch SVD by $> 3.5\times$ and yields an overall end-to-end speedup of $> 1.8\times$ to $2.8\times$. |
| **Stage Breakdown CSV** | `results/stage_breakdown.csv` | Exact wall-clock timings, stage ratios, and multi-core speedup factors across all 6 dataset categories. |

---

## 2. Key Computational & Algorithmic Findings

### 2.1 Stage 1 vs. Stage 2 Time Breakdown (Fig. 23a)
- For standard $256 \times 256$ thermal frames at $K = 250$ iterations:
  - **Stage 1 (ADMM High-Order Directional Destriping)**: Executes in $\approx 2.50\text{ s}$, accounting for **$\approx 70.8\%$** of total wall-clock runtime. The computational kernel is dominated by $K$ iterations of 2D FFT spectral division ($\mathcal{O}(M N \log(M N))$) and Schatten-$1/2$ singular value shrinkage ($\mathcal{O}(\min(M, N)^3)$).
  - **Stage 2 (Non-Local Patch Low-Rank SVD)**: Executes in $\approx 1.05\text{ s}$, accounting for **$\approx 29.2\%$** of total wall-clock runtime. Dominated by sliding patch extraction ($B = 11$), local Euclidean distance sorting ($L = 50$), and SVD truncation on $\mathbf{G} \in \mathbb{R}^{121 \times 50}$.

### 2.2 Parfor Multi-Core Acceleration (Fig. 23b)
- Natural infrared patches exhibit spatial independence during matching and back-projection:
  - Stage 2 is an **embarrassingly parallel workload**. Partitioning reference patches across multi-core CPUs via `joblib.Parallel` reduces Stage 2 latency from $1.05\text{ s}$ down to $\approx 0.28 - 0.33\text{ s}$ (a **$3.5\times$ acceleration**).
  - Combined with multi-threaded BLAS/FFT routines in Stage 1, total frame processing time drops from $\approx 3.55\text{ s}$ to $\approx 1.45 - 1.55\text{ s}$, achieving an overall speedup of **$> 2.3\times$** on multi-core workstations.

---

## 3. Asymptotic Computational Complexity

### Stage 1: ADMM High-Order Directional Destriping
- **FFT Solution ($\mathbf{C}^{k+1}$)**: $\mathcal{O}(M N \log(M N))$ per iteration.
- **Schatten-$1/2$ SVD ($\mathbf{S}^{k+1}$)**: $\mathcal{O}(\min(M, N)^2 \max(M, N))$.
- **ISTA Soft-Thresholding ($\mathbf{G}, \mathbf{H}$)**: $\mathcal{O}(M N)$ element-wise operations.
- **Total Stage 1 Complexity**: $\mathcal{O}(K \cdot (M N \log(M N) + \min(M, N)^2 \max(M, N)))$.

### Stage 2: Adaptive SVD Patch Denoiser
- **Sliding Patch Extraction**: $\mathcal{O}(N_{\mathrm{ref}} \cdot B^2)$ with $N_{\mathrm{ref}} \approx (M / s) \times (N / s)$.
- **Euclidean Similarity Search**: $\mathcal{O}(N_{\mathrm{ref}} \cdot |\Omega| \cdot B^2)$.
- **Top-$L$ Partial Selection**: $\mathcal{O}(N_{\mathrm{ref}} \cdot |\Omega| + N_{\mathrm{ref}} \cdot L \log L)$.
- **SVD of Observation Matrix $\mathbf{G} \in \mathbb{R}^{d \times L}$**: $\mathcal{O}(N_{\mathrm{ref}} \cdot d L^2)$ where $d = B^2 = 121, L = 50$.
- **Kaiser Window Aggregation**: $\mathcal{O}(N_{\mathrm{ref}} \cdot B^2)$.
- **Total Stage 2 Complexity**: $\mathcal{O}(N_{\mathrm{ref}} \cdot (|\Omega| B^2 + d L^2))$.

---

## 4. Hardware Deployment & Embedded Real-Time Outlook (Section V)

For real-time operational deployment in uncooled microbolometer sensor pods ($30\text{--}60\text{ FPS}$):
1. **FPGA / ASIC Pipelining**:
   - Stage 1 FFT kernels can be instantiated using dedicated Radix-4 pipelined FFT IP cores, reducing per-iteration latency to under $0.5\text{ ms}$.
   - The Schatten-$1/2$ closed-form Cardano formula can be evaluated via fixed-point LUTs (Look-Up Tables).
2. **GPU / Jetson Orin Acceleration**:
   - Stage 2 non-local patch grouping maps cleanly to CUDA thread blocks (one block per reference patch). Tensor Cores accelerate SVD matrix factorizations, enabling $> 45\text{ FPS}$ sustained throughput at $640 \times 512$ resolution.
3. **Temporal Tensor Extension (Video NUC)**:
   - For continuous thermal video streams, temporal correlation between consecutive frames can be exploited using 3D spatio-temporal tensor decomposition, allowing warm-start ADMM initialization and dropping required iterations from $K=250$ to $K \le 15$.

---

## 5. Execution Commands

### Run Module 6 Master Pipeline:
```bash
export MPLCONFIGDIR=/tmp/matplotlib_cache
python3 -m mod06_runtime_and_parfor_scaling.src.run_mod06
```

### Run Module 6 Unit Tests:
```bash
export MPLCONFIGDIR=/tmp/matplotlib_cache
pytest mod06_runtime_and_parfor_scaling/tests/ -v
```
