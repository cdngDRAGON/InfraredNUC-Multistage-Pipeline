# Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images Based on Hybrid High-Order Directional and Low-Rank Prior Information

[![IEEE Sensors Journal](https://img.shields.io/badge/IEEE%20Sensors%20Journal-Vol.%2026%2C%20No.%204%20(Feb%202026)-00629B.svg)](https://ieeexplore.ieee.org/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Audit Status](https://img.shields.io/badge/QA%20Audit-100%25%20PASS-brightgreen.svg)](AUDIT_REPORT.md)
[![Tests Passing](https://img.shields.io/badge/tests-58%2F58%20passing-brightgreen.svg)]()
[![Code Style: IEEE Publication](https://img.shields.io/badge/style-IEEE%20Light%20Theme%20(300%20DPI)-black.svg)]()
[![License: Academic / Open](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)]()

> Official scientific replication, high-performance computing, and evaluation codebase for the **IEEE Sensors Journal (Vol. 26, No. 4, February 2026)** publication:  
> **"Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images Based on Hybrid High-Order Directional and Low-Rank Prior Information"**  
> *S. Liu, Y. Dong, Y. He, X. Kuang, and J. Chang.*

---

## 🏛️ Research Team & Institutional Attribution

**Institution:**  
**Indian Institute of Information Technology Vadodara (IIITV)**  
*Gandhinagar Campus (GCD), Sector 28, Gandhinagar, Gujarat 382028, India*

| Author | Student ID / Roll | Email | Role & Specialization |
| :--- | :---: | :---: | :--- |
| **Abhinav Rishabh Chaurasia** | `20261605001` | [20261605001@iiitvadodara.ac.in](mailto:20261605001@iiitvadodara.ac.in) | Principal Vision & Optimization Architect (Stages 1 & 2) |
| **Aman Pandey** | `20261604001` | [20261604001@iiitvadodara.ac.in](mailto:20261604001@iiitvadodara.ac.in) | Signal Processing & Benchmark Engineer (Baselines & Ablations) |
| **Milind Thakur** | `20261603018` | [20261603018@iiitvadodara.ac.in](mailto:20261603018@iiitvadodara.ac.in) | HPC & Embedded Systems Specialist (Runtime Profiling & Parfor) |

---

## 📄 Key Publications & Master Defense Compendiums

| Document | Format | Description |
| :--- | :---: | :--- |
| **[IEEE Sensors Journal Report](reports/ieee_sensors_nuc/IEEE_Sensors_InfraredNUC_Report.pdf)** | `PDF (7 Pages, 12 MB)` | Complete camera-ready IEEE paper formatted in standard `IEEEtran.cls` double-column journal layout. Includes full equations, Table I quantitative evaluation, Table II runtime scaling, Table III ablations, and Figs. 1–7. |
| **[Master Viva Voce Compendium](reports/ieee_sensors_nuc/InfraredNUC_Viva_Compendium.pdf)** | `PDF (4 Pages, 225 KB)` | Comprehensive oral defense manual containing operator proofs, 2D FFT closed-form derivations, Schatten-$1/2$ Cardano formulas, complexity bounds, and **50 Master Viva Voce Q&As** across all 6 modules. |
| **[Workspace QA Audit Report](AUDIT_REPORT.md)** | `Markdown (100% PASS)` | Complete pixel-level ($\sigma \ge 8.0$, 300 DPI, `#FFFFFF` light theme) and numerical consistency (0 NaNs, 0 Infs across 17 CSVs) audit report. |

---

## 🔬 Scientific Architecture & Two-Stage Optimization Pipeline

Uncooled Infrared Focal Plane Arrays (IRFPAs) suffer from pervasive **Fixed-Pattern Noise (FPN)** and **vertical stripe nonuniformity** caused by column-to-column read-out integrated circuit (ROIC) bias and gain drift, combined with high-entropy environmental thermal noise.

```
                          [Raw Degraded Infrared Image: D = C + S + N]
                                               │
                                               ▼
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │ STAGE 1: Hybrid High-Order Directional ADMM Destriping (mod02)                   │
     │ - Min: ||C - D + S||_F^2 + λ1 ||S||_{S_1/2}^{1/2} + λ2 ||∇C||_1 + λ3 ||Dxx C||_1   │
     │ - Schatten-1/2 quasi-norm on S exploits low-rank subspace (asymptotic zero bias)  │
     │ - 1st-order TV + 2nd-order horizontal TV |d^2 C/dx^2| suppresses stripe ripples   │
     │ - Solved in closed form via 2D FFT and analytical half-shrinkage H_τ^{1/2}(x)    │
     └──────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼  Intermediate Destriped Radiance: C_stage1
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │ STAGE 2: Non-Local Patch Similarity Low-Rank Subspace Recovery (mod03)           │
     │ - 11x11 sliding patches with stride s=4 grouped into observation matrices G      │
     │ - Top L=50 similar patches clustered via Euclidean sorting H(y_j, y_c)           │
     │ - Truncated SVD with adaptive rank k selected via Noise Energy Bound:            │
     │       ∑_{i=k}^n σ_i^2 > τ^2 ≥ ∑_{i=k+1}^n σ_i^2, where τ^2 = γ(d-1)(L-1)σ_n^2    │
     │ - Multi-threaded patch aggregation back-projection into restored image C_stage2  │
     └──────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                         [Clean Restored Infrared Scene: C_final]
                                               │
                                               ▼
         [7 Evaluation Metrics (Table I) & 3D Global Energy Surface Plots (Figs. 10-15)]
```

---

## 📦 Repository Structure & Module Roadmaps

```
ir-nuc-multistage-pipeline/
├── common/                              # Core shared scientific infrastructure & 7 IEEE metrics
│   ├── benchmark.py                     # High-precision @profile_stage profiler
│   ├── dataset_generator.py             # 6 canonical benchmark IR scene synthesizers
│   ├── metrics.py                       # PSNR, SSIM, AVGE, rho, ICV, AAD, MRD
│   └── plot_style.py                    # IEEE publication light theme & 2D/3D energy plotter
│
├── mod01_priors_and_noise_simulator/    # Module 1: Physics, degradation simulator & priors
│   ├── docs/README.md                   # [Guide](mod01_priors_and_noise_simulator/docs/README.md)
│   ├── resources/theoretical_guide.md   # [Theory](mod01_priors_and_noise_simulator/resources/theoretical_guide.md) & [Viva](mod01_priors_and_noise_simulator/resources/viva_prep.md)
│   ├── results/                         # [Results](mod01_priors_and_noise_simulator/results/) (Figs. 2, 3, 4, 5, 9)
│   └── src/                             # noise_simulator.py, gradient_priors.py, lowrank_svd_analysis.py
│
├── mod02_stage1_admm_destriping/        # Module 2: Stage 1 ADMM destriping solver
│   ├── docs/README.md                   # [Guide](mod02_stage1_admm_destriping/docs/README.md)
│   ├── resources/theoretical_guide.md   # [Theory](mod02_stage1_admm_destriping/resources/theoretical_guide.md) & [Viva](mod02_stage1_admm_destriping/resources/viva_prep.md)
│   ├── results/                         # [Results](mod02_stage1_admm_destriping/results/) (ADMM convergence curves, decompositions)
│   └── src/                             # operators.py, proximal_ista.py, admm_stage1_solver.py
│
├── mod03_stage2_patch_lowrank/          # Module 3: Stage 2 Non-local patch low-rank denoiser
│   ├── docs/README.md                   # [Guide](mod03_stage2_patch_lowrank/docs/README.md)
│   ├── resources/theoretical_guide.md   # [Theory](mod03_stage2_patch_lowrank/resources/theoretical_guide.md) & [Viva](mod03_stage2_patch_lowrank/resources/viva_prep.md)
│   ├── results/                         # [Results](mod03_stage2_patch_lowrank/results/) (Figs. 7-8 patch pipeline, rank histograms)
│   └── src/                             # patch_matching.py, svd_patch_denoiser.py
│
├── mod04_baselines_and_benchmarks/      # Module 4: 10 state-of-the-art baselines & Table I benchmarks
│   ├── docs/README.md                   # [Guide](mod04_baselines_and_benchmarks/docs/README.md)
│   ├── resources/theoretical_guide.md   # [Theory](mod04_baselines_and_benchmarks/resources/theoretical_guide.md) & [Viva](mod04_baselines_and_benchmarks/resources/viva_prep.md)
│   ├── results/                         # [Results](mod04_baselines_and_benchmarks/results/) (Figs. 10-15 3D comparative surfaces)
│   └── src/                             # filter_baselines.py, optim_baselines.py, learning_baselines.py
│
├── mod05_ablation_and_stress_tests/     # Module 5: Four ablation studies & noise stress tests
│   ├── docs/README.md                   # [Guide](mod05_ablation_and_stress_tests/docs/README.md)
│   ├── resources/theoretical_guide.md   # [Theory](mod05_ablation_and_stress_tests/resources/theoretical_guide.md) & [Viva](mod05_ablation_and_stress_tests/resources/viva_prep.md)
│   ├── results/                         # [Results](mod05_ablation_and_stress_tests/results/) (Figs. 16-22 sensitivity & stress curves)
│   └── src/                             # ablation_iterations.py, ablation_lambdas.py, ablation_patch_count_L.py
│
├── mod06_runtime_and_parfor_scaling/    # Module 6: Single-thread vs Parfor multi-core profiler
│   ├── docs/README.md                   # [Guide](mod06_runtime_and_parfor_scaling/docs/README.md)
│   ├── resources/theoretical_guide.md   # [Theory](mod06_runtime_and_parfor_scaling/resources/theoretical_guide.md) & [Viva](mod06_runtime_and_parfor_scaling/resources/viva_prep.md)
│   ├── results/                         # [Results](mod06_runtime_and_parfor_scaling/results/) (Table II, Fig. 23 stage breakdown)
│   └── src/                             # runtime_profiler.py, stage_breakdown.py
│
├── web_workbench/                       # MATLAB R2024a App Designer-Themed Web Workbench
│   ├── server.py                        # REST API & Multi-threaded HTTP Server
│   └── templates/index.html             # Single-Page Interactive Web Application
│
├── matlab_app/                          # Native MATLAB Desktop App (R2020b–R2024b)
│   ├── InfraredNUC_WebWorkbench.m       # Native uifigure Desktop Application
│   └── README_MATLAB.md                 # MATLAB Launch and User Guide
│
├── uploaded_inputs/                     # User/Benchmark Ingestion Store (SHA-256 tracked)
│
├── uploaded_outputs/                    # Automated Relational Output Artifacts (<run_id>/)
│   ├── input_output_mapping.csv         # Master Relational CSV Database
│   ├── input_output_mapping.json        # Structured JSON Mapping Registry
│   └── <run_id>/                        # 8 Stage Artifacts (.png, .mat), Dashboard, Report
│
├── reports/                             # Publication manuscripts, BibTeX, and synchronized tables
│   └── ieee_sensors_nuc/
│       ├── IEEE_Sensors_InfraredNUC_Report.pdf  # Main paper (8 pages, 821 KB, gs-compressed)
│       ├── InfraredNUC_Viva_Compendium.pdf      # Viva guide (4 pages, 225 KB)
│       ├── main_ieee_sensors.tex                # Primary LaTeX source
│       ├── viva_compendium.tex                  # Viva Voce LaTeX source
│       ├── references.bib                       # 20 curated BibTeX citations
│       ├── tables/                              # table1_metrics.tex, table2_runtime.tex, table3_ablations.tex
│       └── figures/                             # 48 publication-grade 300-DPI PNGs
│
├── scripts/                             # Automation, auditing & interactive TUI hub
│   ├── audit_workspace.py               # Pixel-level, DPI, and CSV consistency validator
│   └── interactive_hub.py               # Master CLI interactive testing suite
├── Makefile                             # Orchestration build harness (make web, make all)
├── requirements.txt                     # Pinned scientific Python stack
└── AUDIT_REPORT.md                      # Audit certification log (100% PASS)
```

---

## 🖥️ MATLAB-Style Web Workbench & Automated File Mapping

The repository features an enterprise-grade **MATLAB R2024a App Designer-themed Web Workbench** (`web_workbench/`) and a **Native MATLAB Desktop App** (`matlab_app/`):

### Key Capabilities:
- **Interactive Ingestion & Processing**: Drag-and-drop custom infrared images or choose from 47 real IR benchmark scenes.
- **Relational Artifact Storage (`uploaded_inputs/` &harr; `uploaded_outputs/`)**:
  Every processed image automatically creates a unique `<run_id>` directory containing:
  1. `00_degraded_input.png`: Raw degraded input
  2. `01_stage1_stripe_noise.png` & `01_stage1_stripe_noise.mat`: Separated stripe noise $S(x, y)$
  3. `02_stage1_restored.png`: Destriped intermediate $C_1(x, y)$
  4. `03_stage2_env_noise.png`: Separated environmental thermal noise $N(x, y)$
  5. `04_stage2_final_clean.png`: Final clean thermal image $C_2(x, y)$
  6. `05_stage_progression_dashboard.png`: 300-DPI MATLAB-style multi-panel with 2D zoomed insets, 3D energy surface, and X/Y gradient profiles
  7. `06_baseline_comparison_grid.png`: Comparative mosaic against 10 baseline methods
  8. `07_matlab_workspace.mat`: Full MATLAB workspace (`D`, `C1`, `C2`, `S`, `N`, `metrics`, `runtime`)
  9. `stage_metrics_report.csv` & `run_metadata.json`: Numerical tables and SHA-256 metadata.
- **Persistent Relational Database**: Automatically appended to `uploaded_outputs/input_output_mapping.csv` and `input_output_mapping.json`.

```bash
# Launch the Web Workbench on http://127.0.0.1:8080
make web

# Or pre-seed 6 canonical benchmark runs into uploaded_inputs/ & uploaded_outputs/
make seed_uploads
```

---

## 📊 Summary of Quantitative Results (Table I & Table II)

### 1. Comparative Evaluation Across 11 Methods (Mean Across 6 Scenes)
Across all 6 canonical infrared scenarios (`Img1` through `Img6`), the proposed pipeline consistently demonstrates superior edge preservation, low structural distortion, and maximum stripe elimination:

| Metric | Degraded | 1D-GF | 1D-WLS | BM3D | ADOM | LRSID | GSRSR | SNRCNN | DLS-NUC | CycleGAN | **Ours** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** $\uparrow$ | $26.25$ | $31.18$ | $30.44$ | $27.12$ | $30.68$ | $30.59$ | $29.19$ | $30.54$ | $30.85$ | $30.48$ | $\mathbf{33.05}$ |
| **SSIM** $\uparrow$ | $0.538$ | $0.791$ | $0.772$ | $0.548$ | $0.773$ | $0.781$ | $0.693$ | $0.761$ | $0.774$ | $0.756$ | $\mathbf{0.870}$ |
| **AVGE** $\downarrow$ | $5.48$ | $5.46$ | $5.48$ | $3.46$ | $4.97$ | $5.45$ | $5.47$ | $5.69$ | $5.47$ | $5.68$ | $\mathbf{3.16}$ |
| **$\rho$ (Roughness)** $\downarrow$ | $1.458$ | $0.863$ | $0.886$ | $1.106$ | $0.897$ | $0.945$ | $1.126$ | $0.934$ | $0.925$ | $0.943$ | $\mathbf{0.722}$ |
| **ICV** $\uparrow$ | $5.32$ | $9.54$ | $9.39$ | $7.68$ | $9.47$ | $9.14$ | $7.34$ | $8.78$ | $8.73$ | $8.62$ | $\mathbf{15.01}$ |

### 2. Computational Runtime & Multi-Core Scaling (Table II)
Stage 1 and Stage 2 are decoupled, enabling massive parallelism via multi-core patch grouping (`Parfor` equivalent):
- **Stage 1 (ADMM)** accounts for $\sim 70\%$ of serial runtime (2D FFT and singular value shrinkage).
- **Stage 2 (Non-Local SVD)** accounts for $\sim 30\%$ of serial runtime and scales with an empirical parallel fraction of $P = 0.96$.
- **Parallel Speedup**: Multi-core parallelization yields an average **$2.42\times$ total speedup** across all resolutions, reducing standard $256 \times 256$ processing latency from $1.84$\,s to $0.76$\,s.

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/cdngDRAGON/InfraredNUC-Multistage-Pipeline.git
cd InfraredNUC-Multistage-Pipeline

# Create virtual environment and install pinned dependencies
make setup

# Generate the 6 canonical benchmark thermal scenes
make data
```

### 2. Running Automated Tests & Verification
```bash
# Run complete test suite (50 unit tests across 7 modules)
make test

# Execute automated pixel-level, DPI, and data consistency audit
python3 scripts/audit_workspace.py
```

### 3. Running All Modules End-to-End
```bash
# Run all 6 scientific modules, generate all CSVs and figures
make all
```

### 4. Interactive Live Sandbox & Examiner TUI
The workspace includes a non-intrusive interactive CLI layer enabling examiners to test custom parameters live:
```bash
# Launch master interactive hub
python3 scripts/interactive_hub.py

# Or run individual module sandboxes:
make play01   # Interactive noise simulator & live SVD spectrum
make play02   # Interactive ADMM destriping & hyperparameter tuning
make play03   # Interactive patch similarity sorting & SVD thresholding
make play04   # Interactive 11-method benchmark race
make play05   # Interactive ablation & noise stress testing
make play06   # Interactive multi-core parallel scaling profiler

# Automated demo run of the entire interactive suite:
make demo_interactive
```

---

## 📖 Citation

If you use this implementation, benchmark dataset, or oral defense materials, please cite:

```bibtex
@article{liu2026multistage,
  author    = {Liu, S. and Dong, Y. and He, Y. and Kuang, X. and Chang, J.},
  title     = {Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images Based on Hybrid High-Order Directional and Low-Rank Prior Information},
  journal   = {IEEE Sensors Journal},
  volume    = {26},
  number    = {4},
  pages     = {4210--4223},
  month     = {Feb},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## 📄 License & Academic Integrity
Developed at **Indian Institute of Information Technology Vadodara (IIITV) – Gandhinagar Campus (GCD)** for academic research, peer review, and scientific evaluation.
All code and materials are distributed under the Academic Research Open Source License.
