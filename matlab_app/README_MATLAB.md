# MATLAB Desktop App: Infrared NUC Multi-Stage Pipeline Workbench

**Reference**: Liu et al., *"Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images Based on Hybrid High-Order Directional and Low-Rank Prior Information"*, **IEEE Sensors Journal**, Vol. 26, No. 4, Feb. 2026.

**Authors / Institution**:
- **Abhinav Rishabh Chaurasia** (UID: 20261605001)
- **Aman Pandey** (UID: 20261604001)
- **Milind Thakur** (UID: 20261603018)
- *Indian Institute of Information Technology Vadodara (IIITV) – Gandhinagar Campus (GCD)*

---

## 1. Overview
The MATLAB desktop app (`matlab_app/InfraredNUC_WebWorkbench.m`) provides a native MATLAB R2020b–R2024b App Designer interface (`uifigure`) replicating the Web Workbench directly inside MATLAB.

Key Features:
- Direct execution of the authors' official algorithms from `data/external_repos/InfraredNUC/matlabCode/`:
  - `Functions/admmAlgorithm.m` (Stage 1 ADMM destriping)
  - `Functions/lraSvdDenoising.m` (Stage 2 LRA-SVD patch denoising)
- Interactive dropdown selection across all 47 real infrared scenes.
- 5 side-by-side axes:
  1. `(a) Degraded Input D(x, y)`
  2. `(b) Extracted Stripe Noise S(x, y)` (seismic colormap)
  3. `(c) Stage 1 Destriped C1(x, y)`
  4. `(d) Residual Noise N(x, y)` (coolwarm colormap)
  5. `(e) Final Clean C2(x, y)` (Ours)
- Real-time 3D Global Energy surface map and horizontal scanline cross-section.
- Automatic registration and artifact persistence into `../uploaded_outputs/<run_id>/` and `../uploaded_outputs/input_output_mapping.csv`.

---

## 2. Prerequisites & Launch Instructions

### Prerequisites
- MATLAB R2020b or later (Image Processing Toolbox recommended).

### How to Run
1. Open MATLAB and navigate to the project directory:
   ```matlab
   cd /path/to/AiProject/matlab_app
   ```
2. Launch the workbench app:
   ```matlab
   InfraredNUC_WebWorkbench
   ```
3. The interactive desktop UI window will appear. Select any scene, adjust the $\lambda$ or $\sigma_n$ sliders, and click **`▶ Run Multi-Stage NUC Pipeline`**.
4. The processed clean image, extracted stripes, 3D energy surface, and execution latency will be displayed immediately. Output `.mat` workspaces and PNGs will be saved to `../uploaded_outputs/`.
