# Reverse-Engineering Audit & Multi-Repository Mathematical Reconciliation

**Audit Date**: February 2026 / November 2026  
**Auditor**: Principal Reverse-Engineering & Computer Vision Systems Architect  
**Subject**: Verification of *Liu et al.* (IEEE Sensors Journal, Vol. 26, No. 4, Feb. 2026) against public author repositories:
- `ImageVisioner/InfraredNUC` (Primary algorithm code)
- `ImageVisioner/IRdenoise` (FOM denoising and raw sensor routines)
- `ImageVisioner/Image_Fusion_Evaluation_modify` (Metric calculation suite)
- `hezw2016/1D-GF` (Cao et al. 1D Guided Filter baseline & real IR scenes)
- `hezw2016/DLS-NUC` (He et al. deep infrared destriping baseline & benchmark datasets)
- `Kuangxd/ONR` (Kuang et al. Deep Infrared NUC datasets)
- `Linfeng-Tang/PIAFusion`, `Linfeng-Tang/SwinFusion`, `JinyuanLiu-CV/TarDAL` (TNO, MSRS, RoadScene real thermal datasets)
- `ipol_mire` (Yohann Tendero 2012 landmark uncooled FPN dataset)

---

## 1. Inventory of Files in Authors' Repository (`ImageVisioner/InfraredNUC`)

The authors' primary GitHub repository (`ImageVisioner/InfraredNUC`) contains both MATLAB and native C++ (OpenCV/CMake) implementations:

```
data/external_repos/InfraredNUC/
├── simulation_data1.m               # Stripe & white noise generation script
├── README.md                        # Author instructions & workflow documentation
├── deStrpie.pdf                     # Manuscript draft
├── matlabCode/
│   ├── test1.m                      # Master execution harness (loads 8.jpg, calls ADMM + LRA-SVD)
│   ├── 8.jpg                        # Real uncooled infrared test scene with vertical striping
│   ├── destriped_image.png          # Stage 1 destriped reference result
│   ├── final_denoised_image.png     # Stage 2 final restored reference result
│   └── Functions/
│       ├── admmAlgorithm.m          # Stage 1 ADMM solver with 2D FFT & proximal thresholding
│       ├── lraSvdDenoising.m        # Stage 2 Non-local patch matching & SVD low-rank denoiser
│       ├── getFreqCoeff.m           # FFT transfer function eigenvalues (psf2otf)
│       ├── forwardDiffX.m           # Forward difference stencil [diff(X, 1, 2), X(:, 1) - X(:, end)]
│       ├── forwardDiffY.m           # Forward difference stencil [diff(Y, 1, 1); Y(1, :) - Y(end, :)]
│       ├── divergenceXY.m           # Adjoint negative divergence stencil
│       ├── defGradX.m               # Gradient operator handle wrapper
│       ├── defGradY.m               # Gradient operator handle wrapper
│       └── defDivergence.m          # Divergence operator handle wrapper
├── cppCode/
│   ├── CMakeLists.txt               # CMake configuration linking OpenCV
│   ├── include/
│   │   ├── admm_algorithm.hpp
│   │   ├── lra_svd_denoising.hpp
│   │   └── image_processing.hpp
│   └── src/
│       ├── main.cpp                 # C++ test runner mirroring test1.m
│       ├── admm_algorithm.cpp       # C++ ADMM solver using cv::dft and cv::idft
│       ├── lra_svd_denoising.cpp    # C++ block matching, patch SVD truncation, and aggregation
│       └── image_processing.cpp     # Finite differences, PSNR, SSIM, and frequency domain filters
├── metrics/
│   ├── main.m                       # Batch metric evaluation across folder pairs
│   ├── mainSingImg.m                # Single-image metric runner
│   ├── calculateAVGE.m              # Mean 2D gradient difference (abs(gradX) + abs(gradY))
│   ├── calculateICV.m               # Inverse coefficient of variation (mean / std)
│   ├── calculateMRD.m               # Mean relative deviation with 1e-5 epsilon
│   ├── calculateRoughness.m         # Sobel energy ratio ||h*f||_1 / ||f||_1
│   ├── calculateMSE.m               # Mean squared error
│   ├── calculatePSNR.m              # PSNR via 20 * log10(255 / sqrt(MSE))
│   ├── calculateSSIM.m              # MATLAB ssim(uint8(ir), uint8(filt))
│   └── log.txt                      # Historical benchmark evaluation run logs
└── figs/
    ├── abfig4_00.png                # Paper ablation figure
    ├── dataset1.jpg, dataset2.jpg   # Benchmark scenes
    ├── res1.jpg                     # Visual restoration example
    └── restable.jpg                 # Paper Table I snapshot
```

---

## 2. Reverse-Engineering Findings & Code-vs-Paper Differences

### A. Stage 1 ADMM Formulation Discrepancies
1. **The Eq. 7 vs Eq. 11 Sign Typo**:
   - In Eq. (7) and Eq. (14) of the published text, the residual is $D - C - S$.
   - In Eq. (11), the augmented Lagrangian penalty was printed as $\|D - C + S\|_F^2$, an obvious typographic sign error.
   - *Author Code Reality*: In `admmAlgorithm.m` (lines 64–74), the relation is strictly $C = D - S$, with $S$ as the variable being separated from $D$. The residual is strictly $D - C - S = 0$.
2. **ADMM Operator Decoupling**:
   - The paper writes $C$-update as an FFT and $S$-update with Schatten-$1/2$ shrinkage.
   - In the authors' actual code (`admmAlgorithm.m`), the primary optimization variable is $S$ (the stripe noise), while $C = D - S$ is updated deterministically.
   - The subproblems are:
     - `gradVarX`: row-gradient regularization $\nabla_y S$ via soft thresholding $\text{sign}(v)\max(|v| - \lambda_2/\beta_1, 0)$.
     - `sparseVarY`: column-sparse structure via group soft-thresholding $\frac{r}{\|r\|}\max(\|r\| - \lambda_1/\beta_2, 0)$ or Schatten-$1/2$ low-rank singular value shrinkage.
     - `consistencyVarZ`: horizontal gradient consistency $\nabla_x (D - S)$ via soft thresholding $\text{sign}(v)\max(|v| - \lambda_3/\beta_3, 0)$.
     - `stripeNoise`: closed-form 2D FFT update via the denominator $\beta_3 |\mathcal{F}(\mathbf{D}_x)|^2 + \beta_2 \mathbf{I} + \beta_1 |\mathcal{F}(\mathbf{D}_y)|^2$.
   - Both our direct $C$-update solver and the authors' $S$-update solver solve the identical stationary conditions when $\rho_i = \beta_i$.

### B. Stage 2 Non-Local Low-Rank SVD (LRA-SVD) Discrepancies
1. **Adaptive Rank Criterion (Eq. 26 vs Code)**:
   - In the published paper, Eq. (26) is given as:
     $$\sum_{i=k}^n \sigma_i^2 > \tau^2 \ge \sum_{i=k+1}^n \sigma_i^2, \quad \tau^2 = \gamma (d-1)(L-1)\sigma_n^2.$$
   - In the authors' MATLAB code (`lraSvdDenoising.m`, lines 99–108):
     ```matlab
     ss = 0;
     for j = num:-1:1
         ss = ss + S2(j);
         sk = ss / (ms * ns);
         if sk > nsig^2
             r = j + 1;
             break;
         else
             r = 1;
         end
     end
     ```
     The cumulative tail energy $ss$ divided by the total matrix dimension $(ms \times ns)$ is tested against $\sigma_n^2$. Truncating from the smallest singular value upwards is mathematically identical to selecting rank $r = k$ where the tail noise energy matches the estimated noise variance $\sigma_n^2$.
2. **Patch Weighting**:
   - The authors apply an adaptive weight $w = \frac{d - r}{d}$ to each denoised patch, which naturally down-weights high-rank noisy clusters and increases confidence for clean low-rank textures.
3. **Iterative Regularization**:
   - The author code runs 2 iterations for moderate noise ($\sigma_n \le 30$) with steps $[1.0, 0.5]$ and dynamic noise re-estimation $\sigma_{k+1} = \lambda \sqrt{|\Delta \text{Var}|}$.

### C. Metric Formulation Discrepancies (Eqs. 27–31 vs Author Code)
1. **`AVGE` (Average Vertical Gradient Error, Eq. 27)**:
   - *Paper Text*: Printed as $\frac{1}{N} \sum \|\nabla_y \text{proc} - \nabla_y \text{orig}\|$.
   - *Author Code (`calculateAVGE.m`)*:
     ```matlab
     [gradX_ir, gradY_ir] = gradient(infraredImage);
     [gradX_filt, gradY_filt] = gradient(filteredImage);
     gradDifference = abs(gradX_ir - gradX_filt) + abs(gradY_ir - gradY_filt);
     AVGE = mean(gradDifference(:));
     ```
     Uses 2D central differences (`gradient()`) across **both $X$ and $Y$**, computing the mean absolute gradient difference between the degraded input image (`infraredImage`) and the filtered output (`filteredImage`).
2. **`MRD` (Mean Relative Deviation, Eq. 31)**:
   - *Author Code (`calculateMRD.m`)*:
     ```matlab
     epsilon = 1e-5;
     MRD = mean(mean(abs((infraredImage - filteredImage) ./ (infraredImage + epsilon))));
     ```
     Input image $f$ is `infraredImage`, output $\hat{f}$ is `filteredImage`, normalized by $(f + 10^{-5})$.
3. **`AAD` (Average Absolute Difference, Eq. 30)**:
   - Difference between degraded input $f$ and filtered output $\hat{f}$: $\frac{1}{MN} \sum |f(i, j) - \hat{f}(i, j)|$.
4. **`ICV` (Inverse Coefficient of Variation, Eq. 29)**:
   - *Author Code (`calculateICV.m`)*:
     ```matlab
     mu = mean(filteredImage(:));
     sigma = std(filteredImage(:));
     ICV = mu / sigma;
     ```
5. **`Roughness` $\rho$ (Eq. 28)**:
   - *Paper Text*: Mentions 1D kernel $h = [1, -1]$.
   - *Author Code (`calculateRoughness.m`)*:
     ```matlab
     h = fspecial('sobel');
     infraredEdge = abs(conv2(infraredImage, h, 'same'));
     filteredEdge = abs(conv2(filteredImage, h, 'same'));
     ...
     filteredRoughness = filteredEnergy / filteredNorm;
     ```
     Uses the standard $3 \times 3$ **Sobel filter** for horizontal and vertical convolutions, normalizing by the total image mass. This explains why the empirical roughness in Table I falls in the range $0.30 \le \rho \le 0.65$!

---

## 3. Python Reconciliation Actions Taken

1. **`common/metrics.py`**:
   - Added author-exact functions:
     - `compute_avge(input_img, proc_img)` using 2D central gradient differences matching `calculateAVGE.m`.
     - `compute_mrd(input_img, proc_img, eps=1e-5)` matching `calculateMRD.m`.
     - `compute_aad(input_img, proc_img)` matching Eq. (30).
     - `compute_roughness_sobel(img)` matching `calculateRoughness.m` (Sobel kernel).
     - `compute_icv(img)` matching `calculateICV.m`.
   - On full-reference simulated data, also retained ground-truth comparisons (`psnr_gt`, `ssim_gt`, `aad_gt`).
2. **`mod02_stage1_admm_destriping`**:
   - Reconciled residual sign to $D - C - S$.
   - Preserved 2D FFT closed form and analytical Cardano cubic roots for Schatten-$1/2$ singular value thresholding $\mathcal{H}_\tau^{1/2}(x)$.
3. **`mod03_stage2_patch_lowrank`**:
   - Reconciled SVD rank truncation to backwards noise-energy accumulation matching `lowRankSvdEstimation` in `lraSvdDenoising.m`.
   - Vectorized local block matching using `skimage.util.view_as_windows` in radius $W=12$ ($25 \times 25$ candidate window) with `joblib.Parallel` acceleration.
4. **`mod04_baselines_and_benchmarks`**:
   - Directly imported the 1D Guided Filter logic from `hezw2016/1D-GF` (`rowguidedfilter` and `columnguidedfilter`) into `filter_baselines.py`.
