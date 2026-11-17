# Module 4: 11-Method Baselines and Benchmarks Suite

## 1. Overview
Module 4 implements the complete comparative destriping benchmark suite from Section IV-B of the IEEE Sensors Journal paper:
- **Filtering-based baselines**: `1D-GF` (Cao et al. 2016), `1D-WLS` (Dong et al. 2018), `BM3D` (Dabov et al. 2007)
- **Optimization-based baselines**: `ADOM` (Kim et al. 2023), `LRSID` (Chang et al. 2016), `GSRSR` (Chen et al. 2017)
- **Deep learning-based baselines**: `SNRCNN` (Kuai et al. 2017), `DLS-NUC` (He et al. 2019), `SNRWDNN` (Guan et al. 2019), `DestripeCycleGAN` (Xiao et al. 2020)
- **Proposed Unified Pipeline**: `Ours` (Stage 1 ADMM + Stage 2 Non-Local Patch Low-Rank SVD)

All 11 methods are evaluated across all 6 benchmark infrared scenes (`Img1` to `Img6`) using the 7 quantitative metrics from Section IV-A2:
- **AVGE** (Eq. 27): Average Vertical Gradient Error ($\downarrow$)
- **MRD** (Eq. 31): Mean Relative Deviation ($\downarrow$)
- **AAD** (Eq. 30): Average Absolute Difference ($\downarrow$)
- **ICV** (Eq. 29): Inverse Coefficient of Variation ($\uparrow$)
- **$\rho$** (Eq. 28): Roughness index ($\downarrow$)
- **PSNR (dB)**: Peak Signal-to-Noise Ratio ($\uparrow$)
- **SSIM**: Structural Similarity Index ($\uparrow$)

---

## 2. Directory Layout
```
mod04_baselines_and_benchmarks/
├── src/
│   ├── filter_baselines.py         # 1D-GF (Cao 2016), 1D-WLS (Dong 2018), and BM3D patch filter
│   ├── optim_baselines.py          # ADOM (Kim 2023), LRSID (Chang 2016), and GSRSR (Chen 2017)
│   ├── learning_baselines.py       # SNRCNN, DLS-NUC, SNRWDNN, DestripeCycleGAN inference models
│   ├── proposed_pipeline.py        # Unified wrapper chaining Stage 1 (mod02) + Stage 2 (mod03)
│   └── run_mod04.py                # Master benchmark runner generating Table I CSV and Figs. 10-15
├── tests/
│   └── test_mod04.py               # Unit tests verifying all 11 methods run cleanly and Ours excels
├── docs/
│   ├── README.md                   # This summary guide
│   ├── ieee_section_mod04.tex      # IEEE LaTeX Section IV-B manuscript
│   └── table1_quantitative.tex     # Booktabs Table I (bold best, underlined second-best)
├── resources/
│   ├── theoretical_guide.md        # Mathematical comparison of failure modes across paradigms
│   └── viva_prep.md                # Viva Q&A on 1D-GF edge blurring, ADOM limits, and Ours
└── results/                        # table1_quantitative_evaluation.csv & fig10..fig15_*.png
```

---

## 3. Qualitative Figures 10–15
For each scene, a $2 \times 6$ method panel is generated showing `Degraded Img` + the 11 comparison methods:
- Top panel: Restored 2D infrared image with red bounding box and magnified zoom inset in the corner.
- Bottom panel: 3D Global Energy surface profile (`cmap='jet'`) displaying the pixel intensity landscape $Z = \mathbf{C}(X, Y)$.

Generated figures:
- `fig10_yohann_scene1.png` (Scene 1: Yohann Checkerboard)
- `fig11_yohann_scene2.png` (Scene 2: Yohann Indoor Monitor)
- `fig12_he_scene3.png` (Scene 3: He Building Skyline)
- `fig13_he_scene4.png` (Scene 4: He House Pedestrian)
- `fig14_tno_scene5.png` (Scene 5: TNO Facade Windows)
- `fig15_llvip_scene6.png` (Scene 6: LLVIP Street Cars)

---

## 4. Execution Commands
```bash
# Execute master benchmark across all 11 methods and generate figures 10-15
python3 -m mod04_baselines_and_benchmarks.src.run_mod04

# Run test suite
pytest mod04_baselines_and_benchmarks/tests/ -v
```
