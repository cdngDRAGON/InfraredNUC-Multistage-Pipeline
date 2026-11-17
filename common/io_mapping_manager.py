"""common/io_mapping_manager.py - Automated Relational Input/Output Mapping Pipeline.

Manages relational file mapping between user/benchmark inputs (uploaded_inputs/)
and rich stage-by-stage multi-artifact outputs (uploaded_outputs/<run_id>/).

For every processed image, produces:
  00_degraded_input.png             - Input degraded infrared image
  01_stage1_stripe_noise.png        - Separated stripe noise S(x, y)
  01_stage1_stripe_noise.mat        - High-precision raw MATLAB array for S
  02_stage1_restored.png            - Destriped intermediate C1(x, y)
  03_stage2_env_noise.png           - Separated environmental noise N(x, y)
  04_stage2_final_clean.png         - Final clean infrared image C2(x, y)
  05_stage_progression_dashboard.png- 300-DPI MATLAB-style multi-panel with 2D zoomed insets,
                                      3D surface energy map, X/Y gradient profiles, SVD curve
  06_baseline_comparison_grid.png   - Multi-panel comparison against baseline methods
  07_matlab_workspace.mat           - Full MATLAB workspace file (scipy.io.savemat)
  stage_metrics_report.csv          - Numerical metrics table (PSNR, SSIM, AVGE, ICV, rho, AAD, MRD)
  run_metadata.json                 - Detailed run parameters, timestamps, hashes, hardware info

Maintains relational database / registry:
  uploaded_outputs/input_output_mapping.csv
  uploaded_outputs/input_output_mapping.json
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from PIL import Image
from scipy import io as sio

from common.matlab_port_engine import run_matlab_pipeline, run_matlab_stage1_admm, run_matlab_stage2_lra_svd
from common.metrics import evaluate_all_metrics
from common.plot_style import set_ieee_style


class IOMappingManager:
    """Manages the uploaded_inputs/ <-> uploaded_outputs/ relational execution lifecycle."""

    def __init__(self, root_dir: Union[str, Path] = "."):
        self.root_dir = Path(root_dir).resolve()
        self.inputs_dir = self.root_dir / "uploaded_inputs"
        self.outputs_dir = self.root_dir / "uploaded_outputs"

        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.mapping_csv = self.outputs_dir / "input_output_mapping.csv"
        self.mapping_json = self.outputs_dir / "input_output_mapping.json"
        self._init_registry()

    def _init_registry(self) -> None:
        """Initialize the master mapping CSV and JSON registries if absent."""
        if not self.mapping_csv.exists():
            headers = [
                "run_id",
                "timestamp",
                "input_filename",
                "input_sha256",
                "source_dataset",
                "height",
                "width",
                "noise_mode",
                "psnr_stage1",
                "psnr_final",
                "ssim_final",
                "avge_final",
                "rho_final",
                "icv_final",
                "stage1_time_sec",
                "stage2_time_sec",
                "total_time_sec",
                "output_dir",
                "dashboard_path",
                "matlab_workspace_path",
            ]
            with open(self.mapping_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        if not self.mapping_json.exists():
            with open(self.mapping_json, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def ingest_input(
        self,
        input_data: Union[str, Path, np.ndarray, bytes],
        filename: Optional[str] = None,
        source_dataset: str = "custom_upload",
    ) -> Tuple[str, Path, np.ndarray]:
        """Ingest an input image into uploaded_inputs/, compute SHA-256, and return a unique run_id.

        Parameters
        ----------
        input_data : str | Path | np.ndarray | bytes
            File path, numpy 2D array, or raw bytes.
        filename : Optional[str]
            Target filename (e.g. 'yohann_checkerboard.png').
        source_dataset : str
            Dataset category name.

        Returns
        -------
        run_id : str
            Unique identifier: YYYYMMDD_HHMMSS_<stem>_<short_sha>
        saved_input_path : Path
            Path to saved input file in uploaded_inputs/.
        img_array : np.ndarray
            2D float64 image array in [0, 255].
        """
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if isinstance(input_data, (str, Path)):
            p = Path(input_data)
            base_name = filename or p.name
            raw_bytes = p.read_bytes()
            img = Image.open(p).convert("L")
            img_arr = np.array(img, dtype=np.float64)
        elif isinstance(input_data, bytes):
            base_name = filename or "uploaded_image.png"
            raw_bytes = input_data
            import io
            img = Image.open(io.BytesIO(input_data)).convert("L")
            img_arr = np.array(img, dtype=np.float64)
        elif isinstance(input_data, np.ndarray):
            base_name = filename or "array_input.png"
            img_arr = np.asarray(input_data, dtype=np.float64)
            if img_arr.ndim == 3:
                img_arr = np.mean(img_arr, axis=2)
            # Clip and normalize if float in [0, 1]
            if np.max(img_arr) <= 1.01:
                img_arr = img_arr * 255.0
            img_arr = np.clip(img_arr, 0.0, 255.0)
            u8 = img_arr.astype(np.uint8)
            pil_img = Image.fromarray(u8)
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        stem = Path(base_name).stem.replace(" ", "_")
        run_id = f"run_{now_str}_{stem}_{sha256[:8]}"

        # Save exact copy into uploaded_inputs/
        saved_input_name = f"{run_id}_input_{base_name}"
        saved_path = self.inputs_dir / saved_input_name
        saved_path.write_bytes(raw_bytes)

        return run_id, saved_path, img_arr

    def process_and_map(
        self,
        run_id: str,
        input_arr: np.ndarray,
        clean_gt: Optional[np.ndarray] = None,
        source_dataset: str = "custom_upload",
        input_filename: str = "input.png",
        params: Optional[Dict[str, Any]] = None,
        run_baselines: bool = False,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Execute the two-stage NUC pipeline, generate all 8 artifacts, and register the run.

        Parameters
        ----------
        run_id : str
            Unique run ID from ingest_input.
        input_arr : np.ndarray
            Degraded image array (H, W) in [0, 255].
        clean_gt : Optional[np.ndarray]
            Clean reference ground truth (if available).
        source_dataset : str
            Dataset category name.
        input_filename : str
            Original input filename.
        params : Optional[Dict[str, Any]]
            Algorithm options for Stage 1 and Stage 2.
        run_baselines : bool
            Whether to execute the 10 comparative baselines for the comparison grid.
        n_jobs : int
            Parallel worker threads (-1 for all cores).

        Returns
        -------
        run_record : Dict[str, Any]
            Complete summary record of the mapped execution.
        """
        set_ieee_style()
        run_dir = self.outputs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        H, W = input_arr.shape
        opts = params or {}
        engine = opts.get("engine", "proposed_ours")
        noise_sigma = float(opts.get("noise_sigma", 1.0))
        max_iter_s1 = int(opts.get("max_iter_s1", 40))
        fast_mode = bool(opts.get("fast_mode", False))

        # 1. Execute Pipeline
        t_start = time.perf_counter()
        if engine == "matlab_port":
            C1, S1, t_s1_dict = run_matlab_stage1_admm(input_arr, max_iter=max_iter_s1)
            t_s1 = t_s1_dict["time_sec"]
            C2, N2, t_s2_dict = run_matlab_stage2_lra_svd(
                C1, noise_sigma=noise_sigma, n_jobs=n_jobs
            )
            t_s2 = t_s2_dict["time_sec"]
            t_total = time.perf_counter() - t_start
        else:
            from mod04_baselines_and_benchmarks.src.proposed_pipeline import solve_proposed_ours
            C2, S_total, diag = solve_proposed_ours(
                input_arr,
                maxiter_s1=max_iter_s1,
                fast_mode=fast_mode,
                n_jobs=n_jobs,
            )
            C1 = diag["C_stage1"]
            S1 = diag["S_stage1"]
            N2 = C1 - C2
            t_s1 = diag["stage1_time_sec"]
            t_s2 = diag["stage2_time_sec"]
            t_total = diag["total_time_sec"]

        # 2. Evaluate Metrics
        ref_img = clean_gt if clean_gt is not None else input_arr
        metrics_deg = evaluate_all_metrics(ref_img, input_arr, degraded=input_arr)
        metrics_s1 = evaluate_all_metrics(ref_img, C1, degraded=input_arr)
        metrics_final = evaluate_all_metrics(ref_img, C2, degraded=input_arr)

        # 3. Save Artifacts 00..04 (PNGs and .MAT)
        path_00 = run_dir / "00_degraded_input.png"
        path_01_png = run_dir / "01_stage1_stripe_noise.png"
        path_01_mat = run_dir / "01_stage1_stripe_noise.mat"
        path_02 = run_dir / "02_stage1_restored.png"
        path_03 = run_dir / "03_stage2_env_noise.png"
        path_04 = run_dir / "04_stage2_final_clean.png"

        Image.fromarray(np.clip(input_arr, 0, 255).astype(np.uint8)).save(path_00)
        Image.fromarray(np.clip(C1, 0, 255).astype(np.uint8)).save(path_02)
        Image.fromarray(np.clip(C2, 0, 255).astype(np.uint8)).save(path_04)

        # Save stripe normalized to 0-255 visualization
        s_vis = S1 - np.min(S1)
        s_vis = (s_vis / (np.max(s_vis) + 1e-8) * 255.0).astype(np.uint8)
        Image.fromarray(s_vis).save(path_01_png)
        sio.savemat(path_01_mat, {"stripeNoise": S1, "run_id": run_id})

        # Save environmental noise normalized visualization
        n_vis = N2 - np.min(N2)
        n_vis = (n_vis / (np.max(n_vis) + 1e-8) * 255.0).astype(np.uint8)
        Image.fromarray(n_vis).save(path_03)

        # 4. Generate Artifact 05: 300-DPI MATLAB-Style Dashboard
        path_05 = run_dir / "05_stage_progression_dashboard.png"
        self._generate_dashboard(
            path_05, input_arr, C1, C2, S1, N2, metrics_deg, metrics_s1, metrics_final, t_s1, t_s2
        )

        # 5. Generate Artifact 06: Baseline Comparison Grid (if requested)
        path_06 = run_dir / "06_baseline_comparison_grid.png"
        if run_baselines:
            self._generate_baseline_grid(path_06, input_arr, ref_img, C2)
        else:
            # Generate compact visual verification grid
            self._generate_compact_verification_grid(path_06, input_arr, C1, C2, S1)

        # 6. Save Artifact 07: MATLAB Workspace .mat
        path_07 = run_dir / "07_matlab_workspace.mat"
        sio.savemat(
            path_07,
            {
                "D": input_arr,
                "C1": C1,
                "C2": C2,
                "S": S1,
                "N": N2,
                "metrics_deg": metrics_deg,
                "metrics_stage1": metrics_s1,
                "metrics_final": metrics_final,
                "runtime": {
                    "stage1_sec": t_s1,
                    "stage2_sec": t_s2,
                    "total_sec": t_total,
                },
                "run_id": run_id,
            },
        )

        # 7. Generate Stage Metrics CSV and Run Metadata JSON
        metrics_csv_path = run_dir / "stage_metrics_report.csv"
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Stage", "PSNR_dB", "SSIM", "AVGE", "Roughness_rho", "ICV", "AAD", "MRD", "Time_sec"])
            writer.writerow(["00_Degraded", f"{metrics_deg['PSNR']:.2f}", f"{metrics_deg['SSIM']:.4f}", f"{metrics_deg['AVGE']:.4f}", f"{metrics_deg['Roughness_rho']:.4f}", f"{metrics_deg['ICV']:.2f}", f"{metrics_deg['AAD']:.4f}", f"{metrics_deg['MRD']:.4f}", "0.000"])
            writer.writerow(["01_Stage1_ADMM", f"{metrics_s1['PSNR']:.2f}", f"{metrics_s1['SSIM']:.4f}", f"{metrics_s1['AVGE']:.4f}", f"{metrics_s1['Roughness_rho']:.4f}", f"{metrics_s1['ICV']:.2f}", f"{metrics_s1['AAD']:.4f}", f"{metrics_s1['MRD']:.4f}", f"{t_s1:.3f}"])
            writer.writerow(["02_Stage2_Final", f"{metrics_final['PSNR']:.2f}", f"{metrics_final['SSIM']:.4f}", f"{metrics_final['AVGE']:.4f}", f"{metrics_final['Roughness_rho']:.4f}", f"{metrics_final['ICV']:.2f}", f"{metrics_final['AAD']:.4f}", f"{metrics_final['MRD']:.4f}", f"{t_s2:.3f}"])

        input_bytes = input_arr.tobytes()
        input_sha256 = hashlib.sha256(input_bytes).hexdigest()

        run_metadata = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "input_filename": input_filename,
            "input_sha256": input_sha256,
            "source_dataset": source_dataset,
            "dimensions": {"height": H, "width": W},
            "parameters": opts,
            "timing": {
                "stage1_time_sec": t_s1,
                "stage2_time_sec": t_s2,
                "total_time_sec": t_total,
                "stage1_split_pct": (t_s1 / (t_total + 1e-12)) * 100.0,
                "stage2_split_pct": (t_s2 / (t_total + 1e-12)) * 100.0,
            },
            "metrics": {
                "degraded": metrics_deg,
                "stage1": metrics_s1,
                "final": metrics_final,
            },
            "artifacts": {
                "00_degraded_input": str(path_00.relative_to(self.root_dir)),
                "01_stage1_stripe_noise_png": str(path_01_png.relative_to(self.root_dir)),
                "01_stage1_stripe_noise_mat": str(path_01_mat.relative_to(self.root_dir)),
                "02_stage1_restored": str(path_02.relative_to(self.root_dir)),
                "03_stage2_env_noise": str(path_03.relative_to(self.root_dir)),
                "04_stage2_final_clean": str(path_04.relative_to(self.root_dir)),
                "05_stage_progression_dashboard": str(path_05.relative_to(self.root_dir)),
                "06_baseline_comparison_grid": str(path_06.relative_to(self.root_dir)),
                "07_matlab_workspace_mat": str(path_07.relative_to(self.root_dir)),
                "stage_metrics_report_csv": str(metrics_csv_path.relative_to(self.root_dir)),
            },
        }

        meta_json_path = run_dir / "run_metadata.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2)

        # 8. Append to Master Registries
        self._append_to_registry(run_metadata)

        return run_metadata

    def _append_to_registry(self, meta: Dict[str, Any]) -> None:
        """Atomically append run to input_output_mapping.csv and JSON."""
        # 1. Append CSV row
        csv_row = [
            meta["run_id"],
            meta["timestamp"],
            meta["input_filename"],
            meta["input_sha256"][:12],
            meta["source_dataset"],
            meta["dimensions"]["height"],
            meta["dimensions"]["width"],
            meta["parameters"].get("noise_mode", "standard"),
            f"{meta['metrics']['stage1']['PSNR']:.2f}",
            f"{meta['metrics']['final']['PSNR']:.2f}",
            f"{meta['metrics']['final']['SSIM']:.4f}",
            f"{meta['metrics']['final']['AVGE']:.4f}",
            f"{meta['metrics']['final']['Roughness_rho']:.4f}",
            f"{meta['metrics']['final']['ICV']:.2f}",
            f"{meta['timing']['stage1_time_sec']:.3f}",
            f"{meta['timing']['stage2_time_sec']:.3f}",
            f"{meta['timing']['total_time_sec']:.3f}",
            str(self.outputs_dir / meta["run_id"]),
            meta["artifacts"]["05_stage_progression_dashboard"],
            meta["artifacts"]["07_matlab_workspace_mat"],
        ]
        with open(self.mapping_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)

        # 2. Update JSON
        all_runs = []
        if self.mapping_json.exists():
            try:
                with open(self.mapping_json, "r", encoding="utf-8") as f:
                    all_runs = json.load(f)
            except Exception:
                all_runs = []

        # Remove existing if same run_id
        all_runs = [r for r in all_runs if r.get("run_id") != meta["run_id"]]
        all_runs.append(meta)

        with open(self.mapping_json, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2)

    def _generate_dashboard(
        self,
        save_path: Path,
        D: np.ndarray,
        C1: np.ndarray,
        C2: np.ndarray,
        S: np.ndarray,
        N: np.ndarray,
        m_deg: Dict[str, float],
        m_s1: Dict[str, float],
        m_fin: Dict[str, float],
        t1: float,
        t2: float,
    ) -> None:
        """Render 300-DPI MATLAB-style multi-panel stage progression dashboard."""
        fig = plt.figure(figsize=(16, 10), dpi=300, facecolor="white")
        gs = fig.add_gridspec(2, 5, height_ratios=[1.1, 0.9], hspace=0.30, wspace=0.25)

        # Top row: 5 image panels
        # 1. Degraded
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(D, cmap="gray", vmin=0, vmax=255)
        ax1.set_title(f"(a) Degraded Input\nPSNR: {m_deg['PSNR']:.2f} dB", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # 2. Stage 1 Restored
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(C1, cmap="gray", vmin=0, vmax=255)
        ax2.set_title(f"(b) Stage 1 Restored\nPSNR: {m_s1['PSNR']:.2f} dB (+{m_s1['PSNR']-m_deg['PSNR']:.2f})", fontsize=10, fontweight="bold")
        ax2.axis("off")

        # 3. Final Stage 2 Clean
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(C2, cmap="gray", vmin=0, vmax=255)
        ax3.set_title(f"(c) Final Stage 2 Clean\nPSNR: {m_fin['PSNR']:.2f} dB | SSIM: {m_fin['SSIM']:.4f}", fontsize=10, fontweight="bold")
        ax3.axis("off")

        # 4. Separated Stripe S
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(S, cmap="seismic", vmin=-np.max(np.abs(S)), vmax=np.max(np.abs(S)))
        ax4.set_title(f"(d) Extracted Stripe S(x,y)\nStd: {np.std(S):.2f}", fontsize=10, fontweight="bold")
        ax4.axis("off")
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # 5. Residual Noise N
        ax5 = fig.add_subplot(gs[0, 4])
        im5 = ax5.imshow(N, cmap="coolwarm", vmin=-np.max(np.abs(N)), vmax=np.max(np.abs(N)))
        ax5.set_title(f"(e) Residual Noise N(x,y)\nStd: {np.std(N):.2f}", fontsize=10, fontweight="bold")
        ax5.axis("off")
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        # Bottom row: Diagnostics
        # 6. 3D Global Energy Surface Map of Final Clean Image
        ax6 = fig.add_subplot(gs[1, 0:2], projection="3d")
        H, W = C2.shape
        step_down = max(1, H // 32)
        X_grid, Y_grid = np.meshgrid(np.arange(0, W, step_down), np.arange(0, H, step_down))
        Z_grid = C2[::step_down, ::step_down]
        surf = ax6.plot_surface(X_grid, Y_grid, Z_grid, cmap="viridis", edgecolor="none", alpha=0.85)
        ax6.set_title("3D Global Energy Surface Map (Final)", fontsize=10, fontweight="bold")
        ax6.set_xlabel("Width", fontsize=8)
        ax6.set_ylabel("Height", fontsize=8)
        ax6.set_zlabel("Intensity", fontsize=8)
        ax6.view_init(elev=45, azim=-60)

        # 7. Directional Scanline Profiles (Horizontal X vs Vertical Y)
        ax7 = fig.add_subplot(gs[1, 2:4])
        mid_row = H // 2
        ax7.plot(D[mid_row, :], label="Degraded (Row Profile)", color="#D9531F", alpha=0.6, linewidth=1.0)
        ax7.plot(C1[mid_row, :], label="Stage 1 Restored", color="#EDB120", alpha=0.75, linewidth=1.2)
        ax7.plot(C2[mid_row, :], label="Stage 2 Clean (Ours)", color="#0072BD", linewidth=1.5)
        ax7.set_title(f"Horizontal Scanline Intensity Profile (Row {mid_row})", fontsize=10, fontweight="bold")
        ax7.set_xlabel("Horizontal Pixel Coordinate x", fontsize=9)
        ax7.set_ylabel("Intensity", fontsize=9)
        ax7.legend(loc="upper right", fontsize=8)
        ax7.grid(True, linestyle="--", alpha=0.5)

        # 8. SVD Singular Value Distribution Curve
        ax8 = fig.add_subplot(gs[1, 4])
        _, s_deg, _ = np.linalg.svd(D - np.mean(D), full_matrices=False)
        _, s_c2, _ = np.linalg.svd(C2 - np.mean(C2), full_matrices=False)
        _, s_str, _ = np.linalg.svd(S, full_matrices=False)

        k_max = min(30, len(s_deg))
        ax8.semilogy(s_deg[:k_max], label="Degraded", marker="o", markersize=3, color="#D9531F")
        ax8.semilogy(s_c2[:k_max], label="Clean", marker="s", markersize=3, color="#0072BD")
        ax8.semilogy(s_str[:k_max], label="Stripe S", marker="^", markersize=3, color="#7E2F8E")
        ax8.set_title("SVD Energy Distribution", fontsize=10, fontweight="bold")
        ax8.set_xlabel("Singular Value Index", fontsize=9)
        ax8.set_ylabel("Singular Value (log)", fontsize=9)
        ax8.legend(loc="upper right", fontsize=7)
        ax8.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(
            f"Infrared NUC Multi-Stage Pipeline Progression\nStage 1: {t1:.2f}s (~{t1/(t1+t2+1e-8)*100:.0f}%) | Stage 2: {t2:.2f}s (~{t2/(t1+t2+1e-8)*100:.0f}%) | Total: {t1+t2:.2f}s",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def _generate_compact_verification_grid(
        self, save_path: Path, D: np.ndarray, C1: np.ndarray, C2: np.ndarray, S: np.ndarray
    ) -> None:
        """Render a clean 4-panel visual verification grid."""
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=300, facecolor="white")
        axes[0].imshow(D, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Input Degraded", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(C1, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Stage 1 Destriped", fontsize=9, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(C2, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("Stage 2 Final Clean (Ours)", fontsize=9, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(S, cmap="seismic")
        axes[3].set_title("Isolated Stripe Component S", fontsize=9, fontweight="bold")
        axes[3].axis("off")

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def _generate_baseline_grid(
        self, save_path: Path, D: np.ndarray, clean: np.ndarray, ours: np.ndarray
    ) -> None:
        """Generate comparative 2x6 mosaic comparing all methods."""
        from mod04_baselines_and_benchmarks.src.run_mod04 import evaluate_scene_all_methods
        results = evaluate_scene_all_methods(clean, D, fast_mode=True)

        fig, axes = plt.subplots(2, 6, figsize=(18, 6.5), dpi=300, facecolor="white")
        ax_flat = axes.ravel()

        methods_order = [
            "Degraded",
            "1D-GF",
            "1D-WLS",
            "ADOM",
            "BM3D",
            "DLS-NUC",
            "SNRCNN",
            "LRSID",
            "GSRSR",
            "SNRWDNN",
            "DestripeCycleGAN",
            "Ours",
        ]

        for i, m_name in enumerate(methods_order):
            ax = ax_flat[i]
            if m_name in results:
                im_data = results[m_name]["restored"]
                psnr_val = results[m_name]["metrics"]["PSNR"]
                ssim_val = results[m_name]["metrics"]["SSIM"]
                ax.imshow(im_data, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{m_name}\nPSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}", fontsize=8, fontweight="bold")
            else:
                ax.imshow(D, cmap="gray", vmin=0, vmax=255)
                ax.set_title(m_name, fontsize=8)
            ax.axis("off")

        fig.suptitle("Comparative Evaluation Across All 11 NUC Methods", fontsize=11, fontweight="bold", y=0.98)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def list_mappings(self) -> List[Dict[str, Any]]:
        """Return list of all registered runs from input_output_mapping.json."""
        if not self.mapping_json.exists():
            return []
        try:
            with open(self.mapping_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_mapping(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single run metadata by run_id."""
        mappings = self.list_mappings()
        for m in mappings:
            if m.get("run_id") == run_id:
                return m
        return None
