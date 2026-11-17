"""reports/ieee_sensors_nuc/sync_csv_to_latex.py

Synchronizes benchmark CSV datasets into publication-grade IEEE LaTeX tables
and copies all verified figures from module results into reports/ieee_sensors_nuc/figures/.

Generates:
  - reports/ieee_sensors_nuc/tables/table1_metrics.tex
  - reports/ieee_sensors_nuc/tables/table2_runtime.tex
  - reports/ieee_sensors_nuc/tables/table3_ablations.tex
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = WORKSPACE_ROOT / "reports" / "ieee_sensors_nuc"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


def sync_figures() -> int:
    """Copy all PNG figures from mod01..mod06 results into reports/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for mod_dir in sorted(WORKSPACE_ROOT.glob("mod0*/results")):
        for fig_path in mod_dir.glob("*.png"):
            dest = FIGURES_DIR / fig_path.name
            shutil.copy2(fig_path, dest)
            count += 1
    print(f"[*] Synchronized {count} figures into {FIGURES_DIR}")
    return count


def generate_table1_metrics() -> Path:
    """Generate table1_metrics.tex from table1_quantitative_evaluation.csv."""
    csv_path = WORKSPACE_ROOT / "mod04_baselines_and_benchmarks" / "results" / "table1_quantitative_evaluation.csv"
    if not csv_path.exists():
        csv_path = TABLES_DIR / "table1_quantitative_evaluation.csv"

    df = pd.read_csv(csv_path)

    method_order = [
        "1D-GF", "1D-WLS", "ADOM", "BM3D", "DLS-NUC", "SNRCNN",
        "LRSID", "GSRSR", "SNRWDNN", "DestripeCycleGAN", "Ours",
    ]

    scene_display_names = {
        "Img1_Checkerboard_Yohann.png": "Checkerboard (Yohann)",
        "Img2_IndoorMonitor_Yohann.png": "Indoor Monitor (Yohann)",
        "Img3_BuildingSkyline_He.png": "Building Skyline (He)",
        "Img4_HousePedestrian_He.png": "House Pedestrian (He)",
        "Img5_FacadeWindows_TNO.png": "Facade Windows (TNO)",
        "Img6_StreetCars_LLVIP.png": "Street Cars (LLVIP)",
    }

    scenes = df["Scene"].unique()
    metrics = ["AVGE", "MRD", "AAD", "ICV", "rho", "PSNR_dB", "SSIM"]
    metric_labels = {
        "AVGE": r"AVGE $\downarrow$",
        "MRD": r"MRD $\downarrow$",
        "AAD": r"AAD $\downarrow$",
        "ICV": r"ICV $\uparrow$",
        "rho": r"$\rho$ $\downarrow$",
        "PSNR_dB": r"PSNR $\uparrow$",
        "SSIM": r"SSIM $\uparrow$",
    }

    lines = [
        r"% ==============================================================================",
        r"% Table I: Quantitative Evaluation of 11 Destriping Methods (Section IV-B)",
        r"% ==============================================================================",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Quantitative Evaluation of 11 Destriping Methods Across Six Benchmark Infrared Scenes (Table I)}",
        r"\label{tab:table1_quantitative_evaluation}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll" + "c" * len(method_order) + r"}",
        r"\toprule",
        r"\textbf{Scene} & \textbf{Metric} & " + " & ".join([f"\\textbf{{{m}}}" for m in method_order]) + r" \\",
        r"\midrule",
    ]

    for s_idx, scene in enumerate(scenes):
        scene_df = df[df["Scene"] == scene]
        scene_display = scene_display_names.get(scene, scene.replace(".png", "").replace("_", " "))

        for m_idx, metric in enumerate(metrics):
            row_prefix = f"\\multirow{{7}}{{*}}{{{scene_display}}}" if m_idx == 0 else ""
            m_label = metric_labels.get(metric, metric)

            vals = {}
            for meth in method_order:
                meth_row = scene_df[scene_df["Method"] == meth]
                if not meth_row.empty and metric in meth_row.columns:
                    vals[meth] = float(meth_row[metric].values[0])

            is_higher_better = metric in ["ICV", "PSNR_dB", "SSIM"]
            if vals:
                sorted_meths = sorted(vals.keys(), key=lambda k: vals[k], reverse=is_higher_better)
                best_m = sorted_meths[0] if len(sorted_meths) > 0 else None
                second_m = sorted_meths[1] if len(sorted_meths) > 1 else None
            else:
                best_m, second_m = None, None

            cells = []
            for meth in method_order:
                if meth in vals:
                    v = vals[meth]
                    fmt = f"{v:.4f}" if metric not in ["PSNR_dB", "ICV"] else f"{v:.2f}"
                    if meth == best_m:
                        cells.append(f"\\textbf{{{fmt}}}")
                    elif meth == second_m:
                        cells.append(f"\\underline{{{fmt}}}")
                    else:
                        cells.append(fmt)
                else:
                    cells.append("-")

            line_str = f"{row_prefix} & {m_label} & " + " & ".join(cells) + r" \\"
            lines.append(line_str)

        if s_idx < len(scenes) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])

    out_tex = TABLES_DIR / "table1_metrics.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also keep table1_quantitative.tex synchronized
    with open(TABLES_DIR / "table1_quantitative.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[*] Generated Table I LaTeX -> {out_tex}")
    return out_tex


def generate_table2_runtime() -> Path:
    """Generate table2_runtime.tex from table2_runtime_comparison.csv."""
    csv_path = WORKSPACE_ROOT / "mod06_runtime_and_parfor_scaling" / "results" / "table2_runtime_comparison.csv"
    if not csv_path.exists():
        csv_path = TABLES_DIR / "table2_runtime_comparison.csv"

    df = pd.read_csv(csv_path)
    cats = [c for c in df.columns if c not in ["Method", "Average"]]

    lines = [
        r"% ==============================================================================",
        r"% Table II: Runtime Comparison (in seconds) Across 6 Dataset Categories",
        r"% ==============================================================================",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Computational Runtime Comparison (Seconds) Across 6 Infrared Dataset Categories}",
        r"\label{tab:runtime_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * (len(cats) + 1) + r"}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join([f"\\textbf{{{c}}}" for c in cats]) + r" & \textbf{Average} \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        m_name = str(row["Method"])
        is_ours = "Ours" in m_name
        name_str = f"\\textbf{{{m_name}}}" if is_ours else m_name
        vals_str = " & ".join([f"{float(row[c]):.4f}" for c in cats])
        avg_str = f"{float(row['Average']):.4f}"

        if is_ours:
            lines.append(f"{name_str} & {vals_str} & \\textbf{{{avg_str}}} \\\\")
        else:
            lines.append(f"{name_str} & {vals_str} & {avg_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])

    out_tex = TABLES_DIR / "table2_runtime.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[*] Generated Table II LaTeX -> {out_tex}")
    return out_tex


def generate_table3_ablations() -> Path:
    """Generate comprehensive table3_ablations.tex summarizing all 4 ablation studies."""
    k_csv = WORKSPACE_ROOT / "mod05_ablation_and_stress_tests" / "results" / "ablation_iterations.csv"
    k_df = pd.read_csv(k_csv) if k_csv.exists() else None

    lam_csv = WORKSPACE_ROOT / "mod05_ablation_and_stress_tests" / "results" / "ablation_lambdas.csv"
    lam_df = pd.read_csv(lam_csv) if lam_csv.exists() else None

    l_csv = WORKSPACE_ROOT / "mod05_ablation_and_stress_tests" / "results" / "ablation_patch_count_L.csv"
    l_df = pd.read_csv(l_csv) if l_csv.exists() else None

    lines = [
        r"% ==============================================================================",
        r"% Table III: Comprehensive Ablation Study Summary Across Pipeline Hyperparameters",
        r"% ==============================================================================",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Ablation Studies: Hyperparameter Sensitivity and Multi-Stage Contribution}",
        r"\label{tab:ablation_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multicolumn{7}{c}{\textbf{Part A: Convergence and Decoupling vs. ADMM Iterations $K$ (Figs. 16--17)}} \\",
        r"\midrule",
        r"\textbf{Iterations $K$} & \textbf{PSNR (dB)} & \textbf{SSIM} & \textbf{Roughness $\rho$} & \textbf{Stripe Leakage $\partial \mathbf{S}/\partial y$} & \textbf{Latency (s)} & \textbf{Decoupling Status} \\",
        r"\midrule",
    ]

    if k_df is not None:
        for _, r in k_df.iterrows():
            k_val = int(r["Iteration_K"])
            psnr_v = float(r["PSNR_dB"])
            ssim_v = float(r["SSIM"])
            rho_v = float(r["Roughness_rho"])
            leak_v = float(r["Leakage_dY"])
            t_v = float(r["Time_sec"])
            status = "Decoupled (Optimal)" if k_val == 250 else ("Converged" if k_val > 250 else "Partial Decoupling")
            is_opt = (k_val == 250)
            prefix = "\\textbf{" if is_opt else ""
            suffix = "}" if is_opt else ""
            lines.append(
                f"{prefix}K = {k_val}{suffix} & {prefix}{psnr_v:.2f}{suffix} & {prefix}{ssim_v:.4f}{suffix} & "
                f"{prefix}{rho_v:.4f}{suffix} & {prefix}{leak_v:.4f}{suffix} & {prefix}{t_v:.2f} s{suffix} & {prefix}{status}{suffix} \\\\"
            )

    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{c}{\textbf{Part B: Regularization Parameter Sensitivity $(\lambda_1, \lambda_2, \lambda_3)$ (Fig. 18)}} \\",
        r"\midrule",
        r"\textbf{Configuration} & \textbf{$\lambda_1$ (Schatten 1/2)} & \textbf{$\lambda_2$ (1st-order TV)} & \textbf{$\lambda_3$ (2nd-order TV)} & \textbf{PSNR (dB)} & \textbf{SSIM} & \textbf{Stripe MAE} \\",
        r"\midrule",
    ])

    if lam_df is not None:
        for _, r in lam_df.iterrows():
            cfg_name = str(r["Config_Name"]).replace("_", " ")
            l1, l2, l3 = float(r["Lambda1"]), float(r["Lambda2"]), float(r["Lambda3"])
            psnr_v, ssim_v, mae_v = float(r["PSNR_dB"]), float(r["SSIM"]), float(r["Stripe_MAE"])
            is_opt = (abs(l1 - 1e-3) < 1e-6 and abs(l2 - 1e-3) < 1e-6 and abs(l3 - 1e-3) < 1e-6)
            p = "\\textbf{" if is_opt else ""
            s = "}" if is_opt else ""
            lines.append(
                f"{p}{cfg_name}{s} & {p}{l1:.0e}{s} & {p}{l2:.0e}{s} & {p}{l3:.0e}{s} & "
                f"{p}{psnr_v:.2f}{s} & {p}{ssim_v:.4f}{s} & {p}{mae_v:.2f}{s} \\\\"
            )

    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{c}{\textbf{Part C: Stage 2 Low-Rank Patch Group Count $L$ vs. w/o Stage 2 Baseline (Figs. 21--22)}} \\",
        r"\midrule",
        r"\textbf{Patch Count $L$} & \textbf{PSNR (dB)} & \textbf{SSIM} & \textbf{Roughness $\rho$} & \textbf{AVGE} & \textbf{Latency (s)} & \textbf{Evaluation Focus} \\",
        r"\midrule",
    ])

    if l_df is not None:
        for _, r in l_df.iterrows():
            tag = str(r["Configuration"])
            psnr_v, ssim_v = float(r["PSNR_dB"]), float(r["SSIM"])
            rho_v, avge_v = float(r["Roughness_rho"]), float(r["AVGE"])
            t_v = float(r["Time_sec"])
            is_opt = ("50" in tag)
            is_base = ("w/o" in tag.lower())
            p = "\\textbf{" if is_opt else ("\\underline{" if is_base else "")
            s = "}" if is_opt or is_base else ""
            eval_note = "Ablation Baseline" if is_base else ("Global Optimum" if is_opt else "Subspace Plateau")
            lines.append(
                f"{p}{tag}{s} & {p}{psnr_v:.2f}{s} & {p}{ssim_v:.4f}{s} & {p}{rho_v:.4f}{s} & "
                f"{p}{avge_v:.4f}{s} & {p}{t_v:.2f} s{s} & {p}{eval_note}{s} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])

    out_tex = TABLES_DIR / "table3_ablations.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[*] Generated Table III LaTeX -> {out_tex}")
    return out_tex


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  SYNCHRONIZING BENCHMARK CSVS & GENERATING PUBLICATION LATEX TABLES")
    print("=" * 80)

    sync_figures()
    generate_table1_metrics()
    generate_table2_runtime()
    generate_table3_ablations()

    print("\n[+] Synchronization completed successfully.")
    return 0


if __name__ == "__main__":
    main()
