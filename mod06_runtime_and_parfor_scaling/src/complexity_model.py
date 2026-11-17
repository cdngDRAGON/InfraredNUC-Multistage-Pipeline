"""Computational Complexity Model and Table II Formatter (Section IV-D).

Theoretical & Empirical Complexity Analysis:
  - Stage 1 ADMM: O(K1 * (MN log(MN) + min(M^2 N, M N^2)))
  - Stage 2 Patch SVD: O((MN / s^2) * (R^2 B^2 + B^2 L^2))
  - Multi-Core Parallel Scaling (Parfor): Amdahl's Law S(p) = 1 / ((1 - P) + P/p)
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


def generate_table2_latex(
    scaling_df: pd.DataFrame,
    resolution_df: pd.DataFrame,
) -> str:
    """Generate IEEE-formatted LaTeX Table II for runtime and parallel scaling."""
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Computational Complexity, Multi-Core Scaling, and Hardware Telemetry (Table II Equiv.)}",
        r"\label{tab:runtime_scaling}",
        r"\begin{tabular}{lccccc}",
        r"\hline\hline",
        r"\multicolumn{6}{c}{\textbf{Part A: Multi-Threaded Parallel Scaling ($256 \times 256$ IR Scene)}} \\",
        r"\hline",
        r"\textbf{Cores ($p$)} & \textbf{Time (s)} $\downarrow$ & \textbf{Speedup ($S_p$)} $\uparrow$ & \textbf{Efficiency (\%)} $\uparrow$ & \textbf{Amdahl $S_p$} & \textbf{Parallel Fraction $P$} \\",
        r"\hline",
    ]

    for _, row in scaling_df.iterrows():
        p = int(row["Cores"])
        t = row["Time_sec"]
        sp = row["Speedup"]
        eff = row["Efficiency_pct"]
        amd = row["Amdahl_Speedup"]
        p_val = row["Amdahl_P"]
        latex_lines.append(
            f"{p:d} & {t:.3f} & {sp:.2f}$\\times$ & {eff:.1f}\\% & {amd:.2f}$\\times$ & {p_val:.2f} \\\\"
        )

    latex_lines.extend(
        [
            r"\hline",
            r"\multicolumn{6}{c}{\textbf{Part B: Resolution Scaling and Memory Footprint}} \\",
            r"\hline",
            r"\textbf{Resolution} & \textbf{Pixels} & \textbf{Stage 1 (s)} & \textbf{Stage 2 (s)} & \textbf{Total Time (s)} & \textbf{Peak RAM (MB)} \\",
            r"\hline",
        ]
    )

    for _, row in resolution_df.iterrows():
        res = row["Resolution"]
        pix = int(row["Pixels"])
        s1 = row["Stage1_Time_s"]
        s2 = row["Stage2_Time_s"]
        tot = row["Total_Time_s"]
        ram = row["Peak_RAM_MB"]
        latex_lines.append(
            f"{res} & {pix:,d} & {s1:.3f} & {s2:.3f} & {tot:.3f} & {ram:.1f} \\\\"
        )

    latex_lines.extend(
        [
            r"\hline\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex_lines)
