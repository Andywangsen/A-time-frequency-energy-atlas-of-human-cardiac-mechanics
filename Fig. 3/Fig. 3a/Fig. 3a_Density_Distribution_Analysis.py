# -*- coding: utf-8 -*-
"""
==============================================================================
Density Plot Script

Function: Load Ctrl-H and HF data from Excel files,
          plot Kernel Density Estimation (KDE) density plot, perform statistical tests and calculate effect size.

Source: Fig. 3a_Ctrl-H.xlsx, Fig. 3a_HF.xlsx
Output: Fig. 3a_Density_Plot_TopJournal.png
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from tkinter import filedialog
import os

# Figure Config
FIG_WIDTH = 10
FIG_HEIGHT = 5

# Data column name
DATA_COLUMN = "ER_{2-100Hz}[0-20/0-30%]"


def select_files():
    """
    Select or automatically load Excel files.

    Prioritize checking default file location, if not exists then pop up file selection dialog.

    Returns:
        tuple: (ctrl_h_file, hf_file)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ctrl_h_target = os.path.join(current_dir, "Fig. 3a_Ctrl-H.xlsx")
    hf_target = os.path.join(current_dir, "Fig. 3a_HF.xlsx")

    if os.path.exists(ctrl_h_target) and os.path.exists(hf_target):
        return ctrl_h_target, hf_target

    root = tk.Tk()
    root.withdraw()

    ctrl_h_file = filedialog.askopenfilename(
        title="Select Ctrl-H Excel file", filetypes=[("Excel files", "*.xlsx *.xls")]
    )

    hf_file = filedialog.askopenfilename(
        title="Select HF Excel file", filetypes=[("Excel files", "*.xlsx *.xls")]
    )

    root.destroy()
    return ctrl_h_file, hf_file


def load_data(file_path, column_name=None):
    """
    Load data from Excel file.

    Args:
        file_path (str): Excel file path
        column_name (str): Specified column name, if not exists use the first numeric column

    Returns:
        numpy.ndarray: Data array with NaN removed
    """
    df = pd.read_excel(file_path)

    if column_name and column_name in df.columns:
        return df[column_name].dropna().values
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return df[numeric_cols[0]].dropna().values
    return None


def get_significance_symbol(p_value):
    """
    Return significance symbol based on p-value.

    Args:
        p_value (float): p-value

    Returns:
        str: Significance symbol (***, **, *, ns)
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.

    Use pooled standard deviation to calculate effect size of two groups.

    Args:
        group1, group2 (array-like): Two groups of data

    Returns:
        float: Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size magnitude.

    Args:
        d (float): Cohen's d value

    Returns:
        str: Effect size level (negligible, small, medium, large)
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def plot_density(ctrl_h_data, hf_data, output_dir):
    """
    Draw and save density plot.

    Process:
    1. Calculate statistics for two groups (mean, standard deviation)
    2. Perform independent sample t-test
    3. Calculate Cohen's d effect size
    4. Plot KDE density curve
    5. Add mean line and significance annotation
    6. Add text annotations
    7. Save figure file
    """
    print("[Step 1] Configuring figure parameters...")
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    print("[Step 2] Calculating statistics...")
    ctrl_mean = np.mean(ctrl_h_data)
    ctrl_std = np.std(ctrl_h_data, ddof=1)
    hf_mean = np.mean(hf_data)
    hf_std = np.std(hf_data, ddof=1)

    # Perform t-test and calculate effect size
    t_stat, p_value = stats.ttest_ind(ctrl_h_data, hf_data)
    sig_symbol = get_significance_symbol(p_value)

    cohens_d = calculate_cohens_d(ctrl_h_data, hf_data)
    effect_size_interp = interpret_cohens_d(cohens_d)

    print(f"  Ctrl-H: Mean={ctrl_mean:.4f}, SD={ctrl_std:.4f}, n={len(ctrl_h_data)}")
    print(f"  HF: Mean={hf_mean:.4f}, SD={hf_std:.4f}, n={len(hf_data)}")
    print(f"  t-test: t={t_stat:.4f}, p={p_value:.4e} ({sig_symbol})")
    print(f"  Cohen's d={cohens_d:.4f} ({effect_size_interp})")

    print("[Step 3] Plotting density curve...")
    # Calculate x-axis range
    x_min = min(ctrl_h_data.min(), hf_data.min())
    x_max = max(ctrl_h_data.max(), hf_data.max())
    x_range = np.linspace(
        x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1, 500
    )

    # Ctrl-H Density Curve
    kde_ctrl = stats.gaussian_kde(ctrl_h_data)
    density_ctrl = kde_ctrl(x_range)
    ax.plot(
        x_range,
        density_ctrl,
        color="#3498db",
        linewidth=2,
        label=f"Ctrl-H Mean:{ctrl_mean:.2f}±{ctrl_std:.2f} (n={len(ctrl_h_data)})",
    )
    ax.fill_between(x_range, density_ctrl, alpha=0.3, color="#3498db")

    # HF Density Curve
    kde_hf = stats.gaussian_kde(hf_data)
    density_hf = kde_hf(x_range)
    ax.plot(
        x_range,
        density_hf,
        color="#e74c3c",
        linewidth=2,
        label=f"HF Mean:{hf_mean:.2f}±{hf_std:.2f} (n={len(hf_data)})",
    )
    ax.fill_between(x_range, density_hf, alpha=0.3, color="#e74c3c")

    print("[Step 4] Adding mean line and significance annotation...")
    # Draw mean lines
    y_max = max(density_ctrl.max(), density_hf.max())
    ax.axvline(x=ctrl_mean, color="#2980b9", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(x=hf_mean, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.8)

    # Draw significance bracket
    sig_y = y_max * 1.05
    bracket_drop = y_max * 0.02

    ax.plot(
        [ctrl_mean, ctrl_mean],
        [sig_y - bracket_drop, sig_y],
        color="#2c3e50",
        linewidth=1,
    )
    ax.plot(
        [hf_mean, hf_mean], [sig_y - bracket_drop, sig_y], color="#2c3e50", linewidth=1
    )
    ax.plot([ctrl_mean, hf_mean], [sig_y, sig_y], color="#2c3e50", linewidth=1)

    # Add significance symbol and p-value
    mid_x = (ctrl_mean + hf_mean) / 2
    if p_value < 0.001:
        p_text = f"{sig_symbol}\np<0.001"
    else:
        p_text = f"{sig_symbol}\np={p_value:.3f}"

    effect_text = f"Cohen's d={cohens_d:.2f} "
    ax.text(
        mid_x,
        sig_y * 1.02,
        p_text,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        mid_x,
        sig_y * 1.12,
        effect_text,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#8e44ad",
    )

    print("[Step 5] Adding text annotations...")
    # Top-left title annotation
    ax.text(
        0.02,
        0.98,
        "The Collapse of Physiological Constant\n- Loss of Systolic Ejection Consistency in Heart Failure",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

    # HF Area Annotation
    hf_annotation_x = hf_mean - (x_max - x_min) * 0.15
    hf_annotation_y = y_max * 0.4
    ax.annotate(
        "Disease is Messy\nIncreased Heterogeneity",
        xy=(hf_mean - (x_max - x_min) * 0.05, y_max * 0.5),
        xytext=(hf_annotation_x, hf_annotation_y),
        fontsize=9,
        ha="center",
    )

    # Ctrl-H Area Annotation
    ctrl_annotation_x = ctrl_mean + (x_max - x_min) * 0.08
    ctrl_annotation_y = y_max * 0.7
    ax.annotate(
        "Healthy is Strict\nNarrow variance",
        xy=(ctrl_mean + (x_max - x_min) * 0.02, y_max * 0.85),
        xytext=(ctrl_annotation_x, ctrl_annotation_y),
        fontsize=9,
        ha="center",
    )

    print("[Step 6] Setting figure style...")
    # Set labels and title
    ax.set_xlabel(r"$ER_{{2-100Hz}}$[0-20/0-30%]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Density Plot: Ctrl-H vs HF ($ER_{2-100Hz}$[0-20/0-30%])", fontsize=14)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust y-axis range
    ax.set_ylim(0, y_max * 1.35)

    plt.tight_layout()

    print("[Step 7] [OK] Plot saved...")
    png_path = os.path.join(output_dir, "Fig. 3a_Density.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()


def main():
    """
    Main program entry.

    Process:
    1. Select or load data files
    2. Load Ctrl-H and HF data
    3. Plot density plot
    """
    print("[Step 0] Selecting or loading data files...")
    ctrl_h_file, hf_file = select_files()

    if not ctrl_h_file or not hf_file:
        return

    ctrl_h_data = load_data(ctrl_h_file, DATA_COLUMN)
    hf_data = load_data(hf_file, DATA_COLUMN)

    if ctrl_h_data is None or hf_data is None:
        return

    output_dir = os.path.dirname(ctrl_h_file)
    plot_density(ctrl_h_data, hf_data, output_dir)


if __name__ == "__main__":
    main()
