# -*- coding: utf-8 -*-
"""
==============================================================================
Figure 4-a: Nature Style ER200300 Violin Plot

Generates publication-quality violin plots demonstrating the progressive collapse of Energy Ratio (ER)
across different clinical states.
Includes box plot skeleton, scatter data, physiological homeostasis zone, significance annotations,
and detailed statistical reports.

Key Features:
- Load data for different clinical groups from multiple Excel files
- Draw Nature-style violin plots (with box plots and scatter points)
- Display physiological homeostasis zone (green background band)
- Perform Mann-Whitney U test for between-group comparisons
- Display mean ± standard deviation and sample size
- Generate detailed statistical analysis report
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tkinter import Tk, filedialog

np.random.seed(42)

DPI = 300

VIOLIN_WIDTH = 0.8
VIOLIN_ALPHA = 0.6
VIOLIN_LINEWIDTH = 1.2

BOX_WIDTH = 0.15
BOX_LINEWIDTH = 1.2
BOX_COLOR = "black"
MEDIAN_LINEWIDTH = 2

STRIP_SIZE = 3
STRIP_ALPHA = 0.5
STRIP_COLOR = "gray"

PALETTE = ["#4DAF4A", "#FFFF33", "#FF7F00", "#E41A1C", "#984EA3", "#377EB8"]

TITLE = 'The "Physiological Staircase": Progressive Collapse of Energy Ratio Across Clinical States'
Y_LABEL = "$ER_{{2-100Hz}}$[0-20/0-30%]"

SHOW_HOMEOSTASIS_ZONE = True
HOMEOSTASIS_YMIN = 0.82
HOMEOSTASIS_YMAX = 0.88
HOMEOSTASIS_COLOR = "#90EE90"
HOMEOSTASIS_ALPHA = 0.3
HOMEOSTASIS_LABEL = "Physiological\nHomeostasis\nZone"
HOMEOSTASIS_LABEL_COLOR = "green"

HOMEOSTASIS_REF_GROUP = 0
HOMEOSTASIS_REF_METHOD = "mean_std"
HOMEOSTASIS_REF_FACTOR = 1.0

SHOW_GRID = True
GRID_ALPHA = 0.3
GRID_LINESTYLE = "-"
GRID_COLOR = "gray"

SHOW_SIGNIFICANCE = True
SIG_COMPARISONS = [
    ("CVD-LRRCV", "HF-LRRCV", "Failure-lnduced Collapse"),
    ("CVD-HRRCV", "HF-HRRCV", "Failure-lnduced Collapse"),
    ("Ctrl-H", "CVD-LRRCV", ""),
    ("CVD-LRRCV", "CVD-HRRCV", "Rhythm-Induced Attenuation"),
    ("CVD-HRRCV", "HF-LRRCV", ""),
    ("HF-LRRCV", "HF-HRRCV", ""),
]

Y_MIN = None
Y_MAX = None

SORT_BY = None
SORT_ASCENDING = False

MANUAL_ORDER = ["Ctrl-H", "CVD-LRRCV", "CVD-HRRCV", "HF-LRRCV", "HF-HRRCV"]

VIOLIN_CUT = True

OUTPUT_PREFIX = "Fig. 4a_Energy_Ratio_Staircase_Violin"

VALUE_COLUMN = "ER_{2-100Hz}[0-20/0-30%]"


def select_xlsx_files():
    """
    Select or automatically load Excel data files.

    Tries to automatically load files based on MANUAL_ORDER first, otherwise prompts file selection dialog.

    Returns:
        list: List of selected Excel file paths
    """
    if MANUAL_ORDER:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        auto_files = []
        missing_files = []

        for group_name in MANUAL_ORDER:
            file_path = os.path.join(script_dir, f"Fig. 4a_{group_name}.xlsx")
            if os.path.exists(file_path):
                auto_files.append(file_path)
            else:
                missing_files.append(f"Fig. 4a_{group_name}.xlsx")

        if not missing_files:
            return auto_files
        else:
            pass

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_paths = filedialog.askopenfilenames(
        title="Select xlsx files",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
        initialdir=os.path.dirname(os.path.abspath(__file__)),
    )

    root.destroy()
    return list(file_paths)


def load_data(file_paths):
    """
    Load and integrate data from all Excel files.

    Args:
        file_paths (list): List of Excel file paths

    Returns:
        tuple: (Combined DataFrame, Group order list)
    """
    all_data = []
    group_order = []

    for fp in file_paths:
        df = pd.read_excel(fp)
        group_name = os.path.splitext(os.path.basename(fp))[0].replace("Fig. 4a_", "")
        group_order.append(group_name)

        if VALUE_COLUMN and VALUE_COLUMN in df.columns:
            values = df[VALUE_COLUMN].dropna()
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                values = df[numeric_cols[0]].dropna()
            else:
                continue

        temp_df = pd.DataFrame({"Value": values, "Group": group_name})
        all_data.append(temp_df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined, group_order
    return None, []


def calc_pvalue(group1_data, group2_data):
    """
    Calculate p-value between two groups.

    Uses Mann-Whitney U test (non-parametric) for comparison.

    Args:
        group1_data: Data for group 1
        group2_data: Data for group 2

    Returns:
        float: p-value
    """
    stat, pval = stats.mannwhitneyu(group1_data, group2_data, alternative="two-sided")
    return pval


def pval_to_stars(pval):
    """
    Convert p-value to significance symbol.

    Args:
        pval (float): p-value

    Returns:
        str: Significance symbol (***, **, *, ns)
    """
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return "ns"


def calc_whisker_bounds(group_data):
    """
    Calculate whisker bounds for box plot.

    Uses standard box plot rule: Q1-1.5*IQR to Q3+1.5*IQR

    Args:
        group_data: Group data

    Returns:
        tuple: (Lower bound, Upper bound)
    """
    q1 = group_data.quantile(0.25)
    q3 = group_data.quantile(0.75)
    iqr = q3 - q1

    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr

    lower_bound = group_data[group_data >= lower_whisker].min()
    upper_bound = group_data[group_data <= upper_whisker].max()

    return lower_bound, upper_bound


def plot_violin_nature(data, group_order, sig_comparisons=None):
    """
    Generate Nature-style violin plot.

    Draws publication-quality chart including violin plot, box plot, scatter points,
    physiological homeostasis zone, and significance annotations.

    Args:
        data (pd.DataFrame): DataFrame containing 'Value' and 'Group' columns
        group_order (list): Order of groups
        sig_comparisons (list): Significance comparison list [(Group1, Group2, Label), ...]

    Returns:
        matplotlib.figure.Figure: Generated figure object
    """
    print(f"Plotting group order: {group_order}")

    fig, ax = plt.subplots(figsize=(11, 5))

    n_groups = len(group_order)
    colors = PALETTE[:n_groups]

    data_min = data["Value"].min()
    data_max = data["Value"].max()
    data_range = data_max - data_min

    y_min = Y_MIN if Y_MIN is not None else data_min - data_range * 0.1
    y_max = Y_MAX if Y_MAX is not None else data_max + data_range * 0.6

    ax.set_ylim(y_min, y_max)

    if SHOW_GRID:
        ax.yaxis.grid(
            True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, color=GRID_COLOR
        )
        ax.set_axisbelow(True)

    if SHOW_HOMEOSTASIS_ZONE:
        if HOMEOSTASIS_REF_METHOD == "fixed":
            h_min, h_max = HOMEOSTASIS_YMIN, HOMEOSTASIS_YMAX
        elif HOMEOSTASIS_REF_GROUP is not None and HOMEOSTASIS_REF_GROUP < n_groups:
            ref_group = group_order[HOMEOSTASIS_REF_GROUP]
            ref_data = data[data["Group"] == ref_group]["Value"]

            if HOMEOSTASIS_REF_METHOD == "mean_std":
                ref_mean = ref_data.mean()
                ref_std = ref_data.std()
                h_min = ref_mean - HOMEOSTASIS_REF_FACTOR * ref_std
                h_max = ref_mean + HOMEOSTASIS_REF_FACTOR * ref_std
            elif HOMEOSTASIS_REF_METHOD == "iqr":
                q1 = ref_data.quantile(0.25)
                q3 = ref_data.quantile(0.75)
                iqr = q3 - q1
                h_min = q1 - HOMEOSTASIS_REF_FACTOR * iqr
                h_max = q3 + HOMEOSTASIS_REF_FACTOR * iqr
            else:
                h_min, h_max = HOMEOSTASIS_YMIN, HOMEOSTASIS_YMAX
        else:
            h_min, h_max = HOMEOSTASIS_YMIN, HOMEOSTASIS_YMAX

        ax.axhspan(
            h_min, h_max, color=HOMEOSTASIS_COLOR, alpha=HOMEOSTASIS_ALPHA, zorder=0
        )
        ax.text(
            n_groups - 0.3,
            (h_min + h_max) / 2,
            HOMEOSTASIS_LABEL,
            color=HOMEOSTASIS_LABEL_COLOR,
            ha="left",
            va="center",
        )

    sns.violinplot(
        x="Group",
        y="Value",
        data=data,
        order=group_order,
        palette=colors,
        inner=None,
        linewidth=VIOLIN_LINEWIDTH,
        saturation=0.9,
        ax=ax,
        width=VIOLIN_WIDTH,
        cut=0,
    )

    for violin_part in ax.collections:
        violin_part.set_alpha(VIOLIN_ALPHA)

    if VIOLIN_CUT:
        for i, (collection, group) in enumerate(zip(ax.collections, group_order)):
            group_data = data[data["Group"] == group]["Value"]
            whisker_min, whisker_max = calc_whisker_bounds(group_data)

            if hasattr(collection, "get_paths"):
                for path in collection.get_paths():
                    vertices = path.vertices.copy()
                    vertices[:, 1] = np.clip(vertices[:, 1], whisker_min, whisker_max)
                    path.vertices = vertices

    sns.boxplot(
        x="Group",
        y="Value",
        data=data,
        order=group_order,
        width=BOX_WIDTH,
        boxprops={
            "facecolor": "white",
            "edgecolor": BOX_COLOR,
            "linewidth": BOX_LINEWIDTH,
        },
        whiskerprops={"color": BOX_COLOR, "linewidth": BOX_LINEWIDTH},
        capprops={"color": BOX_COLOR, "linewidth": BOX_LINEWIDTH},
        medianprops={"color": BOX_COLOR, "linewidth": MEDIAN_LINEWIDTH},
        showfliers=False,
        ax=ax,
    )

    sns.stripplot(
        x="Group",
        y="Value",
        data=data,
        order=group_order,
        color=STRIP_COLOR,
        size=STRIP_SIZE,
        alpha=STRIP_ALPHA,
        jitter=True,
        ax=ax,
    )

    for i, group in enumerate(group_order):
        group_data = data[data["Group"] == group]["Value"]
        mean_val = group_data.mean()
        std_val = group_data.std()
        n = len(group_data)

        y_pos = min(group_data.max() + data_range * 0.03, y_max - data_range * 0.15)
        ax.text(
            i,
            y_pos,
            f"{mean_val:.2f} ± {std_val:.2f}",
            ha="center",
            va="bottom",
            color="black",
        )

    if SHOW_SIGNIFICANCE and sig_comparisons:
        y_start = y_max - data_range * 0.05
        y_step = data_range * 0.1

        for idx, (g1_name, g2_name, label) in enumerate(sig_comparisons):
            if g1_name in group_order and g2_name in group_order:
                g1 = group_order.index(g1_name)
                g2 = group_order.index(g2_name)

                group1_data = data[data["Group"] == g1_name]["Value"]
                group2_data = data[data["Group"] == g2_name]["Value"]

                pval = calc_pvalue(group1_data, group2_data)
                stars = pval_to_stars(pval)

                y = y_start - idx * y_step
                h = data_range * 0.02

                ax.plot([g1, g1, g2, g2], [y - h, y, y, y - h], lw=1, c="k")

                mid_x = (g1 + g2) / 2

                if label:
                    ax.text(
                        mid_x,
                        y + data_range * 0.01,
                        label,
                        ha="center",
                        va="bottom",
                    )
                    text_y_offset = 0.04
                else:
                    text_y_offset = 0.01

                ax.text(
                    mid_x,
                    y + data_range * text_y_offset,
                    stars,
                    ha="center",
                    va="bottom",
                )

    sns.despine(top=True, right=True, left=False, bottom=False)
    ax.set_xlabel("")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(TITLE, pad=20)

    new_labels = []
    for group in group_order:
        n = len(data[data["Group"] == group])
        new_labels.append(f"{group}\n(n={n})")

    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(new_labels)

    ax.set_xlim(-0.6, n_groups - 0.25)

    plt.xticks(rotation=0)

    fig.tight_layout()
    return fig


def save_figure(fig, output_dir):
    """
    Save figure as PNG file.

    Args:
        fig (matplotlib.figure.Figure): Figure object
        output_dir (str): Output directory
    """
    png_path = os.path.join(output_dir, f"{OUTPUT_PREFIX}.png")

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")

    print(f"Saved: {png_path}")


def print_detailed_analysis_report(data, group_order, sig_comparisons=None):
    """
    Print detailed statistical analysis report.

    Includes analysis methods, descriptive statistics, and pairwise comparisons.

    Args:
        data (pd.DataFrame): Dataframe
        group_order (list): Order of groups
        sig_comparisons (list): Significance comparison list
    """
    print("\n" + "=" * 80)
    print("Analysis Report")
    print("=" * 80)

    print("\n1. Analysis Methods")
    print("-" * 30)
    print(
        "   - Data Representation: Violin plot with Box plot (Median, Quartiles) and Scatter"
    )
    print("   - Descriptive Statistics: Mean ± SD, Median [Q1, Q3], Sample Size (n)")
    print(
        "   - Statistical Test: Mann-Whitney U test (two-sided) for between-group comparison"
    )
    print("   - Significance Level: * p<0.05, ** p<0.01, *** p<0.001, ns p>=0.05")

    print("\n2. Descriptive Statistics")
    print("-" * 30)
    stats_list = []
    for group in group_order:
        g_data = data[data["Group"] == group]["Value"]
        desc = g_data.describe()
        stats_list.append(
            {
                "Group": group,
                "N": int(desc["count"]),
                "Mean": desc["mean"],
                "SD": desc["std"],
                "Median": desc["50%"],
                "Q1": desc["25%"],
                "Q3": desc["75%"],
                "Min": desc["min"],
                "Max": desc["max"],
            }
        )

    stats_df = pd.DataFrame(stats_list)
    print(
        f"{'Group':<15} {'N':<5} {'Mean ± SD':<20} {'Median [Q1, Q3]':<25} {'Min - Max':<20}"
    )
    print("-" * 90)
    for _, row in stats_df.iterrows():
        mean_sd = f"{row['Mean']:.4f} ± {row['SD']:.4f}"
        median_iqr = f"{row['Median']:.4f} [{row['Q1']:.4f}, {row['Q3']:.4f}]"
        min_max = f"{row['Min']:.4f} - {row['Max']:.4f}"
        print(
            f"{row['Group']:<15} {row['N']:<5} {mean_sd:<20} {median_iqr:<25} {min_max:<20}"
        )

    print("\n3. Pairwise Comparison (Mann-Whitney U Test)")
    print("-" * 30)
    if sig_comparisons:
        print(f"{'Comparison':<30} {'P-Value':<15} {'Significance':<10}")
        print("-" * 60)

        for g1_name, g2_name, label in sig_comparisons:
            if g1_name in group_order and g2_name in group_order:
                group1_data = data[data["Group"] == g1_name]["Value"]
                group2_data = data[data["Group"] == g2_name]["Value"]

                pval = calc_pvalue(group1_data, group2_data)
                stars = pval_to_stars(pval)

                comp_str = f"{g1_name} vs {g2_name}"
                if label:
                    comp_str += f" ({label})"
                print(f"{comp_str:<30} {pval:.6f}        {stars:<10}")
    else:
        print("No comparisons configured")

    print("\n" + "=" * 80)


def main():
    """
    Main program entry.

    """
    print("=" * 60)
    print("Figure 4-a: ER200300 Violin Plot (Nature Style)")
    print("=" * 60)

    print("\n[1/5] Selecting data files...")
    file_paths = select_xlsx_files()
    if not file_paths:
        print("No files selected, exiting")
        return

    print(f"Selected {len(file_paths)} files:")
    for fp in file_paths:
        print(f"  - {os.path.basename(fp)}")

    print("\n[2/5] Loading data...")
    data, group_order = load_data(file_paths)
    if data is None or data.empty:
        print("Data load failed, exiting")
        return

    print(f"Data load complete, total {len(data)} records")

    print("\n[3/5] Preparing group order...")
    final_group_order = MANUAL_ORDER if MANUAL_ORDER else group_order
    final_group_order = [g for g in final_group_order if g in data["Group"].values]

    existing_groups = set(final_group_order)
    for g in group_order:
        if g not in existing_groups:
            final_group_order.append(g)

    final_comparisons = SIG_COMPARISONS

    print(f"Final group order: {final_group_order}")

    print("\n[4/5] Generating violin plot...")
    print("Significance test results:")
    fig = plot_violin_nature(data, final_group_order, final_comparisons)

    print("\n[5/5] Saving results...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_figure(fig, output_dir)

    print_detailed_analysis_report(data, final_group_order, final_comparisons)

    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
