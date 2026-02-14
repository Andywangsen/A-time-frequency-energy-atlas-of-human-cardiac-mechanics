"""
==============================================================================
Figure 3-d: Comparative Violin Plot Analysis

Generates violin plots for three features, comparing distributions between Healthy Control (Ctrl-H)
and Heart Failure (HF) groups. Includes box plot skeleton, scatter data, significance annotations,
and statistical information.

Key Features:
- Load Ctrl-H and HF data from Excel files
- Compare distributions of three cardiac features
- Calculate t-test p-values and show significance annotations
- Display mean ± standard deviation for each group
- Generate publication-quality violin plots
==============================================================================
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import os

np.random.seed(42)

FEATURE_1 = "ER_{2-100Hz}[0-20/0-30%]"
FEATURE_2 = "H_{2-15Hz}[40-100%]"
FEATURE_3 = "H_{15-80Hz}[37-70%]"

FEATURE_LABELS = {
    FEATURE_1: r"$ER_{{2-100Hz}}$[0-20/0-30%]",
    FEATURE_2: r"H$_{{2-15Hz}}$[40-100%]",
    FEATURE_3: r"H$_{{15-80Hz}}$[37-70%]",
}

OUTPUT_VIOLIN_FILENAME = "Fig. 3d_Feature_Comparison_Violin"

VIOLIN_ALPHA = 0.3
VIOLIN_WIDTH = 0.7
BOX_WIDTH = 0.15
STRIP_ALPHA = 0.4
STRIP_SIZE = 2
VIOLIN_PALETTE = ["#56b053", "#e74c3c"]
SHOW_SIGNIFICANCE = True
SHOW_MEAN_STD = True

FIG_WIDTH = 12
FIG_HEIGHT = 4


def select_files():
    """
    Automatically locate Excel data files.

    Returns:
        tuple: (ctrl_h_file, hf_file) paths or None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ctrl_h_file = os.path.join(script_dir, "Fig. 3c,d_Ctrl-H.xlsx")
    hf_file = os.path.join(script_dir, "Fig. 3c,d_HF.xlsx")

    if not os.path.exists(ctrl_h_file):
        ctrl_h_file = None
    else:
        pass

    if not os.path.exists(hf_file):
        hf_file = None
    else:
        pass

    return ctrl_h_file, hf_file


def load_data(file_path):
    """
    Load data from Excel file.

    Args:
        file_path (str): Path to Excel file

    Returns:
        pd.DataFrame: Loaded data
    """
    df = pd.read_excel(file_path)
    return df


def get_features(df, features):
    """
    Extract feature columns from dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list): Requested feature names

    Returns:
        list: List of available feature names
    """
    available = [f for f in features if f in df.columns]
    if len(available) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    return available


def get_significance_symbol(p_value):
    """
    Return significance symbol based on p-value.

    Args:
        p_value (float): p-value from t-test

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


def plot_violin(ctrl_h_df, hf_df, features, output_dir):
    """
    Generate three-feature comparative violin plot.

    Prepare data and plot violin charts, including box plot skeleton, scatter data, and statistics.

    Args:
        ctrl_h_df (pd.DataFrame): Healthy control group data
        hf_df (pd.DataFrame): Heart failure group data
        features (list): Feature names to analyze
        output_dir (str): Directory for output PNG file
    """

    def calc_whisker_bounds(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr

        lower_bound = data[data >= lower_whisker].min()
        upper_bound = data[data <= upper_whisker].max()

        return lower_bound, upper_bound

    def plot_single_violin(feature, feature_idx, ctrl_h_df, hf_df, output_dir):
        fig, ax = plt.subplots()

        ctrl_data = ctrl_h_df[feature].dropna().values
        hf_data = hf_df[feature].dropna().values

        df = pd.DataFrame(
            {
                "Group": ["Ctrl-H"] * len(ctrl_data) + ["HF"] * len(hf_data),
                "Value": np.concatenate([ctrl_data, hf_data]),
            }
        )

        group_order = ["Ctrl-H", "HF"]

        sns.violinplot(
            x="Group",
            y="Value",
            data=df,
            order=group_order,
            palette=VIOLIN_PALETTE,
            inner=None,
            linewidth=0,
            saturation=0.8,
            width=VIOLIN_WIDTH,
            cut=0,
            ax=ax,
        )

        whisker_bounds = [calc_whisker_bounds(ctrl_data), calc_whisker_bounds(hf_data)]
        for j, collection in enumerate(ax.collections):
            collection.set_alpha(VIOLIN_ALPHA)
            if j < len(whisker_bounds):
                whisker_min, whisker_max = whisker_bounds[j]
                if hasattr(collection, "get_paths"):
                    for path in collection.get_paths():
                        vertices = path.vertices.copy()
                        vertices[:, 1] = np.clip(
                            vertices[:, 1], whisker_min, whisker_max
                        )
                        path.vertices = vertices

        sns.boxplot(
            x="Group",
            y="Value",
            data=df,
            order=group_order,
            width=BOX_WIDTH,
            boxprops={"facecolor": "none", "edgecolor": "k", "linewidth": 0.5},
            whiskerprops={"linewidth": 0.5},
            capprops={"linewidth": 0.5},
            medianprops={"linewidth": 1, "color": "k"},
            showfliers=False,
            ax=ax,
        )

        sns.stripplot(
            x="Group",
            y="Value",
            data=df,
            order=group_order,
            color="black",
            alpha=STRIP_ALPHA,
            size=STRIP_SIZE,
            jitter=True,
            ax=ax,
        )

        sns.despine(ax=ax, trim=False)

        y_min = df["Value"].min()
        y_max = df["Value"].max()
        y_range = y_max - y_min

        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))

        if SHOW_SIGNIFICANCE:
            t_stat, p_value = stats.ttest_ind(ctrl_data, hf_data)
            sig_symbol = get_significance_symbol(p_value)

            x1, x2 = 0, 1
            y = y_max + y_range * 0.12
            h = y_range * 0.03
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.5, c="k")
            ax.text(
                (x1 + x2) * 0.5,
                y + h * 1.5,
                sig_symbol,
                ha="center",
                va="bottom",
                color="k",
                fontweight="bold",
            )

        if SHOW_MEAN_STD:
            ctrl_mean = np.mean(ctrl_data)
            ctrl_std = np.std(ctrl_data)
            hf_mean = np.mean(hf_data)
            hf_std = np.std(hf_data)

            ax.text(
                0,
                y_min - y_range * 0.05,
                f"{ctrl_mean:.2f}±{ctrl_std:.2f}",
                ha="center",
                va="top",
                color=VIOLIN_PALETTE[0],
            )
            ax.text(
                1,
                y_min - y_range * 0.05,
                f"{hf_mean:.2f}±{hf_std:.2f}",
                ha="center",
                va="top",
                color=VIOLIN_PALETTE[1],
            )

        ax.set_ylim(y_min - y_range * 0.18, y_max + y_range * 0.30)

        ax.set_xlim(-0.6, 1.6)

        plt.tight_layout()

        plt.close(fig)
        return fig

    print("[1/4] Loading and preparing data...")
    for i, feature in enumerate(features):
        plot_single_violin(feature, i, ctrl_h_df, hf_df, output_dir)

    print("[2/4] Creating merged violin plot...")
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))

    for i, feature in enumerate(features):
        ctrl_data = ctrl_h_df[feature].dropna().values
        hf_data = hf_df[feature].dropna().values

        df = pd.DataFrame(
            {
                "Group": ["Ctrl-H"] * len(ctrl_data) + ["HF"] * len(hf_data),
                "Value": np.concatenate([ctrl_data, hf_data]),
            }
        )

        ax = axes[i]
        group_order = ["Ctrl-H", "HF"]

        sns.violinplot(
            x="Group",
            y="Value",
            data=df,
            order=group_order,
            palette=VIOLIN_PALETTE,
            inner=None,
            linewidth=0,
            saturation=0.8,
            width=VIOLIN_WIDTH,
            cut=0,
            ax=ax,
        )

        whisker_bounds = [calc_whisker_bounds(ctrl_data), calc_whisker_bounds(hf_data)]
        for j, collection in enumerate(ax.collections):
            collection.set_alpha(VIOLIN_ALPHA)
            if j < len(whisker_bounds):
                whisker_min, whisker_max = whisker_bounds[j]
                if hasattr(collection, "get_paths"):
                    for path in collection.get_paths():
                        vertices = path.vertices.copy()
                        vertices[:, 1] = np.clip(
                            vertices[:, 1], whisker_min, whisker_max
                        )
                        path.vertices = vertices

        sns.boxplot(
            x="Group",
            y="Value",
            data=df,
            order=group_order,
            width=BOX_WIDTH,
            boxprops={"facecolor": "none", "edgecolor": "k", "linewidth": 0.5},
            whiskerprops={"linewidth": 0.5},
            capprops={"linewidth": 0.5},
            medianprops={"linewidth": 1, "color": "k"},
            showfliers=False,
            ax=ax,
        )

        sns.stripplot(
            x="Group",
            y="Value",
            data=df,
            order=group_order,
            color="black",
            alpha=STRIP_ALPHA,
            size=STRIP_SIZE,
            jitter=True,
            ax=ax,
        )

        sns.despine(ax=ax, trim=False)

        y_min = df["Value"].min()
        y_max = df["Value"].max()
        y_range = y_max - y_min

        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))

        if SHOW_SIGNIFICANCE:
            t_stat, p_value = stats.ttest_ind(ctrl_data, hf_data)
            sig_symbol = get_significance_symbol(p_value)

            x1, x2 = 0, 1
            y = y_max + y_range * 0.12
            h = y_range * 0.03
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.5, c="k")
            ax.text(
                (x1 + x2) * 0.5,
                y + h * 1.5,
                sig_symbol,
                ha="center",
                va="bottom",
                color="k",
                fontweight="bold",
            )

        if SHOW_MEAN_STD:
            ctrl_mean = np.mean(ctrl_data)
            ctrl_std = np.std(ctrl_data)
            hf_mean = np.mean(hf_data)
            hf_std = np.std(hf_data)

            ax.text(
                0,
                y_min - y_range * 0.05,
                f"{ctrl_mean:.2f}±{ctrl_std:.2f}",
                ha="center",
                va="top",
                color=VIOLIN_PALETTE[0],
            )
            ax.text(
                1,
                y_min - y_range * 0.05,
                f"{hf_mean:.2f}±{hf_std:.2f}",
                ha="center",
                va="top",
                color=VIOLIN_PALETTE[1],
            )

        ax.set_ylim(y_min - y_range * 0.18, y_max + y_range * 0.30)
        ax.set_xlim(-0.6, 1.6)

    plt.suptitle("Feature Comparison: Ctrl-H vs HF", y=1.02)
    plt.tight_layout()

    print("[3/4] Saving PNG file...")
    png_path = os.path.join(output_dir, f"{OUTPUT_VIOLIN_FILENAME}.png")

    plt.savefig(png_path, bbox_inches="tight", facecolor="white")

    print(f"  Saved: {png_path}")

    print("[4/4] Showing plot...")
    plt.show()


def main():
    """
    Main program entry.

    Orchestrates complete violin plot analysis pipeline:
    1. Locate and load data files
    2. Extract features
    3. Generate comparative violin plot
    """
    print("=" * 60)
    print("Violin Plot Comparative Analysis - Figure 3-d")
    print("=" * 60)

    ctrl_h_file, hf_file = select_files()

    if not ctrl_h_file or not hf_file:
        print("Error: Required data files not found")
        return

    print("\nLoading data files...")
    ctrl_h_df = load_data(ctrl_h_file)
    hf_df = load_data(hf_file)
    print(f"  Ctrl-H: {ctrl_h_df.shape[0]} samples, {ctrl_h_df.shape[1]} features")
    print(f"  HF: {hf_df.shape[0]} samples, {hf_df.shape[1]} features")

    features = get_features(ctrl_h_df, [FEATURE_1, FEATURE_2, FEATURE_3])

    if len(features) < 3:
        print("Error: Insufficient features (need at least 3)")
        return

    print(f"\nSelected features: {features}")

    output_dir = os.path.dirname(ctrl_h_file)

    print("\nGenerating violin plot...")
    plot_violin(ctrl_h_df, hf_df, features, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
