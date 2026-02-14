"""
==============================================================================
Physiological stability hierarchy plot script

Function: Load Coefficient of Variation (CV) of physiological features from Excel file,
          and plot horizontal bar chart to show feature stability hierarchy by group.

Source: Fig. 2h_CV_per_sample.xlsx
Output: Fig. 2h.png
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(script_dir, "Fig. 2h_CV_per_sample.xlsx")
OUTPUT_PNG = os.path.join(script_dir, "Fig. 2h_Stability_Hierarchy_Bar.png")

# Feature Grouping: Classified into three levels by physiological stability
# Group 1: Diastolic Filling Load (CV ~44%) - Signal close to noise floor
GROUP1_FEATURES = {
    "Diastolic Filling Load": "E_{2-12Hz}[40-100%]",
}

# Group 2: Systolic Kinetic Energy and Stiffness Tone (CV 13%-18%) - Typical wearable sensor precision
GROUP2_FEATURES = {
    "Myocardial Stiffness Tone": "H_{15-80Hz}[37-70%]",
    "Systolic Impulse Energy": "E_{2-100Hz}[0-10%]",
}

# Group 3: Energy Ratio (CV ~4%) - Robust Physiological Constant
GROUP3_FEATURES = {
    "Systolic Burst Ratio": "ER_{2-100Hz}[0-20/0-30%]",
}

# Chart Style Config
COLOR_GROUP1 = "#B0B0B0"  # Light Gray - Group 1
COLOR_GROUP2 = "#4A90D9"  # Blue - Group 2
COLOR_GROUP3 = "#E74C3C"  # Red - Group 3

FIG_WIDTH = 10
FIG_HEIGHT = 6

FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_ANNOTATION = 9

DPI = 300


def load_and_calculate_cv(file_path, feature_dict):
    """
    Load data from Excel file and calculate average CV of features.

    Args:
        file_path (str): Excel file path
        feature_dict (dict): Feature dictionary {Display Name: Column Name}

    Returns:
        dict: {Display Name: Average CV (%)}
    """
    df = pd.read_excel(file_path)

    result = {}
    for display_name, col_name in feature_dict.items():
        if col_name in df.columns:
            # Extract values, convert to percentage
            values = pd.to_numeric(df[col_name], errors="coerce").dropna()
            mean_cv = values.mean() * 100
            result[display_name] = mean_cv
        else:
            result[display_name] = np.nan

    return result


def create_stability_hierarchy_plot():
    """
    Create and save physiological stability hierarchy plot.

    Process:
    1. Load CV data for three groups of features
    2. Organize data and create plot
    3. Set styles (colors, textures, borders)
    4. Add labels, annotations, and background
    5. Save figure file
    """
    print("[Step 1] Loading CV data...")
    cv_group1 = load_and_calculate_cv(DATA_FILE, GROUP1_FEATURES)
    cv_group2 = load_and_calculate_cv(DATA_FILE, GROUP2_FEATURES)
    cv_group3 = load_and_calculate_cv(DATA_FILE, GROUP3_FEATURES)

    print("[Step 2] Organizing feature data...")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Organize data by group: Group1 (Bottom) -> Group2 (Middle) -> Group3 (Top)
    all_features = []
    all_cvs = []
    all_colors = []
    all_groups = []

    for name, cv in cv_group1.items():
        all_features.append(name)
        all_cvs.append(cv)
        all_colors.append(COLOR_GROUP1)
        all_groups.append(1)

    for name, cv in cv_group2.items():
        all_features.append(name)
        all_cvs.append(cv)
        all_colors.append(COLOR_GROUP2)
        all_groups.append(2)

    for name, cv in cv_group3.items():
        all_features.append(name)
        all_cvs.append(cv)
        all_colors.append(COLOR_GROUP3)
        all_groups.append(3)

    print("[Step 3] Creating plot...")
    y_positions = np.arange(len(all_features))

    bars = ax.barh(
        y_positions,
        all_cvs,
        color=all_colors,
        height=0.55,
        edgecolor="black",
        linewidth=0.8,
    )

    print("[Step 4] Setting styles...")
    # Set different styles for different groups
    for i, (bar, group) in enumerate(zip(bars, all_groups)):
        if group == 1:
            # Group 1: Dashed border and diagonal hatch
            bar.set_edgecolor("#888888")
            bar.set_linewidth(1.5)

        elif group == 3:
            # Group 3: Bold red border
            bar.set_linewidth(2.5)
            bar.set_edgecolor("#C0392B")

    ax.set_yticks(y_positions)

    ax.set_ylabel("Features", fontsize=FONT_SIZE_LABEL, fontweight="medium")
    ax.set_xlabel(
        "Coefficient of Variation (CV, %)",
        fontsize=FONT_SIZE_LABEL,
        fontweight="medium",
    )
    max_cv = max(all_cvs)
    ax.set_xlim(0, max_cv * 1.6)

    print("[Step 5] Adding labels and annotations...")
    # Add value labels and special annotations
    for i, (bar, cv, group, feature) in enumerate(
        zip(bars, all_cvs, all_groups, all_features)
    ):
        ax.text(
            cv + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{cv:.1f}%",
            va="center",
            fontsize=FONT_SIZE_ANNOTATION + 1,
            fontweight="bold",
        )

        if group == 1:
            ax.text(
                0,
                bar.get_y() + bar.get_height() / 2,
                "    Near Noise Floor Variation (Silent Zone)",
                va="center",
                ha="left",
                fontsize=FONT_SIZE_ANNOTATION + 4,
                style="italic",
                color="black",
            )

        if group == 3:
            ax.text(
                cv + 5,
                bar.get_y() + bar.get_height() / 2,
                " Robust Physiological Constant",
                va="center",
                fontsize=FONT_SIZE_ANNOTATION + 1,
                fontweight="bold",
                color=COLOR_GROUP3,
            )

    # Calculate Y-range for each group for background color
    group1_y = [i for i, g in enumerate(all_groups) if g == 1]
    group2_y = [i for i, g in enumerate(all_groups) if g == 2]
    group3_y = [i for i, g in enumerate(all_groups) if g == 3]

    # Add group background blocks
    if group1_y:
        ax.axhspan(
            min(group1_y) - 0.4,
            max(group1_y) + 0.4,
            facecolor="#F5F5F5",
            alpha=0.5,
            zorder=0,
        )
    if group2_y:
        ax.axhspan(
            min(group2_y) - 0.4,
            max(group2_y) + 0.4,
            facecolor="#E8F4FD",
            alpha=0.5,
            zorder=0,
        )
    if group3_y:
        ax.axhspan(
            min(group3_y) - 0.4,
            max(group3_y) + 0.4,
            facecolor="#FDEDEC",
            alpha=0.5,
            zorder=0,
        )

    # Add group labels (right side)
    x_label = max_cv * 1.35

    if group1_y:
        y_mid = np.mean(group1_y)
        ax.text(
            x_label,
            y_mid,
            "Diastolic Filling Load\n($E_{2-12Hz}$[40-100%])",
            va="center",
            ha="left",
            fontsize=FONT_SIZE_ANNOTATION,
            color="#666666",
            linespacing=1.2,
        )

    if group2_y:
        y_mid = np.mean(group2_y)
        ax.text(
            x_label,
            y_mid,
            "Systolic Impulse Energy\n($E_{2-100Hz}$[0-10%])\n\n\n\nMyocardial Stiffness Tone\n($H_{15-80Hz}$[37-70%])",
            va="center",
            ha="left",
            fontsize=FONT_SIZE_ANNOTATION,
            color=COLOR_GROUP2,
            linespacing=1.2,
        )

    if group3_y:
        y_mid = np.mean(group3_y)
        ax.text(
            x_label,
            y_mid,
            "Systolic Burst Ratio\n($ER_{2-100Hz}$[0-20/0-30%])",
            va="center",
            ha="left",
            fontsize=FONT_SIZE_ANNOTATION,
            color=COLOR_GROUP3,
            fontweight="bold",
            linespacing=1.2,
        )

    # Add horizontal separators
    if group1_y and group2_y:
        ax.axhline(
            y=max(group1_y) + 0.5,
            color="#AAAAAA",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
        )
    if group2_y and group3_y:
        ax.axhline(
            y=max(group2_y) + 0.5,
            color="#AAAAAA",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_title(
        "Feature Variation Hierarchy (n=24,3 Times @ 3 Weeks)",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=20,
    )

    ax.xaxis.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    print("[Step 6] [OK] Plot saved: {}".format(OUTPUT_PNG))
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")

    plt.show()

    return fig, ax


if __name__ == "__main__":
    fig, ax = create_stability_hierarchy_plot()
