"""
==============================================================================
Figure 5-c: Multi-Group Triangular Radar Chart

Generates publication-quality radar charts comparing features of different clinical groups across three physical dimensions.
Supports automatic loading of multiple Excel files, Z-score normalization, and feature direction adjustment.

Key Features:
- Load data for different clinical groups from multiple Excel files
- Automatically find matching feature columns
- Apply Z-score normalization
- Support feature direction inversion
- Draw polar coordinate radar charts
- Generate publication-quality PNG images
==============================================================================
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_FILENAME = "Fig. 5c_Triangular_Radar_Chart"

FEATURES_CONFIG = [
    ("E_{2-100Hz}[0-10%]", "Systolic Initial Impulse ", "E$_{2-100Hz}$[0-10%]", +1),
    ("E_{2-12Hz}[40-100%]", "Diastolic Filling Load", "E$_{2-12Hz}$[40-100%]", +1),
    (
        "H_{10-80Hz}[39-70%]",
        "Myocardial Stiffness Tone",
        "H$_{10-80Hz}$[39-70%]",
        -1,
    ),
]

COLORS = [
    "#4DAF4A",
    "#984EA3",
    "#E41A1C",
    "#9B59B6",
    "#F39C12",
    "#1ABC9C",
    "#E91E63",
    "#795548",
]

ALPHA_FILL = 0.25
ALPHA_LINE = 0.8
LINE_WIDTH = 1.5

USE_ZSCORE = True


def find_matching_column(df, feature_pattern):
    """
    Find matching column name.

    Args:
        df (pd.DataFrame): Dataframe
        feature_pattern (str): Feature column name pattern

    Returns:
        str: Matching column name, or None if not found
    """
    for col in df.columns:
        if feature_pattern in col:
            return col
    return None


def get_group_name_from_filename(filepath):
    """
    Extract group name from file path.

    Args:
        filepath (str): File path

    Returns:
        str: Group name (without extension)
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    return name


def calculate_zscore(value, reference_mean, reference_std):
    """
    Calculate Z-score normalization value.

    Args:
        value (float): Raw value
        reference_mean (float): Reference mean
        reference_std (float): Reference standard deviation

    Returns:
        float: Z-score value
    """
    return (value - reference_mean) / (reference_std + 1e-10)


def generate_axis_labels(features_config):
    """
    Generate radar chart axis labels based on feature configuration.

    Args:
        features_config (list): Feature configuration list

    Returns:
        list: List of axis labels
    """
    labels = []
    for feature_col, label, description, sign in features_config:
        labels.append(f"{label}\n({description})")
    return labels


def radar_chart(ax, values_list, labels, colors, group_names, title):
    """
    Draw polar coordinate radar chart.

    Args:
        ax: matplotlib polar axes object
        values_list (list): List of data values for each group
        labels (list): List of axis labels
        colors (list): List of colors
        group_names (list): List of group names
        title (str): Chart title
    """
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    for values, color, name in zip(values_list, colors, group_names):
        values_closed = values.tolist() + values.tolist()[:1]
        ax.plot(
            angles,
            values_closed,
            "o-",
            linewidth=LINE_WIDTH,
            color=color,
            label=name[8:],
            alpha=ALPHA_LINE,
        )
        ax.fill(angles, values_closed, color=color, alpha=ALPHA_FILL)

    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))


# ==================== Main Program ====================
def main():
    """
    Main program entry.

    Orchestrates the complete radar chart analysis pipeline:
    1. Automatically load Excel data files
    2. Extract feature data
    3. Apply Z-score normalization
    4. Adjust feature directions
    5. Generate radar chart
    6. Save results
    """
    print("=" * 60)
    print("=" * 60)
    print("Figure 5-c: Multi-Group Triangular Radar Chart")
    print("=" * 60)

    print("\n[1/6] Configuring features...")
    print(f"Current config: {len(FEATURES_CONFIG)} features:")
    for i, (col, label, desc, sign) in enumerate(FEATURES_CONFIG, 1):
        sign_text = "Invert" if sign < 0 else "Normal"
        print(f"  {i}. {label} ({desc}) - {sign_text}")

    print("\n[2/6] Automatically loading data files...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = [
        "Fig. 5c_Ctrl-H.xlsx",
        "Fig. 5c_HFpEF.xlsx",
        "Fig. 5c_HFrEF.xlsx",
    ]
    file_paths = [os.path.join(current_dir, f) for f in file_names]

    existing_paths = []
    for fp in file_paths:
        if os.path.exists(fp):
            existing_paths.append(fp)
        else:
            alternative_dir = os.path.join(
                os.path.dirname(current_dir),
                "Figure 5-c HFpEF & HFrEF",
            )
            alt_fp = os.path.join(alternative_dir, os.path.basename(fp))
            if os.path.exists(alt_fp):
                existing_paths.append(alt_fp)

    file_paths = existing_paths
    if not file_paths:
        print("No valid files found, exiting")
        return

    print(f"Selected {len(file_paths)} files:")
    for fp in file_paths:
        print(f"  - {os.path.basename(fp)}")

    print("\n[3/6] Reading data...")
    features = [feat[0] for feat in FEATURES_CONFIG]

    dataframes = []
    group_names = []
    feature_cols_list = []

    for fp in file_paths:
        df = pd.read_excel(fp)
        group_name = get_group_name_from_filename(fp)

        feature_cols = []
        all_found = True
        for feat in features:
            col = find_matching_column(df, feat)
            if col:
                feature_cols.append(col)
            else:
                all_found = False
                break

        if all_found:
            dataframes.append(df)
            group_names.append(group_name)
            feature_cols_list.append(feature_cols)
            print(f"  {group_name}: {len(df)} samples")

    if len(dataframes) == 0:
        print("No valid data files, exiting")
        return

    print("\n[4/6] Extracting feature means...")
    values_list = []
    for df, feature_cols in zip(dataframes, feature_cols_list):
        values = np.array([df[col].mean() for col in feature_cols])
        values_list.append(values)

        print(f"  Raw Mean: {values}")

    print("\n[5/6] Applying Z-score normalization and feature adjustment...")
    if USE_ZSCORE:
        for feat_idx in range(len(features)):
            combined = pd.concat(
                [df[feature_cols_list[i][feat_idx]] for i, df in enumerate(dataframes)]
            )
            ref_mean = combined.mean()
            ref_std = combined.std()

            for group_idx in range(len(values_list)):
                values_list[group_idx][feat_idx] = calculate_zscore(
                    values_list[group_idx][feat_idx], ref_mean, ref_std
                )

    for group_idx in range(len(values_list)):
        for feat_idx in range(len(features)):
            sign = FEATURES_CONFIG[feat_idx][3]
            if sign < 0:
                values_list[group_idx][feat_idx] *= -1

    print("Final values for plotting:")
    for name, values in zip(group_names, values_list):
        print(f"  {name}: {values}")

    print("\n[6/6] Generating radar chart...")
    colors = COLORS[: len(group_names)]
    axis_labels = generate_axis_labels(FEATURES_CONFIG)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    title = "Tri-Polar Mechanic Topology of HF Phenotyping"

    radar_chart(
        ax,
        values_list=values_list,
        labels=axis_labels,
        colors=colors,
        group_names=group_names,
        title=title,
    )

    plt.tight_layout()

    output_dir = os.path.dirname(file_paths[0])
    png_path = os.path.join(output_dir, f"{OUTPUT_FILENAME}.png")

    plt.savefig(png_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")

    # plt.show()
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
