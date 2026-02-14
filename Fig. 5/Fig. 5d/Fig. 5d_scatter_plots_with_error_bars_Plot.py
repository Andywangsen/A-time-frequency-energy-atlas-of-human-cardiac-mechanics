# -*- coding: utf-8 -*-
"""
==============================================================================
Figure 5-d: scatter_plots_with_error_bars Plot - Clustered Box Plot

Generates publication-quality raincloud plots comparing feature distributions across different clinical groups.
Supports automatic loading of multiple Excel files, Z-score normalization, and Bootstrap confidence intervals.

Key Features:
- Load data for different clinical groups from multiple Excel files
- Automatically find matching feature columns
- Apply Z-score normalization
- Support feature direction inversion
- Draw raincloud plots (Scatter + Median + Confidence Interval)
- Perform Kruskal-Wallis and Dunn tests
- Generate publication-quality PNG images
==============================================================================
"""

import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
from scipy import stats
from itertools import combinations

np.random.seed(42)

OUTPUT_RAINCLOUD = "Fig. 5d_scatter_plots_with_error_bars"

FEATURES_CONFIG = [
    ("E_{2-100Hz}[0-10%]", "Systolic Initial Impulse", "", 1),
    ("E_{2-12Hz}[40-100%]", "Diastolic Filling Load", "", +1),
    ("H_{10-80Hz}[39-70%]", "Myocardial Stiffness Tone", "", -1),
]

COLORS = {
    "Ctrl": "#4DAF4A",
    "HFpEF": "#984EA3",
    "HFrEF": "#E41A1C",
}
COLOR_LIST = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C"]

ALPHA_FILL = 0.25
ALPHA_LINE = 0.5
LINE_WIDTH = 1

BOOTSTRAP_N = 10000
CI_LEVEL = 0.95
CLUSTER_SPACING = 1.0

RAIN_JITTER = 0.04
RAIN_ALPHA = 0.4
RAIN_SIZE = 6
CLOUD_WIDTH = 0
CLOUD_ALPHA = 0

USE_ZSCORE = True


def find_matching_column(df, feature_pattern):
    """
    Find matching column name in DataFrame.

    Args:
        df: pandas DataFrame object
        feature_pattern: Feature pattern string to search

    Returns:
        Matching column name, or None if not found
    """
    for col in df.columns:
        if feature_pattern in col:
            return col
    return None


def get_group_name_from_filename(filepath):
    """
    Extract group name from file path (filename without extension).

    Args:
        filepath: Full path of the file

    Returns:
        Filename without extension as group name
    """
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0].replace("Fig. 5d_", "")


def get_color_for_group(group_name, idx):
    """
    Get color for group based on name match, fallback to color list if no match.

    Args:
        group_name: Group name string
        idx: Index in color list

    Returns:
        Hex color code
    """
    for key in COLORS:
        if key.lower() in group_name.lower():
            return COLORS[key]
    return COLOR_LIST[idx % len(COLOR_LIST)]


def dunn_test_bonferroni(groups_data, group_names):
    """
    Perform Dunn test with Bonferroni correction.

    Used for post-hoc pairwise comparisons after significant Kruskal-Wallis test.
    Calculates rank statistics, Z-values, and corrected p-values.

    Args:
        groups_data: List of data for each group
        group_names: List of group names

    Returns:
        Dictionary with keys (Group1, Group2) and values containing z, p_raw, p_corrected
    """
    all_data = []
    group_labels = []
    for i, data in enumerate(groups_data):
        all_data.extend(data)
        group_labels.extend([i] * len(data))

    all_data = np.array(all_data)
    group_labels = np.array(group_labels)

    ranks = stats.rankdata(all_data)
    n_total = len(all_data)

    group_mean_ranks = {}
    group_sizes = {}
    for i in range(len(groups_data)):
        mask = group_labels == i
        group_mean_ranks[i] = np.mean(ranks[mask])
        group_sizes[i] = np.sum(mask)

    n_comparisons = len(list(combinations(range(len(groups_data)), 2)))
    results = {}

    for i, j in combinations(range(len(groups_data)), 2):
        ni, nj = group_sizes[i], group_sizes[j]
        mean_rank_diff = abs(group_mean_ranks[i] - group_mean_ranks[j])

        se = np.sqrt((n_total * (n_total + 1) / 12) * (1 / ni + 1 / nj))

        z = mean_rank_diff / se

        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        p_corrected = min(p_value * n_comparisons, 1.0)

        results[(group_names[i], group_names[j])] = {
            "z": z,
            "p_raw": p_value,
            "p_corrected": p_corrected,
        }

    return results


def get_significance_text(p_value):
    """
    Get significance symbol based on p-value.

    Args:
        p_value: p-value

    Returns:
        Significance symbol: '***'(p<0.001), '**'(p<0.01), '*'(p<0.05), 'ns'(not significant)
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def bootstrap_median_ci(data, n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL):
    """
    Calculate confidence interval for median using Bootstrap.

    Suitable for non-normal data, calculates percentile confidence intervals via resampling.

    Args:
        data: Input data array
        n_bootstrap: Number of bootstrap resamples (default 10000)
        ci_level: Confidence level (default 0.95)

    Returns:
        Tuple (median, CI lower bound, CI upper bound)
    """
    data = np.array(data)
    n = len(data)

    bootstrap_medians = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_medians[i] = np.median(sample)

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_medians, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_medians, (1 - alpha / 2) * 100)
    median = np.median(data)

    return median, ci_lower, ci_upper


def raincloud_plot(ax, raw_data, group_order, color_map, features_config):
    np.random.seed(42)
    """
    Draw Raincloud Plot.

    Overlays at the same X position:
    - Scatter plot (Rain): Jittered raw data points
    - Median + CI (Umbrella): Median and 95% CI from Bootstrap

    Args:
        ax: matplotlib axes object
        raw_data: Dictionary mapping group names to feature data lists
        group_order: Ordered list of group names
        color_map: Dictionary mapping group names to colors
        features_config: Feature configuration list

    Returns:
        Tuple (cluster_centers, group_width, group_gap) for significance annotation
    """
    from scipy.stats import gaussian_kde

    n_groups = len(group_order)
    n_features = len(features_config)
    cluster_centers = np.arange(n_features) * CLUSTER_SPACING

    group_width = 0.25
    group_gap = 0.06

    for feat_idx in range(n_features):
        center = cluster_centers[feat_idx]
        total_width = n_groups * group_width + (n_groups - 1) * group_gap
        start = center - total_width / 2 + group_width / 2

        for group_idx, group in enumerate(group_order):
            x_pos = start + group_idx * (group_width + group_gap)
            data = np.array(raw_data[group][feat_idx])
            color = color_map[group]

            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            whisker_low = q1 - 1.5 * iqr
            whisker_high = q3 + 1.5 * iqr
            whisker_low = max(whisker_low, np.min(data))
            whisker_high = min(whisker_high, np.max(data))

            try:
                kde = gaussian_kde(data)
                y_vals = np.linspace(whisker_low, whisker_high, 100)
                density = kde(y_vals)
                density = density / density.max() * (group_width * 0.8)

            except Exception:
                pass

            jitter = np.random.uniform(
                -group_width * 0.2, group_width * 0.2, size=len(data)
            )
            ax.scatter(
                x_pos + jitter,
                data,
                s=RAIN_SIZE,
                color=color,
                alpha=RAIN_ALPHA,
                zorder=3,
                edgecolors="none",
            )

            median, ci_lower, ci_upper = bootstrap_median_ci(data)

            ax.plot(
                [x_pos, x_pos],
                [ci_lower, ci_upper],
                color=color,
                linewidth=2,
                solid_capstyle="round",
                zorder=6,
            )
            cap_w = group_width * 0.2
            ax.plot(
                [x_pos - cap_w, x_pos + cap_w],
                [ci_lower, ci_lower],
                color=color,
                linewidth=2,
                zorder=6,
            )
            ax.plot(
                [x_pos - cap_w, x_pos + cap_w],
                [ci_upper, ci_upper],
                color=color,
                linewidth=2,
                zorder=6,
            )
            ax.scatter(
                [x_pos],
                [median],
                s=80,
                color=color,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )

    feature_labels = [cfg[1] for cfg in features_config]
    ax.set_xticks(cluster_centers)
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels(feature_labels)
    ax.set_xlim(cluster_centers[0] - 0.6, cluster_centers[-1] + 0.6)
    ax.set_ylim(-3.8, 4.3)
    ax.set_ylabel("Normalized Intensity (Z-score)")
    ax.tick_params(axis="y")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    legend_handles = [
        plt.scatter([], [], s=80, color=color_map[g], linewidths=1.5)
        for g in group_order
    ]

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[g],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
        )
        for g in group_order
    ]
    ax.legend(
        handles=legend_handles, labels=group_order, loc="upper right", frameon=False
    )
    ax.legend(legend_handles, group_order, loc="upper right")
    ax.set_title("Quantitative Feature Divergence", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return cluster_centers, group_width, group_gap


def add_significance_annotations_raincloud(
    ax,
    normalized_data,
    group_order,
    features_config,
    cluster_centers,
    group_width,
    group_gap,
):
    """
    Add significance annotations to raincloud plot.

    Performs global Kruskal-Wallis test, if significant then performs Dunn test,
    annotates significance symbols for comparisons between HFpEF and HFrEF groups.

    Args:
        ax: matplotlib axes object
        normalized_data: Dictionary of normalized data
        group_order: Ordered list of group names
        features_config: Feature configuration list
        cluster_centers: Center positions of features on x-axis
        group_width: Width occupied by each group
        group_gap: Gap between groups
    """
    n_groups = len(group_order)
    n_features = len(features_config)

    ctrl_idx = hfpef_idx = hfref_idx = None
    for i, g in enumerate(group_order):
        if "ctrl" in g.lower():
            ctrl_idx = i
        elif "hfpef" in g.lower() or "pef" in g.lower():
            hfpef_idx = i
        elif "hfref" in g.lower() or "ref" in g.lower():
            hfref_idx = i

    if ctrl_idx is None or hfpef_idx is None or hfref_idx is None:
        return

    def get_x_pos(feat_idx, group_idx):
        center = cluster_centers[feat_idx]
        total_width = n_groups * group_width + (n_groups - 1) * group_gap
        start = center - total_width / 2 + group_width / 2
        return start + group_idx * (group_width + group_gap)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    line_color = "#333333"
    line_width = 1.2

    for feat_idx in range(n_features):
        groups_data = [normalized_data[g][feat_idx] for g in group_order]
        h_stat, kw_p = stats.kruskal(*groups_data)

        if kw_p >= 0.05:
            sig_text = "ns"
        else:
            dunn_results = dunn_test_bonferroni(groups_data, group_order)
            key1 = (group_order[hfpef_idx], group_order[hfref_idx])
            key2 = (group_order[hfref_idx], group_order[hfpef_idx])
            p_corrected = dunn_results.get(
                key1, dunn_results.get(key2, {"p_corrected": 1.0})
            )["p_corrected"]
            sig_text = get_significance_text(p_corrected)

        x1 = get_x_pos(feat_idx, hfpef_idx)
        x2 = get_x_pos(feat_idx, hfref_idx)

        _, _, ci_upper1 = bootstrap_median_ci(
            normalized_data[group_order[hfpef_idx]][feat_idx]
        )
        _, _, ci_upper2 = bootstrap_median_ci(
            normalized_data[group_order[hfref_idx]][feat_idx]
        )
        max_y = max(ci_upper1, ci_upper2)
        y_line = max_y + y_range * 0.06

        ax.plot(
            [x1, x1, x2, x2],
            [y_line - y_range * 0.015, y_line, y_line, y_line - y_range * 0.015],
            color=line_color,
            linewidth=line_width,
        )
        ax.text(
            (x1 + x2) / 2,
            y_line + y_range * 0.01,
            sig_text,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def main():
    """
    Main program entry.

    Executes complete data processing pipeline:
    1. Load multiple Excel files
    2. Extract specified feature columns
    3. Apply Z-score normalization
    4. Draw raincloud plot
    5. Perform statistical tests
    6. Save result images
    """
    print("=" * 70)
    print("=" * 70)
    print("Figure 5-d Raincloud Plot - Clustered Box Plot")
    print("=" * 70)

    print(f"\n[Config]Loaded {len(FEATURES_CONFIG)} heart function features:")
    for i, (col, label, desc, sign) in enumerate(FEATURES_CONFIG, 1):
        sign_text = "Invert" if sign < 0 else "Normal"
        print(f"  {i}. {label:30s} - {sign_text}")

    print("\n[Step 1]Automatically loading data files...")
    features = [feat[0] for feat in FEATURES_CONFIG]

    dataframes = []
    group_names = []
    feature_cols_list = []
    raw_data = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = [
        "Fig. 5d_Ctrl-H.xlsx",
        "Fig. 5d_HFpEF.xlsx",
        "Fig. 5d_HFrEF.xlsx",
    ]
    file_paths = [os.path.join(current_dir, f) for f in file_names]

    print(f"  Selected {len(file_paths)} files:")
    for fp in file_paths:
        print(f"    - {os.path.basename(fp)}")

    print("\n[Step 2]Reading data...")
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
            raw_data[group_name] = [df[col].dropna().values for col in feature_cols]
            print(f"  [OK] {group_name:20s} - {len(df):4d} samples")

    if len(dataframes) == 0:
        print("  [X] No valid data files, exiting")
        return

    print("\n[Step 3]Applying Z-score normalization...")
    if USE_ZSCORE:
        normalized_data = {group: [] for group in raw_data}
        for feat_idx in range(len(FEATURES_CONFIG)):
            all_vals = np.concatenate([raw_data[g][feat_idx] for g in raw_data])
            global_mean = np.mean(all_vals)
            global_std = np.std(all_vals) + 1e-10

            sign = FEATURES_CONFIG[feat_idx][3]
            for group in raw_data:
                z_vals = (raw_data[group][feat_idx] - global_mean) / global_std
                if sign < 0:
                    z_vals = -z_vals
                normalized_data[group].append(z_vals)
        print(
            f"  [OK] Normalization complete - Global Mean={global_mean:.3f}, Std={global_std:.3f}"
        )

    print("\n[Step 4]Confirming group order and colors...")
    group_order = []
    for target in ["Ctrl", "HFpEF", "HFrEF"]:
        for group in group_names:
            if target.lower() in group.lower() and group not in group_order:
                group_order.append(group)
                break
    for group in group_names:
        if group not in group_order:
            group_order.append(group)

    color_map = {g: get_color_for_group(g, i) for i, g in enumerate(group_order)}
    print(f"  [OK] Group Order: {' -> '.join(group_order)}")

    print("\n[Step 5]Plotting raincloud plot...")
    output_dir = os.path.dirname(file_paths[0])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    rc_centers, rc_width, rc_gap = raincloud_plot(
        ax2, normalized_data, group_order, color_map, FEATURES_CONFIG
    )
    add_significance_annotations_raincloud(
        ax2, normalized_data, group_order, FEATURES_CONFIG, rc_centers, rc_width, rc_gap
    )
    plt.tight_layout()

    png_path2 = os.path.join(output_dir, f"{OUTPUT_RAINCLOUD}.png")
    fig2.savefig(png_path2, bbox_inches="tight", facecolor="white", dpi=300)
    png_path2 = os.path.join(output_dir, f"{OUTPUT_RAINCLOUD}.png")
    fig2.savefig(png_path2, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"  [OK] Raincloud plot saved: {png_path2}")

    print("\n[Step 6]Statistical Test (Kruskal-Wallis + Dunn's Test)...")
    print("-" * 70)

    for feat_idx, (col, label, desc, sign) in enumerate(FEATURES_CONFIG):
        print(f"\n{label}:")
        groups_data = [normalized_data[g][feat_idx] for g in group_order]

        if len(groups_data) >= 2:
            h_stat, kw_p = stats.kruskal(*groups_data)
            kw_sig = "Significant [OK]" if kw_p < 0.05 else "Not Significant"
            print(f"  [Global] Kruskal-Wallis: H={h_stat:.3f}, p={kw_p:.4f} ({kw_sig})")

        if kw_p < 0.05:
            print("  [Post-hoc] Dunn's Test (Bonferroni correction):")
            dunn_results = dunn_test_bonferroni(groups_data, group_order)
            for (g1, g2), result in dunn_results.items():
                sig = get_significance_text(result["p_corrected"])
                print(
                    f"    {g1:15s} vs {g2:15s}: z={result['z']:6.3f}, p_corrected={result['p_corrected']:.4f} {sig}"
                )
        else:
            print(
                "  [Post-hoc] Global test not significant, skipping pairwise comparisons"
            )

    print("\n" + "=" * 70)
    print("[Done]All processing complete!")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
