# -*- coding: utf-8 -*-
"""
==============================================================================
Extended Data Fig. 2 Plotting Script

Function:
Load pre-calculated efficiency analysis results from pickle file and generate dual-axis efficiency curve plot.

Usage:
1. Run main analysis script to generate pickle data file
2. Run this script to load data and generate image

==============================================================================
"""

import os
import pickle
import matplotlib.pyplot as plt


# Configure font support
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")


def plot_efficiency_curve(df_res, output_dir, label_pair_name):
    """
    Plot Dual-Axis Efficiency Curve

    Plot three curves showing the relationship between Area, Significance, and Efficiency Index,
    and mark the optimal threshold point.

    Args:
        df_res: DataFrame containing analysis results
        output_dir: Directory for output images
        label_pair_name: Name of label pair (used for filename)

    Returns:
        None (Directly displays and saves image)
    """
    x = df_res["Threshold_Percent"]

    # Set plot style
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        try:
            plt.style.use("seaborn-whitegrid")
        except:
            plt.style.use("ggplot")
    fig, ax1 = plt.subplots(figsize=(10, 7), dpi=150)

    # ========== Plot Area Curve (Left Axis) ==========
    color_area = "#64CB9C"  # Grey
    ax1.set_xlabel("Energy Filtering Threshold (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel(
        "Area Remaining (%)", color=color_area, fontsize=13, fontweight="bold"
    )

    line1 = ax1.plot(
        x,
        df_res["Remaining_Area"] * 100,
        color=color_area,
        # linestyle="--",
        linewidth=1,
        alpha=1,
        label="Area Remaining (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_area)
    ax1.set_ylim(0, 105)
    ax1.grid(visible=True, linestyle="--", alpha=0.3)

    # ========== Plot Significance and Efficiency Curves (Right Axis) ==========
    ax2 = ax1.twinx()
    color_sig = "#64BAF8"  # Blue
    color_eff = "#FE7BAC"  # Red

    ax2.set_ylabel(
        "Significance & Efficiency Index(Blue+Pink)",
        color="black",
        fontsize=13,
        fontweight="bold",
    )

    # Average Significance Curve
    line2 = ax2.plot(
        x,
        df_res["Mean_Significance"],
        color=color_sig,
        linewidth=1,
        alpha=1,
        label="Mean Significance (-log$_{10}$ (P-value))",
    )

    # Efficiency Index Curve (Core)
    ax2.fill_between(x, 0, df_res["Efficiency_Index"], color=color_eff, alpha=0.1)
    line3 = ax2.plot(
        x,
        df_res["Efficiency_Index"],
        color=color_eff,
        linewidth=1,
        label="Net Signal Gain ",
    )

    # ========== Mark Optimal Point ==========
    max_idx = df_res["Efficiency_Index"].idxmax()
    peak_x = df_res.iloc[max_idx]["Threshold_Percent"]
    peak_y = df_res.iloc[max_idx]["Efficiency_Index"]

    # Draw vertical line
    plt.axvline(x=peak_x, color="black", linestyle=":", linewidth=2, ymax=0.95)

    # Mark star
    ax2.plot(
        peak_x,
        peak_y,
        marker="*",
        markersize=20,
        color="gold",
        markeredgecolor="black",
        zorder=10,
    )

    # Annotation text
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=color_eff, lw=1)
    ax2.annotate(
        f"Optimal Filter Threshold: {peak_x:.0f}%\n(Maximum signal gain)",
        xy=(peak_x, peak_y),
        xytext=(peak_x, peak_y + 0.15),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        fontsize=11,
        fontweight="bold",
        ha="center",
        color=color_eff,
        bbox=bbox_props,
    )

    # ========== Merge Legends ==========
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    plt.title(
        f"Optimal Filter Selection: Ctrl-H vs HF",
        fontsize=15,
        y=1.15,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save as PNG
    save_path_png = os.path.join(output_dir, f"Extended Data Fig. 2.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    print(f"[OK] PNG image saved: {save_path_png}")

    plt.show()


def main():
    """
    Main Program Entry

    Process Flow:
    1. Get script directory
    2. Find pickle data file
    3. Load analysis results
    4. Generate efficiency curve plot
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, "Extended Data Fig. 2_plot_data.pkl")

    # Check if pickle file exists
    if not os.path.exists(pkl_path):
        print(f"Error: pickle file not found: {pkl_path}")
        print("Please run the main analysis script to generate this file first.")
        return

    # Load data
    print(f"Loading data: {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df_res = data["df_res"]
    label_a = data["label_a"]
    label_b = data["label_b"]

    print("[OK] Data loading complete.")
    print("Generating image...")
    plot_efficiency_curve(df_res, script_dir, f"{label_a}_vs_{label_b}")
    print("[OK] Image generation complete.")


if __name__ == "__main__":
    main()
