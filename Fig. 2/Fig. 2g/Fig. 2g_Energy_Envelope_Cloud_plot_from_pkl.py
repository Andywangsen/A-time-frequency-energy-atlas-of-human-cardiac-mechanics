# -*- coding: utf-8 -*-
"""
==============================================================================
Panel g: The Cloud & The Lightning - Energy Envelope Visualization

Data Privacy and Reproducibility Statement:
Due to the patient privacy and data sensitivity of the original clinical data
(such as high-precision ECG/SCG waveforms), and in accordance with ethical review
and privacy protection regulations, the original dataset cannot be publicly shared.
To ensure the transparency and reproducibility of the research results, we have provided
the original analysis code and the code for generating the figures consistent with
the paper using the intermediate result files (.pkl) after processing and feature
extraction.

Generates publication-quality ensemble plots showing population energy patterns
with test-retest stability across cardiac cycles. Implements "Mist + Neon" design:
- Background (Mist): Individual energy clouds showing population distribution
- Foreground (Neon): Population averages with Week 1 as baseline, Week 2/3 as follow-ups
- Anatomical anchors: Cardiac phase zones with smooth color transitions
==============================================================================
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================
#        Plotting Functions (Copied from original script)
# ==============================


def plot_cloud_and_lightning_reproduce(
    results, population_weeks, plot_params, output_path=None, show_plot=True, title=None
):
    """
    Reproduce "Cloud & Lightning" Plot

    Parameters:
    - results: dict, processed data
    - population_weeks: dict, population average week data
    - plot_params: dict, plotting parameters
    - output_path: str, save path
    - show_plot: bool, whether to show plot
    - title: str, chart title
    """
    print(f"\n{'=' * 60}")
    print("Reproducing Panel g: Cloud & Lightning")
    print(f"{'=' * 60}")

    # Extract configuration from parameters
    NORMALIZED_TIME_POINTS = plot_params["NORMALIZED_TIME_POINTS"]
    OUTPUT_DPI = plot_params["OUTPUT_DPI"]
    FIG_WIDTH = plot_params["FIG_WIDTH"]
    FIG_HEIGHT = plot_params["FIG_HEIGHT"]

    # Background Mist Parameters
    CLOUD_LINE_WIDTH = plot_params["CLOUD_LINE_WIDTH"]
    CLOUD_ALPHA = plot_params["CLOUD_ALPHA"]
    CLOUD_COLOR = plot_params["CLOUD_COLOR"]

    # Week 1 Parameters
    WEEK1_LINE_COLOR = plot_params["WEEK1_LINE_COLOR"]
    WEEK1_SD_COLOR = plot_params["WEEK1_SD_COLOR"]
    WEEK1_LINE_WIDTH = plot_params["WEEK1_LINE_WIDTH"]
    WEEK1_SD_ALPHA = plot_params["WEEK1_SD_ALPHA"]

    # Week 2 Parameters
    WEEK2_LINE_COLOR = plot_params["WEEK2_LINE_COLOR"]
    WEEK2_LINE_WIDTH = plot_params["WEEK2_LINE_WIDTH"]
    WEEK2_LINE_STYLE = plot_params["WEEK2_LINE_STYLE"]

    # Week 3 Parameters
    WEEK3_LINE_COLOR = plot_params["WEEK3_LINE_COLOR"]
    WEEK3_LINE_WIDTH = plot_params["WEEK3_LINE_WIDTH"]
    WEEK3_LINE_STYLE = plot_params["WEEK3_LINE_STYLE"]

    # Zone Parameters
    ZONE1_START = plot_params["ZONE1_START"]
    ZONE1_END = plot_params["ZONE1_END"]
    ZONE1_COLOR = plot_params["ZONE1_COLOR"]
    ZONE2_START = plot_params["ZONE2_START"]
    ZONE2_END = plot_params["ZONE2_END"]
    ZONE2_COLOR = plot_params["ZONE2_COLOR"]
    ZONE3_START = plot_params["ZONE3_START"]
    ZONE3_END = plot_params["ZONE3_END"]
    ZONE3_COLOR = plot_params["ZONE3_COLOR"]
    PHASE_BG_ALPHA = plot_params["PHASE_BG_ALPHA"]
    ZONE_TRANSITION_WIDTH = plot_params["ZONE_TRANSITION_WIDTH"]

    # ========== Top-tier Journal Font Settings ==========
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )

    # Create Figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=OUTPUT_DPI)

    # Set white background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # X-axis: Normalized Cardiac Cycle (0-100%)
    x_axis = np.linspace(0, 100, NORMALIZED_TIME_POINTS)

    # Collect all envelopes for Y-axis range
    all_envelopes = []
    for txt_name, data in results.items():
        all_envelopes.append(data["envelope"])

    # Pre-calculate Y-axis range
    if all_envelopes:
        all_data = np.concatenate([env.flatten() for env in all_envelopes])
        y_max = (
            np.percentile(all_data[all_data > 0], 99) * 1.15
            if np.any(all_data > 0)
            else 1.0
        )
    else:
        y_max = 1.0

    # ========== Layer 0: Anatomical Background Zones ==========
    print("  - Drawing anatomical background zones...")

    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple (0-1 range)"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        """Convert RGB tuple (0-1 range) to hex color"""
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    def blend_colors(color1, color2, ratio):
        """Blend two colors, ratio=0 returns color1, ratio=1 returns color2"""
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
        blended = tuple(rgb1[i] * (1 - ratio) + rgb2[i] * ratio for i in range(3))
        return rgb_to_hex(blended)

    # Define three zones
    zones = [
        {"start": ZONE1_START, "end": ZONE1_END, "color": ZONE1_COLOR},
        {"start": ZONE2_START, "end": ZONE2_END, "color": ZONE2_COLOR},
        {"start": ZONE3_START, "end": ZONE3_END, "color": ZONE3_COLOR},
    ]

    # Plot core part of each zone
    for i, zone in enumerate(zones):
        core_start = zone["start"]
        core_end = zone["end"]

        if i > 0:
            core_start = zone["start"] + ZONE_TRANSITION_WIDTH / 2

        if i < len(zones) - 1:
            core_end = zone["end"] - ZONE_TRANSITION_WIDTH / 2

        if core_start < core_end:
            ax.axvspan(
                core_start,
                core_end,
                facecolor=zone["color"],
                alpha=PHASE_BG_ALPHA,
                zorder=0,
                edgecolor="none",
            )

    # Plot gradient transition between zones
    for i in range(len(zones) - 1):
        zone_current = zones[i]
        zone_next = zones[i + 1]

        transition_start = zone_current["end"] - ZONE_TRANSITION_WIDTH / 2
        transition_end = zone_next["start"] + ZONE_TRANSITION_WIDTH / 2

        if transition_start < transition_end:
            n_steps = max(10, int(ZONE_TRANSITION_WIDTH * 2))
            step_width = (transition_end - transition_start) / n_steps

            for j in range(n_steps):
                ratio = j / (n_steps - 1) if n_steps > 1 else 0.5
                blended_color = blend_colors(
                    zone_current["color"], zone_next["color"], ratio
                )

                step_start = transition_start + j * step_width
                step_end = step_start + step_width

                ax.axvspan(
                    step_start,
                    step_end,
                    facecolor=blended_color,
                    alpha=PHASE_BG_ALPHA,
                    zorder=0,
                    edgecolor="none",
                )

    # ========== Layer 1: Background Mist ==========
    print("  - Drawing background mist...")
    for txt_name, data in results.items():
        envelope = data["envelope"]
        ax.plot(
            x_axis,
            envelope,
            color=CLOUD_COLOR,
            linewidth=CLOUD_LINE_WIDTH,
            alpha=CLOUD_ALPHA,
            zorder=1,
        )

    print(f"    - Drew {len(all_envelopes)} background lines")

    # ========== Layer 2: Foreground Neon ==========
    if population_weeks:
        print("  - Drawing foreground neon...")

        # Week 1
        if 1 in population_weeks:
            week1_data = population_weeks[1]
            envelope = week1_data["envelope"]
            n = week1_data["n_subjects"]
            er = week1_data["er"]
            er_std = week1_data["er_std"]

            band_lower = week1_data["sd_lower"]
            band_upper = week1_data["sd_upper"]

            ax.fill_between(
                x_axis,
                band_lower,
                band_upper,
                color=WEEK1_SD_COLOR,
                alpha=WEEK1_SD_ALPHA,
                edgecolor="none",
                zorder=2,
                label="Week1 ± 1SD",
            )

            ax.plot(
                x_axis,
                envelope,
                color=WEEK1_LINE_COLOR,
                linewidth=WEEK1_LINE_WIDTH,
                linestyle="-",
                alpha=1.0,
                zorder=5,
                label="Week1 Mean",
            )

            print(f"    - Week 1: n={n}, ER={er:.3f}±{er_std:.3f}")

        # Week 2
        if 2 in population_weeks:
            week2_data = population_weeks[2]
            envelope = week2_data["envelope"]
            n = week2_data["n_subjects"]
            er = week2_data["er"]
            er_std = week2_data["er_std"]

            ax.plot(
                x_axis,
                envelope,
                color=WEEK2_LINE_COLOR,
                linewidth=WEEK2_LINE_WIDTH,
                linestyle=WEEK2_LINE_STYLE,
                alpha=0.9,
                zorder=6,
                label="Week2 Mean",
            )

            print(f"    - Week 2: n={n}, ER={er:.3f}±{er_std:.3f}")

        # Week 3
        if 3 in population_weeks:
            week3_data = population_weeks[3]
            envelope = week3_data["envelope"]
            n = week3_data["n_subjects"]
            er = week3_data["er"]
            er_std = week3_data["er_std"]

            ax.plot(
                x_axis,
                envelope,
                color=WEEK3_LINE_COLOR,
                linewidth=WEEK3_LINE_WIDTH,
                linestyle=WEEK3_LINE_STYLE,
                alpha=0.9,
                zorder=6,
                label="Week3 Mean",
            )

            print(f"    - Week 3: n={n}, ER={er:.3f}±{er_std:.3f}")

    # ========== Set Axes ==========
    ax.set_xlim(0, 100)
    ax.set_ylim(0, y_max)

    ax.set_xlabel("Normalized Cardiac Cycle (0-100%)", fontweight="normal")
    ax.set_ylabel("Instantaneous Energy (a.u.)", fontweight="normal")

    # Set Title
    ax.set_title(
        "Longitudinal Topological Consistency (n=24, 3 Times @ 3 Weeks)",
        fontweight="normal",
        fontsize=10,
        pad=10,
    )

    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))
    ax.yaxis.get_offset_text().set_fontsize(8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Phase Labels (Changed to anatomical labels)
    ax.text(15, y_max * 0.85, "Main Systolic", ha="center", fontsize=10, color="black")
    ax.text(65, y_max * 0.85, "Diastolic", ha="center", fontsize=10, color="black")
    ax.text(
        45,
        y_max * 0.65,
        "Phase-Locked Stability",
        ha="center",
        fontsize=10,
        color="black",
    )

    # Legend
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#E0E0E0",
        fancybox=False,
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()

    # Save Figure
    if output_path:
        # PNG High Resolution
        fig.savefig(
            output_path,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"  - [OK] Saved: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ==============================
#        Main Function
# ==============================


def main():
    """Main Function"""
    print("\n" + "=" * 60)
    print("Panel g Data Reproduction Script")
    print("=" * 60)

    # Automatically get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Automatically find data file
    print("\nSearching for data files (.pkl)...")
    import glob

    pkl_files = glob.glob(os.path.join(script_dir, "*.pkl"))

    if not pkl_files:
        print(f"Error: No .pkl files found in directory {script_dir}.")
        # Fallback: Allow manual selection
        root = tk.Tk()
        root.withdraw()
        data_path = filedialog.askopenfilename(
            title="Manually select plot data file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not data_path:
            return
    else:
        # If multiple, sort by modification time and pick newest
        pkl_files.sort(key=os.path.getmtime, reverse=True)
        data_path = pkl_files[0]
        print(f"  - Automatically found and loaded: {os.path.basename(data_path)}")

    # 2. Load Data
    print("\nLoading plot data...")
    try:
        with open(data_path, "rb") as f:
            plot_data = pickle.load(f)
        print("  - [OK] Data loaded successfully")

        # Show data info
        metadata = plot_data.get("metadata", {})
        print("\nData Information:")
        print(f"  - Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"  - Subjects: {metadata.get('n_subjects', 'N/A')}")
        print(f"  - Pairs: {metadata.get('n_pairs', 'N/A')}")
        # print(f"  - Original Excel: {metadata.get('excel_path', 'N/A')}")

    except Exception as e:
        print(f"Error: Failed to load data: {e}")
        return

    # 3. Determine output directory (default to script directory)
    output_dir = script_dir
    print(f"\nOutput directory: {output_dir}")

    # 5. Generate output filename

    output_filename = "Fig. 2g_Energy_Envelope_Cloud.png"
    output_path = os.path.join(output_dir, output_filename)

    # 6. Reproduce Plot
    print("\nStarting plot reproduction...")
    plot_cloud_and_lightning_reproduce(
        results=plot_data["results"],
        population_weeks=plot_data["population_weeks"],
        plot_params=plot_data["plot_params"],
        output_path=output_path,
        show_plot=False,
    )

    print("\n" + "=" * 60)
    print("Reproduction complete!")
    print("=" * 60)
    print(f"Reproduced plot saved to:\n{output_path}")


if __name__ == "__main__":
    main()
