# -*- coding: utf-8 -*-
"""
===============================================================================
Fig. 2a, 2b: Plot from Pickle Data
===============================================================================
This script loads the pre-computed Grand Average Atlas from `Fig. 2a,2b_plot_data.pkl`
and generates the Ridge Plot (Joyplot) without needing the raw data or
ssqueezepy library.

Key features:
1. Loads `grand_average_atlas` and `percentile` from pickle.
2. Reconstructs the Ridge Plot visualization.
3. Saves the output as `Grand_Average_Ridge_Plot_from_pkl.png`.
===============================================================================
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import gc


NORMALIZED_TIME_POINTS = 1000
NORMALIZED_FREQ_BINS = 256
PJENERGY_VMAX = 0.0000600
ENERGY_UNIT_SCALE = "e-6"
COLORBAR_START_PERCENT = 0
GRID_Z_NTICKS = 4
RIDGELINE_COUNT = 160
RIDGELINE_Y_SPACING = 60
RIDGELINE_ALPHA = 1
RIDGELINE_LINE_COLOR = "black"
RIDGELINE_LINE_WIDTH = 0.3
RIDGELINE_LINE_ALPHA = 0.2
RIDGELINE_REVERSE_ORDER = False

mplstyle.use("fast")

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.autolayout"] = True


def get_truncated_colormap(start_percent=0.0):
    colors_list = [
        (0.00, "#1e0f50"),
        (0.12, "#28499b"),
        (0.25, "#fcb254"),
        (0.60, "#ff6c22"),
        (1.00, "#e11c1c"),
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_spectral", colors_list)
    return custom_cmap


def calculate_percentile_threshold_from_atlas(atlas_matrix, percentile=50):
    # Re-calculate stats needed for plotting based on the loaded atlas
    flattened = atlas_matrix.flatten()
    sorted_energy = np.sort(flattened)
    cumulative_energy = np.cumsum(sorted_energy)
    total_energy = np.sum(atlas_matrix)
    target_energy = total_energy * (percentile / 100.0)
    threshold_idx = np.searchsorted(cumulative_energy, target_energy)

    if threshold_idx < len(sorted_energy):
        threshold = sorted_energy[threshold_idx]
    else:
        threshold = sorted_energy[-1]

    return {"threshold": threshold}


def plot_Fig_2_with_marginals(
    atlas_matrix,
    title_prefix="",
    apply_filter=False,
    bg_stats=None,
    vmax_unified=None,
    annotation_text=None,
    annotation_config=None,
):
    # Generate Ridge Plot Visualization
    atlas_matrix = np.maximum(atlas_matrix, 0)
    total_sum = np.sum(atlas_matrix)
    normalized_atlas_matrix = (
        atlas_matrix / total_sum if total_sum > 1e-9 else atlas_matrix
    )

    if apply_filter and bg_stats is not None:
        threshold = (
            bg_stats["threshold"] / total_sum
            if total_sum > 1e-9
            else bg_stats["threshold"]
        )
        filtered_matrix = normalized_atlas_matrix.copy()
        filtered_matrix[filtered_matrix < threshold] = 0
        normalized_atlas_matrix = filtered_matrix

    if vmax_unified is not None:
        vmax = vmax_unified
    else:
        vmax = np.max(normalized_atlas_matrix)

    fig = plt.figure()

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[35, 1],
    )

    ax_main = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    n_freqs, n_times = normalized_atlas_matrix.shape

    if not apply_filter and COLORBAR_START_PERCENT > 0:
        cmap = get_truncated_colormap(COLORBAR_START_PERCENT)
    else:
        cmap = get_truncated_colormap(0)

    ax_main.set_facecolor("white")

    freq_bins = np.logspace(np.log10(2), np.log10(100), n_freqs)

    ridgeline_indices = np.linspace(0, n_freqs - 1, RIDGELINE_COUNT, dtype=int)

    t_axis = np.linspace(0, 1000, n_times)

    global_max_energy = np.max(normalized_atlas_matrix)

    ridges_by_freq = {}
    for freq_idx in ridgeline_indices[::-1]:
        energy_profile = normalized_atlas_matrix[freq_idx, :]

        if global_max_energy > 0:
            energy_scaled = (energy_profile / global_max_energy) * RIDGELINE_Y_SPACING
        else:
            energy_scaled = np.zeros_like(energy_profile)

        baseline = freq_idx
        nonzero_mask = energy_profile > 1e-10
        if not np.any(nonzero_mask):
            continue

        mask_diff = np.diff(
            np.concatenate(([False], nonzero_mask, [False])).astype(int)
        )
        starts = np.where(mask_diff == 1)[0]
        ends = np.where(mask_diff == -1)[0]

        freq_ridges = []
        for start, end in zip(starts, ends):
            t_segment = t_axis[start:end]
            energy_segment = energy_scaled[start:end]
            energy_val_segment = energy_profile[start:end]

            if len(t_segment) < 2:
                continue

            current_max_energy = np.max(energy_val_segment)
            current_max_height = np.max(energy_segment)

            if current_max_height <= 1e-10:
                continue

            freq_ridges.append(
                {
                    "baseline": baseline,
                    "t_segment": t_segment,
                    "energy_segment": energy_segment,
                    "current_max_energy": current_max_energy,
                    "current_max_height": current_max_height,
                }
            )

        if freq_ridges:
            ridges_by_freq[freq_idx] = freq_ridges

    freq_order = list(ridges_by_freq.keys())

    for i, freq_idx in enumerate(freq_order):
        ridges = ridges_by_freq[freq_idx]
        base_zorder = 10 + i

        for ridge in ridges:
            baseline = ridge["baseline"]
            t_segment = ridge["t_segment"]
            energy_segment = ridge["energy_segment"]
            current_max_energy = ridge["current_max_energy"]
            current_max_height = ridge["current_max_height"]

            verts = []
            verts.append((t_segment[0], baseline))
            for tx, ey in zip(t_segment, baseline + energy_segment):
                verts.append((tx, ey))
            verts.append((t_segment[-1], baseline))
            verts.append((t_segment[0], baseline))

            poly = Polygon(
                verts, facecolor="none", edgecolor="none", zorder=base_zorder
            )
            ax_main.add_patch(poly)

            min_x, max_x = t_segment[0], t_segment[-1]
            min_y, max_y = baseline, baseline + current_max_height

            num_y_pixels = 50
            gradient_data = np.linspace(0, current_max_energy, num_y_pixels).reshape(
                -1, 1
            )

            im_ridge = ax_main.imshow(
                gradient_data,
                origin="lower",
                aspect="auto",
                extent=[min_x, max_x, min_y, max_y],
                cmap=cmap,
                vmin=0,
                vmax=PJENERGY_VMAX,
                zorder=base_zorder,
            )
            im_ridge.set_clip_path(poly)

            if RIDGELINE_LINE_WIDTH > 0 and RIDGELINE_LINE_COLOR.lower() != "none":
                if RIDGELINE_REVERSE_ORDER:
                    start_z = 10 + len(freq_order) + 1
                    line_zorder = start_z  # Simplified zorder logic
                else:
                    line_zorder = base_zorder + 0.1

                ax_main.plot(
                    t_segment,
                    baseline + energy_segment,
                    color=RIDGELINE_LINE_COLOR,
                    linewidth=RIDGELINE_LINE_WIDTH,
                    alpha=RIDGELINE_LINE_ALPHA,
                    solid_capstyle="round",
                    zorder=line_zorder,
                )

    x_tick_positions = [0, 200, 400, 600, 800, 1000]
    x_tick_labels = ["0", "20", "40", "60", "80", "100"]
    ax_main.set_xticks(x_tick_positions)
    ax_main.set_xticklabels(x_tick_labels)
    ax_main.set_xlabel("Normalized Cardiac Cycle (0-100%)")

    if annotation_text:
        ax_main.text(
            0.95,
            0.9,
            annotation_text,
            transform=ax_main.transAxes,
            color="black",
            fontsize=12,
            ha="right",
            va="top",
        )

    if annotation_config:
        if "vertical_line" in annotation_config:
            x_pos = annotation_config["vertical_line"]
            ax_main.axvline(
                x=x_pos, color="black", linestyle="--", linewidth=1.5, zorder=1000
            )

        if "dashed_box" in annotation_config:
            box_cfg = annotation_config["dashed_box"]
            x_range = box_cfg["x"]
            y_range = box_cfg["y"]

            # Convert x (0-100%) to indices (0-1000)
            x_start = x_range[0] * 10
            x_end = x_range[1] * 10

            # Convert y (Hz) to indices using freq_bins
            # Assuming y_range contains [f_min, f_max] in Hz
            min_freq_idx = np.argmin(np.abs(freq_bins - y_range[0]))
            max_freq_idx = np.argmin(np.abs(freq_bins - y_range[1]))

            # Ensure proper ordering
            y_start = min(min_freq_idx, max_freq_idx)
            y_end = max(min_freq_idx, max_freq_idx)

            rect = Rectangle(
                (x_start, y_start),
                x_end - x_start,
                y_end - y_start,
                linewidth=1.5,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
                zorder=1000,
            )
            ax_main.add_patch(rect)

        if "text_zones" in annotation_config:
            for zone in annotation_config["text_zones"]:
                ax_main.text(
                    zone["x"],
                    zone["y"],
                    zone["text"],
                    transform=ax_main.get_xaxis_transform(),
                    color="black",
                    fontsize=14,
                    ha="center",
                    va="top",
                    zorder=1000,
                )

    target_freqs = [2, 5, 10, 20, 50, 100]
    y_tick_positions = []
    for freq in target_freqs:
        idx = np.argmin(np.abs(freq_bins - freq))
        y_tick_positions.append(idx)
    y_tick_labels = ["2", "5", "10", "20", "50", "100"]
    ax_main.set_yticks(y_tick_positions)
    ax_main.set_yticklabels(y_tick_labels)
    ax_main.set_ylabel("Frequency (Hz)")

    ax_main.set_xlim(0, 1000)
    ax_main.set_ylim(0, n_freqs)

    if title_prefix:
        ax_main.set_title(title_prefix, pad=10)

    if ENERGY_UNIT_SCALE == "e-5":
        unit_factor = 1e-5
        unit_label = r"Normalized Energy (a.u.)"
    else:
        unit_factor = 1e-6
        unit_label = r"Normalized Energy (a.u.)"

    norm = Normalize(vmin=0, vmax=PJENERGY_VMAX)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)

    max_scaled = PJENERGY_VMAX / unit_factor
    ticks_scaled = np.linspace(0, max_scaled, GRID_Z_NTICKS)
    ticks_scaled = np.round(ticks_scaled).astype(int)
    ticks = ticks_scaled * unit_factor
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(tick)}" for tick in ticks_scaled])
    cbar.set_label(unit_label)
    cbar.ax.tick_params()

    return fig


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # List of figures to generate
    figures_to_generate = [
        {
            "pickle_file": "Fig. 2a_plot_data.pkl",
            "output_name": "Fig. 2a",
            "title": "Physiological Standard: Healthy Control Population Atlas (n=132)",
        },
        {
            "pickle_file": "Fig. 2b_plot_data.pkl",
            "output_name": "Fig. 2b",
            "title": "Pathological State: Heart Failure Population Atlas (HF, n=145)",
        },
    ]

    # Lower DPI to prevent OOM
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["agg.path.chunksize"] = 10000

    for fig_info in figures_to_generate:
        pkl_filename = os.path.join(current_dir, fig_info["pickle_file"])
        output_name = fig_info["output_name"]
        title = fig_info.get("title", "")

        print(f"\nProcessing {output_name}...")

        if not os.path.exists(pkl_filename):
            print(f"Error: Pickle file not found at {pkl_filename}")
            print(f"Skipping {output_name}...")
            continue

        try:
            with open(pkl_filename, "rb") as f:
                data = pickle.load(f)

            grand_average_atlas = data["grand_average_atlas"]
            threshold_percentile = data.get("percentile", 50)

            print(f"Loaded data from {pkl_filename}")
            print(f"Atlas shape: {grand_average_atlas.shape}")
            print(f"Percentile threshold: {threshold_percentile}")

            # Calculate stats needed for thresholding
            bg_stats = calculate_percentile_threshold_from_atlas(
                grand_average_atlas, percentile=threshold_percentile
            )
            unified_vmax = np.max(grand_average_atlas)

            print(f"Generating Ridge Plot for {output_name}...")

            # Add annotation settings
            annotation_config = None
            if output_name == "Fig. 2a":
                annotation_config = {
                    "vertical_line": 300,
                    "text_zones": [
                        {"x": 150, "y": 0.12, "text": "Main Systolic Ejection Zone"},
                        {
                            "x": 650,
                            "y": 0.12,
                            "text": "Diastolic Filling & Compliance Zone",
                        },
                    ],
                }

            if output_name == "Fig. 2b":
                annotation_config = {
                    "dashed_box": {"x": [40, 95], "y": [4.5, 15]},
                    "text_zones": [
                        {
                            "x": 220,
                            "y": 0.82,
                            "text": " ↑Pump Failure (Energy Collapse)",
                        },
                        {
                            "x": 300,
                            "y": 0.5,
                            "text": "↑Diastolic Filling Load",
                        },
                        {
                            "x": 600,
                            "y": 0.12,
                            "text": "↑Diastolic Filling Load",
                        },
                    ],
                }

            Fig_2 = plot_Fig_2_with_marginals(
                grand_average_atlas,
                title_prefix=title,
                apply_filter=True,
                bg_stats=bg_stats,
                vmax_unified=unified_vmax,
                annotation_text=None,
                annotation_config=annotation_config,
            )

            if Fig_2:
                output_png = os.path.join(current_dir, f"{output_name}.png")
                Fig_2.savefig(output_png)
                plt.close("all")
                print(f"Successfully saved plot to {output_png}")

            # Clear memory explicitly
            del grand_average_atlas
            del data
            del Fig_2
            gc.collect()

        except Exception as e:
            print(f"Error processing {output_name}: {e}")
            import traceback

            traceback.print_exc()
            continue
