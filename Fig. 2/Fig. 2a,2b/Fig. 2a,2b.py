# -*- coding: utf-8 -*-
"""
===============================================================================
Fig. 2a, 2b: Grand Average Ridge Plot Analysis (SCG Time-Frequency Distribution)
===============================================================================
[Analysis Method ]
    This script performs a high-resolution Time-Frequency Analysis of Seismocardiogram
    (SCG) signals to generate the Grand Average Ridge Plots (Joyplots) for Fig. 2a/2b.

    The methodological pipeline includes:
    1. Preprocessing: Bandpass filtering of raw SCG signals (Butterworth).
    2. Segmentation: Extraction of individual cardiac cycles based on ECG R-peak timestamps.
    3. Time-Frequency Transform: Application of Synchrosqueezed Wavelet Transform (SSWT/SSTF)
       using a Generalized Morse Wavelet (via `ssqueezepy`) to obtain sharpened spectral resolution.
    4. Normalization: Resampling of each beat's time-frequency representation (TFR) to a
       standardized 0-100% cardiac cycle duration.
    5. Aggregation: Computation of the ensemble average (Grand Average Atlas) across multiple
       subjects/files.
    6. Visualization: Rendering the spectral energy distribution as a Ridge Plot (stacked
       spectral densities) with percentile-based background noise thresholding.


[Data Privacy & Security Statement]
    Strict Data Privacy Control:
    This source code is provided for methodological reference and peer review purposes only.
    Due to strict data privacy regulations and institutional confidentiality
    agreements, the underlying raw physiological datasets (`.txt` source files) and the
    participant metadata index (`summary.xlsx`) are NOT included in this repository.

    To execute this script:
    Users must provide their own dataset formatted according to the specifications defined
    in the `df_summary` reading logic.

===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import zoom
import pandas as pd
from matplotlib.ticker import MultipleLocator
import pickle


mplstyle.use("fast")

SHOW_PLOTS_AFTER_ANALYSIS = False

NORMALIZED_TIME_POINTS = 1000
NORMALIZED_FREQ_BINS = 256

ENERGY_VMAX = 15
CWTENERGY_VMAX = 140
PJENERGY_VMAX = 0.0000600

ENERGY_UNIT_SCALE = "e-6"

COLORBAR_START_PERCENT = 0

GRID_X_DTICK = 200
GRID_Z_NTICKS = 4

FIG5_2FONT_LABEL = 12
FIG5_2FONT_TICK = 10
FIG5_2FONT_LEGEND = 10

RIDGELINE_COUNT = 160
RIDGELINE_Y_SPACING = 60
RIDGELINE_ALPHA = 1
RIDGELINE_LINE_COLOR = "black"
RIDGELINE_LINE_WIDTH = 0.3
RIDGELINE_LINE_ALPHA = 0.2
RIDGELINE_REVERSE_ORDER = False

try:
    from ssqueezepy import ssq_cwt

    SSQUEEZEPY_AVAILABLE = True
except ImportError:
    SSQUEEZEPY_AVAILABLE = False
    print("Warning: ssqueezepy module not found. SSTF analysis will not be available.")

mplstyle.use("fast")

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.autolayout"] = True

COLOR_ECG = "#333333"
COLOR_SCG = "#009E73"
COLOR_R_PEAK = "black"


def convert_scg_unit(raw_scg, unit="ms2"):
    scale_factor_g = 2.0 / 32768.0

    if unit == "g":
        return raw_scg * scale_factor_g
    elif unit == "ms2":
        return raw_scg * scale_factor_g * 9.80665
    elif unit == "mg":
        return raw_scg * scale_factor_g * 1000.0
    else:
        raise ValueError("Unknown unit")


def convert_ecg_unit(raw_ecg, unit="mV"):
    v_ref = 2454.0
    gain = 12.0
    max_adc = 32768.0

    full_scale_mv = v_ref / gain

    lsb_mv = full_scale_mv / max_adc

    if unit == "mV":
        return raw_ecg * lsb_mv
    elif unit == "uV":
        return raw_ecg * lsb_mv * 1000.0
    else:
        raise ValueError("Unknown unit")


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


def fil(data, fs, f_cut, btype="bandpass"):
    data = np.array(data, dtype=float)
    order = 2
    b, a = butter(order, np.array(f_cut) / (fs / 2), btype=btype)
    return filtfilt(b, a, data)


def detect_r_peaks(data_ecg_raw, ecg_filtered, fs_ecg):
    data_ecg_for_deriv = fil(data_ecg_raw, fs_ecg, [10, 30])
    deriv_ecg = np.convolve(
        data_ecg_for_deriv, np.array([1, 2, 0, -2, -1]) * (fs_ecg / 8.0), mode="same"
    )
    squared_ecg = deriv_ecg**2

    window_size = int(0.15 * fs_ecg)
    cumsum = np.cumsum(np.insert(squared_ecg, 0, 0))
    integrated_ecg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    pad_before = (window_size - 1) // 2
    integrated_ecg = np.pad(integrated_ecg, (pad_before, window_size // 2), mode="edge")

    peaks_collect = []
    window_size2 = 60 * fs_ecg
    for start_idx in range(0, len(ecg_filtered), window_size2):
        end_idx = min(start_idx + window_size2, len(ecg_filtered))
        segment_data = integrated_ecg[start_idx:end_idx]
        if len(segment_data) == 0:
            continue
        cand_peaks, _ = find_peaks(segment_data, distance=int(0.6 * fs_ecg))
        if len(cand_peaks) < 2:
            continue
        height_threshold = 0.1 * np.median(segment_data[cand_peaks])
        avg_interval = np.mean(np.diff(cand_peaks))
        refined_peaks, _ = find_peaks(
            segment_data,
            height=height_threshold,
            distance=max(1, int(0.2 * avg_interval)),
        )
        global_peaks = refined_peaks + start_idx
        half_window = max(1, int(0.1 * avg_interval))
        for gp in global_peaks:
            s_start = max(0, gp - half_window)
            s_end = min(len(ecg_filtered), gp + half_window)
            if s_start >= s_end:
                continue
            local_max_idx = np.argmax(ecg_filtered[s_start:s_end]) + s_start
            peaks_collect.append(local_max_idx)

    peaks = np.unique(peaks_collect)
    if len(peaks) < 2:
        return peaks

    final_peaks = [peaks[0]]
    for next_peak in peaks[1:]:
        if next_peak - final_peaks[-1] > int(0.2 * fs_ecg):
            final_peaks.append(next_peak)
        elif ecg_filtered[next_peak] > ecg_filtered[final_peaks[-1]]:
            final_peaks[-1] = next_peak

    final_peaks_np = np.array(final_peaks)
    if len(final_peaks_np) == 0:
        return final_peaks_np
    peak_amplitudes = ecg_filtered[final_peaks_np]
    thr = 3.5 * np.median(peak_amplitudes) if len(peak_amplitudes) > 0 else 0
    valid_peaks = final_peaks_np[peak_amplitudes <= thr] if thr > 0 else final_peaks_np
    return np.array(sorted(list(set(valid_peaks))))


def calculate_average_beat_atlas(scg_full, r_peaks_indices_ecg, fs_scg, fs_ecg):
    # Calculate average time-frequency atlas for all valid beats
    if len(r_peaks_indices_ecg) < 2:
        return None

    r_peaks_indices_scg = (r_peaks_indices_ecg / fs_ecg * fs_scg).astype(int)
    final_freqs = np.logspace(np.log10(2), np.log10(100), NORMALIZED_FREQ_BINS)
    normalized_beats_sst = []

    for i in range(len(r_peaks_indices_scg) - 1):
        beat_start_idx, beat_end_idx = (
            r_peaks_indices_scg[i],
            r_peaks_indices_scg[i + 1],
        )

        if not (0 <= beat_start_idx < beat_end_idx <= len(scg_full)):
            continue
        single_beat_signal = scg_full[beat_start_idx:beat_end_idx]
        if len(single_beat_signal) < 20:
            continue

        try:
            # Apply Synchrosqueezed Stockwell Transform (SSTF)
            Tx_beat, _, ssq_freqs_beat, *_ = ssq_cwt(
                single_beat_signal,
                wavelet=("gmw", {"gamma": 3, "beta": 20 / 3}),
                scales="log-piecewise",
                fs=fs_scg,
                nv=32,
            )
            abs_Tx_beat = np.abs(Tx_beat)

            # Normalize frequency and resample time
            sort_indices = np.argsort(ssq_freqs_beat)
            interp_Tx_beat = np.apply_along_axis(
                lambda col: np.interp(final_freqs, ssq_freqs_beat[sort_indices], col),
                axis=0,
                arr=abs_Tx_beat[sort_indices, :],
            )

            normalized_sst = zoom(
                interp_Tx_beat,
                (1, NORMALIZED_TIME_POINTS / interp_Tx_beat.shape[1]),
                order=1,
            )
            normalized_beats_sst.append(normalized_sst)
        except Exception as e_beat:
            continue

    if not normalized_beats_sst:
        return None

    average_beat_atlas = np.mean(np.array(normalized_beats_sst), axis=0)
    return average_beat_atlas


def calculate_percentile_threshold(atlas_matrix, percentile=50):
    # Calculate energy threshold for filtering
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

    background_mask = atlas_matrix < threshold
    background_energy_sum = np.sum(atlas_matrix[background_mask])
    foreground_energy_sum = np.sum(atlas_matrix[~background_mask])

    stats = {
        "filter_method": "percentile",
        "percentile": percentile,
        "threshold": threshold,
        "background_energy_sum": background_energy_sum,
        "foreground_energy_sum": foreground_energy_sum,
        "total_energy": total_energy,
        "background_pixels": np.sum(background_mask),
        "foreground_pixels": np.sum(~background_mask),
    }

    return stats


def plot_Fig_2_with_marginals(
    atlas_matrix,
    title_prefix="",
    apply_filter=False,
    bg_stats=None,
    vmax_unified=None,
    annotation_text=None,
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

    if apply_filter:
        pass

    ax_main.set_facecolor("white")

    freq_bins = np.logspace(np.log10(2), np.log10(100), n_freqs)

    ridgeline_indices = np.linspace(0, n_freqs - 1, RIDGELINE_COUNT, dtype=int)

    t_axis = np.linspace(0, 1000, n_times)

    global_max_energy = np.max(normalized_atlas_matrix)

    if isinstance(cmap, str):
        cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap

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

    from matplotlib.patches import Polygon

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
                    line_zorder = 10 + len(freq_order) + 1
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

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

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

    return fig, normalized_atlas_matrix


if __name__ == "__main__":
    if not SSQUEEZEPY_AVAILABLE:
        print("Error: ssqueezepy library not available.")
        exit()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(current_dir, "summary.xlsx")

    if not os.path.exists(summary_file):
        print(f"Error: summary.xlsx not found at {summary_file}")
        exit()

    try:
        df_summary = pd.read_excel(summary_file)
    except Exception as e:
        print(f"Error reading summary file: {e}")
        exit()

    all_atlases = []

    print("Starting processing of file list...")

    for index, row in df_summary.iterrows():
        try:
            txt_filename = row["Source TXT File"]
            time_window_str = row["Time Window"]
            r_peak_timestamps_str = row["R-Peak Timestamps (s)"]
            excel_filename = row["Excel Filename"]

            try:
                t_parts = time_window_str.replace("s", "").split("-")
                xmin = float(t_parts[0])
                xmax = float(t_parts[1])
            except ValueError:
                print(f"  Error parsing time window: {time_window_str}")
                continue

            try:
                r_peak_timestamps = np.array(
                    [
                        float(x.strip())
                        for x in str(r_peak_timestamps_str).split(",")
                        if x.strip()
                    ]
                )
            except ValueError:
                print("  Error parsing R-peak timestamps")
                continue

            file_path = os.path.join(current_dir, txt_filename)
            if not os.path.exists(file_path):
                print(f"  File not found: {txt_filename}")
                continue

            fs_ecg_g, fs_scg_g = 250, 500
            len_ecg, len_accz = [181, 240], [1, 120]
            cut_ecg, cut_accz = [0.5, 100], [2, 100]
            cut_scg = [20, 50]

            try:
                # Load SCG/ECG Data
                data_raw = []
                with open(file_path, "r") as file:
                    for line in file:
                        try:
                            values = [float(x) for x in line.strip().split(",")]
                            if len(values) >= 241:
                                data_raw.append(values[:240])
                        except ValueError:
                            continue
                data_raw = np.array(data_raw)
                if len(data_raw) == 0:
                    print("  No valid data found in file")
                    continue

                data_accz_raw = data_raw[:, len_accz[0] - 1 : len_accz[1]].flatten()
                data_scg_g = fil(data_accz_raw, fs_scg_g, cut_accz, btype="bandpass")

                t_scg = np.arange(len(data_scg_g)) / fs_scg_g

            except Exception as e:
                print(f"  Error loading data: {e}")
                continue

            # Identify valid R-peaks within signal duration
            valid_r_peaks = r_peak_timestamps[
                (r_peak_timestamps >= t_scg[0]) & (r_peak_timestamps <= t_scg[-1])
            ]

            if len(valid_r_peaks) < 2:
                print("  Insufficient R-peaks in signal duration")
                continue

            r_peaks_indices_scg = (valid_r_peaks * fs_scg_g).astype(int)
            r_peaks_indices_ecg_dummy = (
                r_peaks_indices_scg / fs_scg_g * fs_ecg_g
            ).astype(int)

            # Calculate Atlas
            avg_atlas = calculate_average_beat_atlas(
                data_scg_g, r_peaks_indices_ecg_dummy, fs_scg_g, fs_ecg_g
            )

            if avg_atlas is not None:
                # Normalize and collect
                avg_atlas = np.maximum(avg_atlas, 0)
                total_energy = np.sum(avg_atlas)
                normalized_atlas = (
                    avg_atlas / total_energy if total_energy > 1e-9 else avg_atlas
                )
                all_atlases.append(normalized_atlas)
                print("  Atlas calculated and added.")

        except Exception as e:
            print(f"  Error processing row: {e}")
            continue

    if all_atlases:
        print(f"Aggregating {len(all_atlases)} atlases...")
        grand_average_atlas = np.mean(np.array(all_atlases), axis=0)

        filter_percentile = 50.0
        bg_stats = calculate_percentile_threshold(
            grand_average_atlas, percentile=filter_percentile
        )

        pkl_filename = os.path.join(current_dir, "Fig. 2a,2b_plot_data.pkl")
        with open(pkl_filename, "wb") as f:
            pickle.dump(
                {
                    "grand_average_atlas": grand_average_atlas,
                    "percentile": filter_percentile,
                },
                f,
            )
        print(f"Saved plotting data to {pkl_filename}")

        unified_vmax = np.max(grand_average_atlas)

        annotation_text = None

        print("Generating Grand Average Ridge Plot...")
        Fig_2, _ = plot_Fig_2_with_marginals(
            grand_average_atlas,
            apply_filter=True,
            bg_stats=bg_stats,
            vmax_unified=unified_vmax,
            annotation_text=annotation_text,
        )

        if Fig_2:
            output_png = os.path.join(current_dir, "Grand_Average_Ridge_Plot.png")
            Fig_2.savefig(output_png)
            plt.close(Fig_2)
            print(f"Successfully saved plot to {output_png}")
    else:
        print("No atlases were successfully calculated.")
