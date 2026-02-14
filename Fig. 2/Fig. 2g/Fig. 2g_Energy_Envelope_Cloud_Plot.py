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
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import zoom
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback
from datetime import datetime
import pickle

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

try:
    from ssqueezepy import ssq_cwt

    SSQUEEZEPY_AVAILABLE = True
except ImportError:
    print("Error: ssqueezepy library not found. Please run: pip install ssqueezepy")
    exit()

NORMALIZED_TIME_POINTS = 1000
NORMALIZED_FREQ_BINS = 256

FS_SCG = 500
FS_ECG = 250
CUT_SCG_PREPROCESS = [2, 100]
CUT_ECG_PREPROCESS = [0.5, 100]

FILTER_PERCENTILE = 50

OUTPUT_DPI = 300
FIG_WIDTH = 9.2
FIG_HEIGHT = 6.3

SHOW_INDIVIDUAL_CLOUD = True
SHOW_INDIVIDUAL_HIGHLIGHT = False
USE_SD_BAND = True

CLOUD_LINE_WIDTH = 0.3
CLOUD_ALPHA = 0.06
CLOUD_COLOR = "#B0B0B0"

WEEK1_LINE_COLOR = "#2C2C2C"
WEEK1_SD_COLOR = "#BDBBBB"
WEEK1_LINE_WIDTH = 2.0
WEEK1_SD_ALPHA = 0.35

WEEK2_LINE_COLOR = "#3182BD"
WEEK2_LINE_WIDTH = 1.5
WEEK2_LINE_STYLE = "--"

WEEK3_LINE_COLOR = "#E6550D"
WEEK3_LINE_WIDTH = 1.5
WEEK3_LINE_STYLE = ":"

ZONE1_START = 0
ZONE1_END = 30
ZONE1_COLOR = "#F0ADA9"

ZONE2_START = 30
ZONE2_END = 40
ZONE2_COLOR = "#FFFFFF"

ZONE3_START = 40
ZONE3_END = 100
ZONE3_COLOR = "#BFE1F9"

PHASE_BG_ALPHA = 0.3
ZONE_TRANSITION_WIDTH = 10

ER_FREQ_RANGE_SYS = [2, 100]
ER_FREQ_RANGE_DIA = [2, 100]
ER_TIME_RANGE_SYS = [0, 200]
ER_TIME_RANGE_DIA = [0, 300]


def fil(data, fs, f_cut, btype="bandpass"):
    """
    Apply zero-phase bandpass filter using Butterworth design.
    Preserves phase information by filtering forward and backward.
    """
    data = np.array(data, dtype=float)
    order = 2
    b, a = butter(order, np.array(f_cut) / (fs / 2), btype=btype)
    return filtfilt(b, a, data)


def load_and_preprocess_raw_data(filepath):
    """
    Load raw signal data from txt file and apply preprocessing.
    Extracts SCG (120 samples) and ECG (60 samples), applies bandpass filtering.
    """
    print(f"  - Loading: {os.path.basename(filepath)}")
    try:
        data_raw = []
        with open(filepath, "r") as file:
            for line in file:
                try:
                    values = [float(x) for x in line.strip().split(",")]
                    if len(values) >= 240:
                        data_raw.append(values[:240])
                except ValueError:
                    continue

        data_raw = np.array(data_raw)

        # SCG Signal Processing
        data_accz_raw = data_raw[:, 0:120].flatten()
        data_scg_filtered = fil(data_accz_raw, FS_SCG, CUT_SCG_PREPROCESS)

        # ECG Signal Processing
        data_ecg_raw = data_raw[:, 180:240].flatten()
        data_ecg_filtered = fil(data_ecg_raw, FS_ECG, CUT_ECG_PREPROCESS)

        return data_scg_filtered, data_ecg_filtered

    except Exception as e:
        print(f"  - Error: Failed to read file '{filepath}': {e}")
        return None, None


def calculate_average_beat_atlas(scg_full, r_peaks_indices_ecg, fs_scg, fs_ecg):
    """
    Compute average beat time-frequency atlas using synchrosqueezed CWT.
    Extracts individual beats, applies SSQ-CWT, normalizes to standard grid.
    Returns average atlas and list of individual normalized atlases.
    """
    if len(r_peaks_indices_ecg) < 2:
        return None, None

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
            Tx_beat, _, ssq_freqs_beat, *_ = ssq_cwt(
                single_beat_signal,
                wavelet=("gmw", {"gamma": 3, "beta": 20 / 3}),
                scales="log-piecewise",
                fs=fs_scg,
                nv=32,
            )
            abs_Tx_beat = np.abs(Tx_beat)
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
        except Exception:
            continue

    if not normalized_beats_sst:
        return None, None

    average_beat_atlas = np.mean(np.array(normalized_beats_sst), axis=0)
    return average_beat_atlas, normalized_beats_sst


def calculate_1d_energy_envelope(atlas_matrix):
    """
    Calculate 1D energy envelope from 2D time-frequency atlas.
    Sums energy across all frequencies for each time point.
    """
    energy_envelope = np.sum(atlas_matrix, axis=0)
    return energy_envelope


def calculate_er_ratio(atlas_matrix, freq_axis=None):
    """
    Calculate energy ratio (ER) reflecting cardiac mechanical efficiency.
    ER = E_systolic / E_diastolic based on frequency and time ranges.
    """
    if freq_axis is None:
        freq_axis = np.logspace(np.log10(2), np.log10(100), atlas_matrix.shape[0])

    n_freqs, n_times = atlas_matrix.shape

    time_axis = np.linspace(0, 1000, n_times)

    sys_time_mask = (time_axis >= ER_TIME_RANGE_SYS[0]) & (
        time_axis < ER_TIME_RANGE_SYS[1]
    )
    dia_time_mask = (time_axis >= ER_TIME_RANGE_DIA[0]) & (
        time_axis < ER_TIME_RANGE_DIA[1]
    )

    sys_freq_mask = (freq_axis >= ER_FREQ_RANGE_SYS[0]) & (
        freq_axis <= ER_FREQ_RANGE_SYS[1]
    )
    dia_freq_mask = (freq_axis >= ER_FREQ_RANGE_DIA[0]) & (
        freq_axis <= ER_FREQ_RANGE_DIA[1]
    )

    e_sys = np.sum(atlas_matrix[np.ix_(sys_freq_mask, sys_time_mask)])

    e_dia = np.sum(atlas_matrix[np.ix_(dia_freq_mask, dia_time_mask)])

    e_total = np.sum(atlas_matrix)

    # Calculate ER
    if e_total > 0:
        er = e_sys / e_dia
    else:
        er = 0

    return er, e_sys, e_dia, e_total


def calculate_percentile_threshold(atlas_matrix, percentile=50):
    """
    Calculate energy-based threshold (accumulating from lowest energy to X% of total energy)

    Args:
    - atlas_matrix: Energy atlas matrix
    - percentile: Filter percentile (filter out lowest X% of cumulative energy)

    Returns:
    - threshold: Energy threshold
    """
    # Flatten matrix and sort (low to high)
    flattened = atlas_matrix.flatten()
    sorted_energy = np.sort(flattened)

    # Calculate cumulative energy
    cumulative_energy = np.cumsum(sorted_energy)
    total_energy = np.sum(atlas_matrix)

    # Find position where cumulative energy reaches X% of total energy
    target_energy = total_energy * (percentile / 100.0)

    # Find first index where cumulative energy exceeds target
    threshold_idx = np.searchsorted(cumulative_energy, target_energy)

    # Get energy value at this position as threshold
    if threshold_idx < len(sorted_energy):
        threshold = sorted_energy[threshold_idx]
    else:
        threshold = sorted_energy[-1]

    return threshold


# ==============================
#        Data Processing Functions
# ==============================


def process_excel_data(df, base_dir):
    """
    Process Excel data and average R1/R2/R3 replicates for each txt file.
    Groups records by txt filename, loads raw data, computes beat atlases,
    normalizes and filters energy, calculates ER, returns averaged results.
    """
    print(f"\n{'=' * 60}")
    print("Processing Excel data...")
    print(f"{'=' * 60}")

    grouped_data = {}

    for index, row in df.iterrows():
        try:
            source_txt = row["Source TXT File"]
            identifier = str(row.get("ID", ""))
            r_peaks_str = row.get("R-Peak Timestamps (s)", "")

            base_txt_name = os.path.basename(source_txt)

            if base_txt_name not in grouped_data:
                grouped_data[base_txt_name] = {
                    "source_txt": source_txt,
                    "records": [],
                    "identifiers": [],
                }

            grouped_data[base_txt_name]["records"].append(
                {"identifier": identifier, "r_peaks_str": r_peaks_str, "row": row}
            )
            grouped_data[base_txt_name]["identifiers"].append(identifier)

        except Exception as e:
            print(f"  - Error processing row {index}: {e}")
            continue

    print(f"  - Found {len(grouped_data)} unique txt files")
    results = {}
    data_cache = {}

    for txt_name, group_info in grouped_data.items():
        print(f"\nProcessing: {txt_name}")
        print(
            f"  - Contains {len(group_info['records'])} records: {group_info['identifiers']}"
        )

        source_txt_path = os.path.join(base_dir, group_info["source_txt"])

        if not os.path.exists(source_txt_path):
            print(f"  - ! File not found: {source_txt_path}")
            continue

        # Load raw data (using cache)
        if source_txt_path not in data_cache:
            scg_full, ecg_full = load_and_preprocess_raw_data(source_txt_path)
            if scg_full is None:
                continue
            data_cache[source_txt_path] = (scg_full, ecg_full)
        else:
            scg_full, ecg_full = data_cache[source_txt_path]

        # Calculate energy atlas for each record, then average
        atlases = []
        envelopes = []
        ers = []

        for record in group_info["records"]:
            r_peaks_str = record["r_peaks_str"]

            if not r_peaks_str or pd.isna(r_peaks_str):
                continue

            try:
                r_peaks_times = np.array(
                    [float(t) for t in str(r_peaks_str).split(",") if t.strip()]
                )
                r_peak_indices = (r_peaks_times * FS_ECG).astype(int)

                # Calculate average cardiac cycle atlas
                avg_atlas, _ = calculate_average_beat_atlas(
                    scg_full, r_peak_indices, FS_SCG, FS_ECG
                )

                if avg_atlas is not None:
                    # Normalize
                    total_energy = np.sum(avg_atlas)
                    if total_energy > 1e-9:
                        normalized_atlas = avg_atlas / total_energy
                        atlases.append(normalized_atlas)

                        # Calculate 1D energy envelope
                        envelope = calculate_1d_energy_envelope(normalized_atlas)
                        envelopes.append(envelope)

                        # Calculate ER
                        er, e_sys, e_dia, e_total = calculate_er_ratio(normalized_atlas)
                        ers.append(er)

            except Exception as e:
                print(f"    - Error processing record {record['identifier']}: {e}")
                continue

        if atlases:
            # Calculate average
            avg_atlas_final = np.mean(np.array(atlases), axis=0)

            # Normalize averaged atlas
            total_energy = np.sum(avg_atlas_final)
            if total_energy > 1e-9:
                avg_atlas_normalized = avg_atlas_final / total_energy
            else:
                avg_atlas_normalized = avg_atlas_final

            # Filter normalized atlas
            threshold = calculate_percentile_threshold(
                avg_atlas_normalized, percentile=FILTER_PERCENTILE
            )
            avg_atlas_filtered = avg_atlas_normalized.copy()
            avg_atlas_filtered[avg_atlas_filtered < threshold] = 0

            # Calculate 1D energy envelope (using normalized + filtered atlas)
            avg_envelope_final = calculate_1d_energy_envelope(avg_atlas_filtered)

            # Calculate ER (using normalized + filtered atlas)
            avg_er_final = np.mean(ers)  # ER still uses average of individual records

            results[txt_name] = {
                "atlas": avg_atlas_filtered,  # Store normalized + filtered atlas
                "envelope": avg_envelope_final,  # Normalized + filtered envelope
                "er": avg_er_final,
                "n_records": len(atlases),
                "identifiers": group_info["identifiers"],
            }

            print(
                f"  - [OK] Success: Averaged {len(atlases)} records, ER = {avg_er_final:.3f}"
            )
            print(
                f"    - Using normalized + filtered ({FILTER_PERCENTILE}%) energy values"
            )
        else:
            print(f"  - [X] No valid data")

    return results


def identify_test_retest_pairs(results, subject_prefix="TRR"):
    """
    Identify Test-Retest pairs for same subject across different weeks.
    Parses filename format (TRR001_1_08_...), groups by subject and week,
    computes population averages with Mean+/-SD and 95% CI bands.
    """
    subjects = {}
    weeks_data = {1: [], 2: [], 3: []}

    for txt_name, data in results.items():
        match = re.match(r"(TRR\d+)_(\d+)_", txt_name)
        if match:
            subject_id = match.group(1)
            week_num = int(match.group(2))

            if subject_id not in subjects:
                subjects[subject_id] = {}

            subjects[subject_id][week_num] = {"txt_name": txt_name, "data": data}

            if week_num in weeks_data:
                weeks_data[week_num].append(data)

    pairs = []
    for subject_id, weeks in subjects.items():
        if len(weeks) >= 2:
            pairs.append({"subject_id": subject_id, "weeks": weeks})

    population_weeks = {}
    for week_num, data_list in weeks_data.items():
        if data_list:
            envelopes = np.array([d["envelope"] for d in data_list])
            atlases = [d["atlas"] for d in data_list]
            ers = []
            for atlas in atlases:
                er, _, _, _ = calculate_er_ratio(atlas)
                ers.append(er)

            avg_envelope = np.mean(envelopes, axis=0)
            std_envelope = np.std(envelopes, axis=0)

            n = len(data_list)

            sd_lower = avg_envelope - std_envelope
            sd_upper = avg_envelope + std_envelope
            sd_lower = np.maximum(sd_lower, 0)

            sem = std_envelope / np.sqrt(n)
            ci_lower = avg_envelope - 1.96 * sem
            ci_upper = avg_envelope + 1.96 * sem
            ci_lower = np.maximum(ci_lower, 0)

            avg_er = np.mean(ers)
            std_er = np.std(ers)

            population_weeks[week_num] = {
                "envelope": avg_envelope,
                "envelope_std": std_envelope,
                "sd_lower": sd_lower,
                "sd_upper": sd_upper,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "er": avg_er,
                "er_std": std_er,
                "n_subjects": n,
            }

    print(f"\nFound {len(pairs)} subjects with multi-week data")
    for pair in pairs:
        print(f"  - {pair['subject_id']}: Week {list(pair['weeks'].keys())}")

    print(f"\nPopulation average data:")
    for week_num, data in population_weeks.items():
        print(
            f"  - Week {week_num}: n={data['n_subjects']}, ER={data['er']:.3f}+/-{data['er_std']:.3f}"
        )

    return pairs, population_weeks


# ==============================
#        Plotting Functions
# ==============================


def plot_cloud_and_lightning(
    results,
    highlight_subject=None,
    population_weeks=None,
    output_path=None,
    show_plot=True,
    title=None,
):
    """
    Generate Cloud & Lightning plot with mist and neon design.
    Layers: anatomical zones, individual cloud, population averages.
    Outputs: PNG (300 DPI), SVG (vector), PDF (vector).
    """
    print(f"\n{'=' * 60}")
    print("Generating Panel g: Cloud & Lightning")
    print(f"{'=' * 60}")

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

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=OUTPUT_DPI)

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    x_axis = np.linspace(0, 100, NORMALIZED_TIME_POINTS)

    all_envelopes = []
    for txt_name, data in results.items():
        all_envelopes.append(data["envelope"])

    if all_envelopes:
        all_data = np.concatenate([env.flatten() for env in all_envelopes])
        y_max = (
            np.percentile(all_data[all_data > 0], 99) * 1.15
            if np.any(all_data > 0)
            else 1.0
        )
    else:
        y_max = 1.0

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

    # Plot core part of each zone (excluding transition)
    for i, zone in enumerate(zones):
        # Calculate core region (remove transition)
        core_start = zone["start"]
        core_end = zone["end"]

        # If not the first zone, core region starts after transition
        if i > 0:
            core_start = zone["start"] + ZONE_TRANSITION_WIDTH / 2

        # If not the last zone, core region ends before transition
        if i < len(zones) - 1:
            core_end = zone["end"] - ZONE_TRANSITION_WIDTH / 2

        # Ensure core region is valid
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

        # Start and end positions of transition zone
        transition_start = zone_current["end"] - ZONE_TRANSITION_WIDTH / 2
        transition_end = zone_next["start"] + ZONE_TRANSITION_WIDTH / 2

        # Ensure transition zone is valid
        if transition_start < transition_end:
            # Use multiple thin strips to simulate gradient
            n_steps = max(
                10, int(ZONE_TRANSITION_WIDTH * 2)
            )  # Number of gradient steps
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

    print(f"    - Zone 1: {ZONE1_START}-{ZONE1_END}%, Color: {ZONE1_COLOR}")
    print(f"    - Zone 2: {ZONE2_START}-{ZONE2_END}%, Color: {ZONE2_COLOR}")
    print(f"    - Zone 3: {ZONE3_START}-{ZONE3_END}%, Color: {ZONE3_COLOR}")
    print(f"    - Transition width: {ZONE_TRANSITION_WIDTH}%")

    if SHOW_INDIVIDUAL_CLOUD:
        print("  - Drawing background mist (Energy Cloud)...")
        print(f"    - Parameters: line_width={CLOUD_LINE_WIDTH}, alpha={CLOUD_ALPHA}")

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
    else:
        print("  - Skipping background mist (SHOW_INDIVIDUAL_CLOUD=False)")

    if population_weeks:
        print("  - Drawing foreground neon (Neon Lines)...")
        print(
            f"    - Using {'Mean+/-SD (normal corridor)' if USE_SD_BAND else '95% CI'}"
        )

        if 1 in population_weeks:
            week1_data = population_weeks[1]
            envelope = week1_data["envelope"]
            n = week1_data["n_subjects"]
            er = week1_data["er"]
            er_std = week1_data["er_std"]

            if USE_SD_BAND:
                band_lower = week1_data["sd_lower"]
                band_upper = week1_data["sd_upper"]
                band_label = "+/-1SD"
            else:
                band_lower = week1_data["ci_lower"]
                band_upper = week1_data["ci_upper"]
                band_label = "95%CI"

            ax.fill_between(
                x_axis,
                band_lower,
                band_upper,
                color=WEEK1_SD_COLOR,
                alpha=WEEK1_SD_ALPHA,
                edgecolor="none",
                zorder=2,
                label=f"Week 1 {band_label}",
            )

            ax.plot(
                x_axis,
                envelope,
                color=WEEK1_LINE_COLOR,
                linewidth=WEEK1_LINE_WIDTH,
                linestyle="-",
                alpha=1.0,
                zorder=5,
                label=f"Week 1 Mean (n={n})",
            )

            print(f"    - Week 1 (baseline): n={n}, ER={er:.3f}+/-{er_std:.3f}")

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
                label=f"Week 2 (n={n})",
            )

            print(f"    - Week 2 (follow-up): n={n}, ER={er:.3f}+/-{er_std:.3f}")

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
                label=f"Week 3 (n={n})",
            )

            print(f"    - Week 3 (follow-up): n={n}, ER={er:.3f}+/-{er_std:.3f}")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, y_max)

    ax.set_xlabel("Normalized Cardiac Cycle (%)", fontweight="normal")
    ax.set_ylabel("Energy (a.u.)", fontweight="normal")

    if title:
        ax.set_title(title, fontweight="bold", pad=10)
    else:
        n_subjects = len(results)
        ax.set_title(
            f"Population Energy Envelope (n={n_subjects})", fontweight="bold", pad=10
        )

    ax.set_xticks([0, 20, 40, 60, 80, 100])

    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))
    ax.yaxis.get_offset_text().set_fontsize(8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    zone_centers = [
        (ZONE1_START + ZONE1_END) / 2,
        (ZONE2_START + ZONE2_END) / 2,
        (ZONE3_START + ZONE3_END) / 2,
    ]
    zone_labels = ["Zone 1", "Zone 2", "Zone 3"]

    for center, label in zip(zone_centers, zone_labels):
        ax.text(
            center,
            y_max * 0.92,
            label,
            ha="center",
            fontsize=9,
            color="#616161",
            fontweight="medium",
        )

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

    if output_path:
        fig.savefig(
            output_path,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"  - Saved: {output_path}")

        svg_path = output_path.replace(".png", ".svg")
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"  - Saved: {svg_path}")

        pdf_path = output_path.replace(".png", ".pdf")
        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"  - Saved: {pdf_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ==============================
#        GUI Selection Dialog
# ==============================


def select_highlight_subject_dialog(pairs):
    """
    Show Tkinter dialog for user to select plot options.
    Allows toggling individual cloud, population data, and statistical method.
    """
    dialog = tk.Toplevel()
    dialog.title("Panel g Plot Options")
    dialog.geometry("600x600")

    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (210)
    y = (dialog.winfo_screenheight() // 2) - (160)
    dialog.geometry(f"+{x}+{y}")

    main_frame = tk.Frame(dialog, padx=20, pady=15)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Title
    tk.Label(
        main_frame, text="Panel g: Mist + Neon Plot Options", font=("Arial", 12, "bold")
    ).pack(pady=(0, 15))

    # === Background Mist Options ===
    tk.Label(
        main_frame, text="Background Mist (Mist):", font=("Arial", 10, "bold")
    ).pack(anchor="w")

    show_cloud_var = tk.BooleanVar(value=SHOW_INDIVIDUAL_CLOUD)
    tk.Checkbutton(
        main_frame,
        text="Show individual data background cloud",
        variable=show_cloud_var,
        font=("Arial", 9),
    ).pack(anchor="w", padx=15, pady=2)

    # === Foreground Neon Options ===
    tk.Label(
        main_frame, text="Foreground Neon (Neon):", font=("Arial", 10, "bold")
    ).pack(anchor="w", pady=(10, 0))

    show_population_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        main_frame,
        text="Show population average Week 1/2/3 lines",
        variable=show_population_var,
        font=("Arial", 9),
    ).pack(anchor="w", padx=15, pady=2)

    # === Statistics Method Selection ===
    tk.Label(
        main_frame, text="Week 1 Band Statistics Method:", font=("Arial", 10, "bold")
    ).pack(anchor="w", pady=(10, 0))

    stat_method_var = tk.StringVar(value="SD" if USE_SD_BAND else "CI")

    sd_radio = tk.Radiobutton(
        main_frame,
        text="Mean +/- SD (Normal Corridor, Recommended)",
        variable=stat_method_var,
        value="SD",
        font=("Arial", 9),
    )
    sd_radio.pack(anchor="w", padx=15, pady=2)

    ci_radio = tk.Radiobutton(
        main_frame,
        text="95% CI (Confidence Interval)",
        variable=stat_method_var,
        value="CI",
        font=("Arial", 9),
    )
    ci_radio.pack(anchor="w", padx=15, pady=2)

    # Hint Information
    tk.Label(
        main_frame,
        text="Hint: SD band reflects true data dispersion, CI band reflects mean precision",
        font=("Arial", 8),
        fg="gray",
    ).pack(pady=(10, 0))

    result = {"confirmed": False}

    def confirm():
        global SHOW_INDIVIDUAL_CLOUD, USE_SD_BAND
        SHOW_INDIVIDUAL_CLOUD = show_cloud_var.get()
        USE_SD_BAND = stat_method_var.get() == "SD"

        result["confirmed"] = True
        result["show_cloud"] = show_cloud_var.get()
        result["show_population"] = show_population_var.get()
        result["use_sd"] = USE_SD_BAND
        dialog.destroy()

    def cancel():
        result["confirmed"] = False
        dialog.destroy()

    # Buttons
    btn_frame = tk.Frame(main_frame)
    btn_frame.pack(pady=15)

    tk.Button(
        btn_frame, text="Confirm", command=confirm, width=10, font=("Arial", 10)
    ).pack(side=tk.LEFT, padx=10)
    tk.Button(
        btn_frame, text="Cancel", command=cancel, width=10, font=("Arial", 10)
    ).pack(side=tk.LEFT, padx=10)

    dialog.transient()
    dialog.grab_set()
    dialog.wait_window()

    return result if result["confirmed"] else None


# ==============================
#        Main Function
# ==============================


def main():
    """
    Main workflow: Load Excel, process data, identify pairs, generate plot.
    Saves PNG/SVG/PDF outputs and pickle data file with all parameters.
    """
    print("\n" + "=" * 60)
    print("Panel g: The Cloud & The Lightning")
    print("=" * 60)

    root = tk.Tk()
    root.withdraw()

    print("\nSelect master_summary.xlsx file...")
    excel_path = filedialog.askopenfilename(
        title="Select master_summary.xlsx file",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
    )

    if not excel_path:
        print("No file selected, exiting.")
        return

    print(f"  - Selected: {excel_path}")

    base_dir = os.path.dirname(excel_path)

    print("\nReading Excel file...")
    try:
        df = pd.read_excel(excel_path)
        print(f"  - Total {len(df)} records")
        print(f"  - Columns: {list(df.columns)}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read Excel file: {e}")
        return

    results = process_excel_data(df, base_dir)

    if not results:
        messagebox.showerror("Error", "No data processed successfully!")
        return

    print(f"\nSuccessfully processed {len(results)} txt files")

    pairs, population_weeks = identify_test_retest_pairs(results)

    plot_options = select_highlight_subject_dialog(pairs)

    if not plot_options:
        plot_options = {
            "show_cloud": SHOW_INDIVIDUAL_CLOUD,
            "show_population": True,
            "show_individual": False,
        }

    print("\nSelect output directory...")
    output_dir = filedialog.askdirectory(title="Select output directory")

    if not output_dir:
        output_dir = base_dir
        print(f"  - Using default directory: {output_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"Fig. 2g_Cloud_Lightning_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)

    highlight_subject = None
    pop_weeks_to_show = (
        population_weeks if plot_options.get("show_population", True) else None
    )

    plot_cloud_and_lightning(
        results,
        highlight_subject=highlight_subject,
        population_weeks=pop_weeks_to_show,
        output_path=output_path,
        show_plot=True,
    )

    print("\n" + "=" * 60)
    print("Saving plot data...")
    print("=" * 60)

    results_to_save = {}
    for txt_name, data in results.items():
        results_to_save[txt_name] = {k: v for k, v in data.items() if k != "atlas"}
    print("  - Optimized data structure (removed 2D atlas, kept 1D envelope)")

    plot_data = {
        "results": results_to_save,
        "population_weeks": population_weeks,
        "plot_options": plot_options,
        "plot_params": {
            "NORMALIZED_TIME_POINTS": NORMALIZED_TIME_POINTS,
            "NORMALIZED_FREQ_BINS": NORMALIZED_FREQ_BINS,
            "FILTER_PERCENTILE": FILTER_PERCENTILE,
            "FIG_WIDTH": FIG_WIDTH,
            "FIG_HEIGHT": FIG_HEIGHT,
            "OUTPUT_DPI": OUTPUT_DPI,
            "CLOUD_LINE_WIDTH": CLOUD_LINE_WIDTH,
            "CLOUD_ALPHA": CLOUD_ALPHA,
            "CLOUD_COLOR": CLOUD_COLOR,
            "WEEK1_LINE_COLOR": WEEK1_LINE_COLOR,
            "WEEK1_SD_COLOR": WEEK1_SD_COLOR,
            "WEEK1_LINE_WIDTH": WEEK1_LINE_WIDTH,
            "WEEK1_SD_ALPHA": WEEK1_SD_ALPHA,
            "WEEK2_LINE_COLOR": WEEK2_LINE_COLOR,
            "WEEK2_LINE_WIDTH": WEEK2_LINE_WIDTH,
            "WEEK2_LINE_STYLE": WEEK2_LINE_STYLE,
            "WEEK3_LINE_COLOR": WEEK3_LINE_COLOR,
            "WEEK3_LINE_WIDTH": WEEK3_LINE_WIDTH,
            "WEEK3_LINE_STYLE": WEEK3_LINE_STYLE,
            "ZONE1_START": ZONE1_START,
            "ZONE1_END": ZONE1_END,
            "ZONE1_COLOR": ZONE1_COLOR,
            "ZONE2_START": ZONE2_START,
            "ZONE2_END": ZONE2_END,
            "ZONE2_COLOR": ZONE2_COLOR,
            "ZONE3_START": ZONE3_START,
            "ZONE3_END": ZONE3_END,
            "ZONE3_COLOR": ZONE3_COLOR,
            "PHASE_BG_ALPHA": PHASE_BG_ALPHA,
            "ZONE_TRANSITION_WIDTH": ZONE_TRANSITION_WIDTH,
        },
        "metadata": {
            "timestamp": timestamp,
            "excel_path": excel_path,
            "n_subjects": len(results),
            "n_pairs": len(pairs) if pairs else 0,
        },
    }

    data_filename = f"Fig. 2g_Plot_Data.pkl"
    data_path = os.path.join(output_dir, data_filename)

    try:
        with open(data_path, "wb") as f:
            pickle.dump(plot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  - Plot data saved: {data_path}")
        print(f"  - File size: {os.path.getsize(data_path) / 1024:.2f} KB")
    except Exception as e:
        print(f"  - Error saving data: {e}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

    messagebox.showinfo(
        "Complete", f"Plot saved to:\n{output_path}\n\nPlot data saved to:\n{data_path}"
    )


if __name__ == "__main__":
    main()
