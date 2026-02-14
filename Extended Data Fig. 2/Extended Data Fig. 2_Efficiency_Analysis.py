# -*- coding: utf-8 -*-
"""
==============================================================================
Efficiency Index Analysis Tool - Extended Data Fig. 2

Data Privacy and Reproducibility Statement:
Due to the patient privacy and data sensitivity of the original clinical data
(such as high-precision ECG/SCG waveforms), and in accordance with ethical review
and privacy protection regulations, the original dataset cannot be publicly shared.
To ensure the transparency and reproducibility of the research results, we have provided
the original analysis code and the code for generating the figures consistent with
the paper using the intermediate result files (.pkl) after processing and feature
extraction.

Function Overview:
This module implements a complete efficiency index analysis tool to find the optimal balance
point between pixel-level significance and energy threshold. It calculates pixel-level
significance of time-frequency atlases using Mann-Whitney U test, then scans different
energy thresholds to calculate efficiency index (Retained Area x Significance Improvement)
for each threshold, finally finding the optimal filtering threshold.

Main Process:
1. Load raw ECG/SCG data and calculate average cardiac cycle time-frequency atlas
2. Perform pixel-level significance analysis on two sample groups (Mann-Whitney U test)
3. Scan energy threshold range from 0-99%
4. Calculate efficiency index for each threshold
5. Plot efficiency curve and mark optimal threshold

Core Dependencies:
- scipy.stats.mannwhitneyu: Statistical significance test
- ssqueezepy.ssq_cwt: Synchrosqueezing Transform
- tkinter: GUI interaction
- matplotlib: Data visualization

==============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import zoom
from scipy.stats import mannwhitneyu
from ssqueezepy import ssq_cwt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# Configure font support
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")

# ============================================================================
# Core Parameter Configuration
# ============================================================================
NORMALIZED_TIME_POINTS = 1000  # Time axis normalized resolution
NORMALIZED_FREQ_BINS = 256  # Frequency axis normalized resolution
FS_SCG = 500  # SCG sampling rate (Hz)
FS_ECG = 250  # ECG sampling rate (Hz)
CUT_SCG_PREPROCESS = [2, 100]  # SCG bandpass filter range (Hz)


# ============================================================================
# Data Processing Helper Functions
# ============================================================================


def fil(data, fs, f_cut, btype="bandpass"):
    """
    Zero-phase Digital Filter

    Use Butterworth filter and filtfilt to implement zero-phase distortion filtering.

    Args:
        data: Input signal (array)
        fs: Sampling frequency (Hz)
        f_cut: Cutoff frequency or [low, high] (Hz)
        btype: Filter type - "bandpass", "highpass" or "lowpass"

    Returns:
        Filtered signal (zero-phase distortion)
    """
    data = np.array(data, dtype=float)
    order = 2
    b, a = butter(order, np.array(f_cut) / (fs / 2), btype=btype)
    return filtfilt(b, a, data)


def load_and_preprocess_raw_data(filepath):
    """
    Load raw data file and preprocess SCG signal

    Extract SCG (Z-axis acceleration) signal from CSV format raw data file,
    and apply bandpass filtering for preprocessing.

    Args:
        filepath: Raw data file path (.txt format)

    Returns:
        Preprocessed SCG signal (array) or None (if loading fails)
    """
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
        # Extract Z-axis acceleration (SCG signal)
        data_accz_raw = data_raw[:, 0:120].flatten()
        data_scg_filtered = fil(data_accz_raw, FS_SCG, CUT_SCG_PREPROCESS)
        return data_scg_filtered
    except Exception as e:
        return None


def calculate_average_beat_atlas(scg_full, r_peaks_indices_ecg, fs_scg, fs_ecg):
    """
    Calculate Average Cardiac Cycle Time-Frequency Atlas

    Extract cardiac cycles between R-waves from full SCG signal, apply Synchrosqueezing
    Transform (SSTF) to each cycle, then average time-frequency representations of all cycles.

    Args:
        scg_full: Full SCG signal (array)
        r_peaks_indices_ecg: R-wave indices in ECG (array)
        fs_scg: SCG sampling rate (Hz)
        fs_ecg: ECG sampling rate (Hz)

    Returns:
        Normalized average time-frequency atlas (2D array) or None (if insufficient data)

    Process Flow:
        1. Convert ECG R-wave indices to SCG indices
        2. Extract cardiac cycles between R-waves
        3. Apply SSTF transform to each cycle
        4. Normalize frequency axis to logarithmic scale (2-100Hz)
        5. Resample time axis to standard resolution
        6. Average all cycles and normalize
    """
    if len(r_peaks_indices_ecg) < 2:
        return None

    r_peaks_indices_scg = (r_peaks_indices_ecg / fs_ecg * fs_scg).astype(int)
    final_freqs = np.logspace(np.log10(2), np.log10(100), NORMALIZED_FREQ_BINS)
    normalized_beats_sst = []

    # Extract and process each cardiac cycle
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
            # Apply SSTF transform to extract time-frequency representation
            Tx_beat, _, ssq_freqs_beat, *_ = ssq_cwt(
                single_beat_signal,
                wavelet=("gmw", {"gamma": 3, "beta": 20 / 3}),
                scales="log-piecewise",
                fs=fs_scg,
                nv=32,
            )
            abs_Tx_beat = np.abs(Tx_beat)

            # Normalize frequency axis to standard grid
            sort_indices = np.argsort(ssq_freqs_beat)
            interp_Tx_beat = np.apply_along_axis(
                lambda col: np.interp(final_freqs, ssq_freqs_beat[sort_indices], col),
                axis=0,
                arr=abs_Tx_beat[sort_indices, :],
            )

            # Resample time axis to standard resolution
            normalized_sst = zoom(
                interp_Tx_beat,
                (1, NORMALIZED_TIME_POINTS / interp_Tx_beat.shape[1]),
                order=1,
            )
            normalized_beats_sst.append(normalized_sst)
        except Exception:
            continue

    if not normalized_beats_sst:
        return None

    # Average all beats and normalize
    average_beat_atlas = np.mean(np.array(normalized_beats_sst), axis=0)
    total_energy = np.sum(average_beat_atlas)
    if total_energy > 0:
        average_beat_atlas /= total_energy

    return average_beat_atlas


# ============================================================================
# Core Analysis Logic
# ============================================================================


def analyze_efficiency(df, base_dir, label_col):
    """
    Execute pixel-level significance and energy threshold efficiency analysis

    This function is the core of the analysis process, containing 4 main steps:
    1. Load time-frequency atlases of all samples
    2. Calculate pixel-level significance (Mann-Whitney U test)
    3. Calculate global average energy map
    4. Scan thresholds and calculate efficiency index

    Args:
        df: DataFrame containing sample information
        base_dir: Base directory of data files
        label_col: Label column name used for grouping

    Returns:
        (Result DataFrame, Label A, Label B) or None (if analysis fails)
    """

    # ========== Step 1: Load all sample time-frequency maps ==========
    print("\n[Step 1] Loading data and generating time-frequency atlases...")

    atlases_group_a = []
    atlases_group_b = []

    # Get classification labels
    unique_labels = df[label_col].unique()
    if len(unique_labels) != 2:
        print(
            f"Error: Label column must contain exactly 2 categories, current: {unique_labels}"
        )
        return None

    label_a = unique_labels[0]
    label_b = unique_labels[1]
    print(f"   Group A: {label_a}")
    print(f"   Group B: {label_b}")

    success_count = 0
    total_count = len(df)

    for index, row in df.iterrows():
        try:
            # Handle file paths (compatible with relative paths)
            source_txt = row["Source TXT File"]
            txt_path = os.path.join(base_dir, source_txt)
            if not os.path.exists(txt_path):
                parent_dir = os.path.dirname(base_dir)
                txt_path_parent = os.path.join(parent_dir, source_txt)
                if os.path.exists(txt_path_parent):
                    txt_path = txt_path_parent
                else:
                    continue

            scg_signal = load_and_preprocess_raw_data(txt_path)
            if scg_signal is None:
                continue

            r_peaks_str = row["R-Peak Timestamps (s)"]
            r_peaks = np.array([float(t) for t in r_peaks_str.split(",") if t])
            r_peak_indices = (r_peaks * FS_ECG).astype(int)

            atlas = calculate_average_beat_atlas(
                scg_signal, r_peak_indices, FS_SCG, FS_ECG
            )
            if atlas is None:
                continue

            # Store grouped by label
            current_label = row[label_col]
            if current_label == label_a:
                atlases_group_a.append(atlas)
            elif current_label == label_b:
                atlases_group_b.append(atlas)

            success_count += 1
            if success_count % 10 == 0:
                print(f"   Loaded {success_count} / {total_count} files...")

        except Exception as e:
            continue

    print(
        f"[OK] Data loading complete. Group A: {len(atlases_group_a)}, Group B: {len(atlases_group_b)}"
    )

    if len(atlases_group_a) < 5 or len(atlases_group_b) < 5:
        print("Error: insufficient samples in one group (<5).")
        return None

    # Stack into 3D array: (samples, freq, time)
    stack_a = np.array(atlases_group_a)
    stack_b = np.array(atlases_group_b)

    # ========== Step 2: Calculate pixel-level significance ==========
    print("\n[Step 2] Calculating pixel-level significance (Mann-Whitney U Test)...")

    n_freq, n_time = stack_a.shape[1], stack_a.shape[2]
    p_values_map = np.ones((n_freq, n_time))

    # Try vectorized calculation (faster)
    try:
        _, p_matrix = mannwhitneyu(stack_a, stack_b, axis=0, alternative="two-sided")
        p_values_map = p_matrix
        print("   [OK] Vectorized Mann-Whitney U calculation successful.")
    except:
        print("   ! Vectorized calculation not supported, using loop (slower)...")
        for f in range(n_freq):
            if f % 20 == 0:
                print(f"   Processing frequency component {f}/{n_freq}...")
            for t in range(n_time):
                _, p = mannwhitneyu(
                    stack_a[:, f, t], stack_b[:, f, t], alternative="two-sided"
                )
                p_values_map[f, t] = p

    # Logarithmic transformation: -log10(p-value)
    epsilon = 1e-300
    p_values_map = np.maximum(p_values_map, epsilon)
    significance_map = -np.log10(p_values_map)

    # ========== Step 3: Global Average Energy Map ==========
    print("\n[Step 3] Calculating global average energy map...")
    all_atlases = np.concatenate([stack_a, stack_b], axis=0)
    grand_average_energy = np.mean(all_atlases, axis=0)

    # Normalize total energy to 1
    grand_average_energy /= np.sum(grand_average_energy)

    # ========== Step 4: Scan Thresholds and Calculate Efficiency Index ==========
    print("\n[Step 4] Scanning thresholds and calculating efficiency index...")

    flattened_energy = np.sort(grand_average_energy.flatten())
    cumulative_energy_dist = np.cumsum(flattened_energy)
    total_energy_sum = cumulative_energy_dist[-1]

    metrics = []
    threshold_percentages = np.arange(0, 99, 1)

    # Calculate baseline (mean significance at 0% filtering)
    baseline_mask = grand_average_energy > 0
    baseline_mean_sig = np.mean(significance_map[baseline_mask])

    print(f"   Baseline mean significance (0% filtering): {baseline_mean_sig:.4f}")

    for p in threshold_percentages:
        # p is the percentage of energy "filtered out"
        target_sum = total_energy_sum * (p / 100.0)
        cutoff_idx = np.searchsorted(cumulative_energy_dist, target_sum)
        cutoff_value = flattened_energy[cutoff_idx]

        mask = grand_average_energy > cutoff_value
        remaining_pixels = np.sum(mask)
        # Area ratio: Proportion relative to total pixels
        area_ratio = remaining_pixels / grand_average_energy.size

        if remaining_pixels > 0:
            current_mean_sig = np.mean(significance_map[mask])

            # Calculate significance improvement relative to baseline
            sig_improvement = current_mean_sig - baseline_mean_sig
            sig_improvement = max(0, sig_improvement)

            # Efficiency Index = Retained Area * Significance Improvement
            # Physical Meaning: Total "Net" Extra Significance Gained
            efficiency = area_ratio * sig_improvement

        else:
            current_mean_sig = 0
            efficiency = 0

        metrics.append(
            {
                "Threshold_Percent": p,
                "Remaining_Area": area_ratio,
                "Mean_Significance": current_mean_sig,
                "Efficiency_Index": efficiency,
                "Cutoff_Value": cutoff_value,
            }
        )

    results_df = pd.DataFrame(metrics)

    # Normalize efficiency index to 0-1 range
    max_eff = results_df["Efficiency_Index"].max()
    if max_eff > 0:
        results_df["Efficiency_Index"] /= max_eff

    print("[OK] Efficiency analysis complete.")
    return results_df, label_a, label_b


def plot_efficiency_curve(df_res, output_dir, label_pair_name):
    """
    Plot Dual-Axis Efficiency Curve

    Plot three curves:
    1. Retained Area Curve (Grey Dashed) - Shows proportion of pixels retained after filtering
    2. Mean Significance Curve (Blue) - Shows average significance of retained pixels
    3. Efficiency Index Curve (Red) - Shows product of area and significance

    And mark the optimal threshold point (max efficiency index).

    Args:
        df_res: DataFrame containing analysis results
        output_dir: Directory for output images
        label_pair_name: Name of label pair (used for filename)

    Returns:
        None (Directly saves image file)
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
        "Retained Visual Area (%)", color=color_area, fontsize=13, fontweight="bold"
    )

    line1 = ax1.plot(
        x,
        df_res["Remaining_Area"] * 100,
        color=color_area,
        # linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Remaining Area (%)",
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
        label="Net Signal Gain (Optimal Trade-off)",
    )

    # ========== Mark Optimal Point ==========
    max_idx = df_res["Efficiency_Index"].idxmax()
    peak_x = df_res.iloc[max_idx]["Threshold_Percent"]
    peak_y = df_res.iloc[max_idx]["Efficiency_Index"]

    # Draw vertical line
    plt.axvline(x=peak_x, color=color_eff, linestyle=":", linewidth=2, ymax=0.95)

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
        f"Optimal Filter Threshold: {peak_x:.0f}%\n(Max Signal Gain)",
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
        f"Optimal Filtering Threshold Selection: {label_pair_name}",
        fontsize=15,
        y=1.15,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save PNG format
    save_path_png = os.path.join(
        output_dir, f"Optimal_Threshold_Justification_{label_pair_name}.png"
    )
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    print(f"[OK] PNG image saved: {save_path_png}")

    # Save SVG format
    save_path_svg = os.path.join(
        output_dir, f"Optimal_Threshold_Justification_{label_pair_name}.svg"
    )
    plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
    print(f"[OK] SVG image saved: {save_path_svg}")

    plt.show()


# ============================================================================
# Main Program Entry
# ============================================================================


def main():
    """
    Main Program Entry

    Process Flow:
    1. Create Tkinter GUI window
    2. Prompt user to select master_summary.xlsx file
    3. Prompt user to input classification label column name
    4. Execute efficiency analysis
    5. Save analysis results and plot data
    6. Generate efficiency curve plot
    """
    print("=" * 60)
    print("Efficiency Index Analysis Tool")
    print("=" * 60)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    excel_path = filedialog.askopenfilename(
        title="select master_summary.xlsx", filetypes=[("Excel", "*.xlsx")]
    )
    if not excel_path:
        print("No file selected, exiting.")
        return

    base_dir = os.path.dirname(excel_path)
    df = pd.read_excel(excel_path)
    print(f"[OK] Loaded Excel file: {excel_path}")

    # Select classification label column
    cols = list(df.columns)
    label_col = simpledialog.askstring(
        "Input Classification Label Column",
        f"Please input column name for comparison (e.g., Diagnosis, Group):\nAvailable columns: {cols[:5]}...",
        initialvalue="Diagnosis",
    )

    if not label_col or label_col not in df.columns:
        print("Error: Invalid label column.")
        return

    print(f"[OK] Selected label column: {label_col}")

    # Execute analysis
    print("\nStarting efficiency analysis...")
    result = analyze_efficiency(df, base_dir, label_col)

    if result:
        df_res, label_a, label_b = result
        # Get directory of current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Save plot data
        import pickle

        pkl_path = os.path.join(script_dir, "Extended Data Fig. 2_plot_data.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"df_res": df_res, "label_a": label_a, "label_b": label_b}, f)
        print(f"[OK] Plot data saved: {pkl_path}")

        # Generate efficiency curve plot
        print("\nGenerating efficiency curve plot...")
        plot_efficiency_curve(df_res, script_dir, f"{label_a}_vs_{label_b}")
        print("[OK] Analysis complete!")
    else:
        print("[X] Analysis failed.")


if __name__ == "__main__":
    main()
