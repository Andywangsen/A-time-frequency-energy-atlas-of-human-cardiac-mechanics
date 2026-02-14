"""
==============================================================================
# Electrocardiogram (ECG) and Seismocardiogram (SCG) Signal Processing and Time-Frequency Analysis
# Fig. 1b-1e Usage: The 1st cardiac cycle within the selected region
# Fig. 1f Usage: The first 4 cardiac cycles within the selected region
# Fig. 1h-1i Usage: All cardiac cycles within the selected region (Signal Averaging)
# Function:
#   - Read multi-channel physiological signal data
#   - Detect R-peaks
#   - Perform time-frequency analysis (FFT, CWT, SST)
#   - Generate energy feature maps
#   - Support interactive time range selection
==============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from scipy.ndimage import zoom
from matplotlib.widgets import SpanSelector
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from matplotlib.ticker import MultipleLocator

mplstyle.use("fast")

# Display Configuration
SHOW_PLOTS_AFTER_ANALYSIS = False

# Time-Frequency Analysis Parameters
NORMALIZED_TIME_POINTS = 1000
NORMALIZED_FREQ_BINS = 256

# Energy Thresholds
ENERGY_VMAX = 15
CWTENERGY_VMAX = 140
PJENERGY_VMAX = 0.0000600

ENERGY_UNIT_SCALE = "e-6"

# Plotting Configuration
COLORBAR_START_PERCENT = 0

GRID_X_DTICK = 200
GRID_Z_NTICKS = 4

FIG5_2FONT_LABEL = 12
FIG5_2FONT_TICK = 10
FIG5_2FONT_LEGEND = 10

# Ridgeline Configuration
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

# Matplotlib Configuration
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.autolayout"] = True

# Signal Color Definition
COLOR_ECG = "#333333"
COLOR_SCG = "#009E73"
COLOR_R_PEAK = "black"


def convert_scg_unit(raw_scg, unit="ms2"):
    # Convert SCG raw LSB values to physical units
    # Args: raw_scg - raw values, unit - target unit ('g'/'ms2'/'mg')
    # Returns: converted values
    # Scale factor: 2.0/32768.0 (corresponding to +/- 2g range of 16-bit ADC)
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
    # Convert ECG raw LSB values to voltage units
    # Args: raw_ecg - raw values, unit - target unit ('mV'/'uV')
    # Returns: converted values
    # Reference voltage: 2454mV, Gain: 12, ADC range: 16-bit
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
    # Return custom colormap
    # Color gradient: Dark Purple -> Blue -> Yellow -> Orange -> Red
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
    # Zero-phase Butterworth filter
    # Use filtfilt to ensure no phase distortion
    # Args: fs - sampling rate, f_cut - cutoff frequency, btype - filter type
    data = np.array(data, dtype=float)
    order = 2
    b, a = butter(order, np.array(f_cut) / (fs / 2), btype=btype)
    return filtfilt(b, a, data)


def detect_r_peaks(data_ecg_raw, ecg_filtered, fs_ecg):
    # Multi-stage R-peak detection algorithm
    # Steps:
    #   1. Derivative calculation (change detection)
    #   2. Squaring (enhance peaks)
    #   3. Integration (energy accumulation)
    #   4. Segmentation (handle long signals)
    #   5. Peak detection and refinement
    #   6. Amplitude threshold filtering
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
    # Process long signals in segments (60-second windows)
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
            # Refine peak at local maximum
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
    # Apply global amplitude threshold
    thr = 3.5 * np.median(peak_amplitudes) if len(peak_amplitudes) > 0 else 0
    valid_peaks = final_peaks_np[peak_amplitudes <= thr] if thr > 0 else final_peaks_np
    return np.array(sorted(list(set(valid_peaks))))


def plot_fft_spectrum(scg_signal, fs, save_path=None):
    # Plot Power Spectral Density (PSD)
    # Frequency range: 0-100Hz
    sig_mms2 = convert_scg_unit(scg_signal, unit="ms2") * 1000

    N = len(sig_mms2)
    fft_vals = fft(sig_mms2)
    fft_freq = fftfreq(N, 1 / fs)

    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    psd = (np.abs(fft_vals[pos_mask]) ** 2) / (N * fs)

    fig, ax = plt.subplots()
    ax.plot(freqs, psd, color="black", linewidth=1)
    ax.set_xlabel("Frequency(Hz)")

    ax.set_xlim(0, 100)
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(colors="black")

    y_max = np.max(psd)
    if y_max > 0:
        order = int(np.floor(np.log10(y_max)))
        psd_scaled = psd / (10**order)
        ax.clear()

        ax.plot(freqs, psd_scaled, color="black", linewidth=1)
        ax.set_xlabel("Frequency(Hz)")

        ax.set_ylabel(r"PSD($10^{{{}}}$ $(mm/s^2)^2$/Hz)".format(order))

        ax.set_xlim(0, 100)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(colors="black")

        x_text = 40
        y_text = np.max(psd_scaled) * 0.9
        ax.text(
            x_text,
            y_text,
            "Frequency-domain lacks time information",
            fontsize=10,
            color="black",
        )

    else:
        ax.set_ylabel("PSD")

    if save_path:
        plt.savefig(save_path, format="png")

    return fig


def plot_cwt_timefreq(scg_signal, fs, t_start=0, save_path=None):
    # CWT Time-Frequency Analysis
    # Wavelet: Gaussian Morlet (gamma=3, beta=20/3)
    # Frequency range: 2-100Hz (logarithmic scale)
    # Time-Frequency Resolution: CWT < SST (SST is clearer)
    try:
        from ssqueezepy import cwt
        from ssqueezepy.experimental import scale_to_freq

        wavelet = ("gmw", {"gamma": 3, "beta": 20 / 3, "norm": "bandpass"})
        Wx, scales = cwt(scg_signal, wavelet, fs=fs, nv=32)

        freqs = scale_to_freq(scales, wavelet, len(scg_signal), fs=fs)
        t = t_start + np.arange(len(scg_signal)) / fs

        fig = plt.figure()

        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[35, 1],
        )

        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        cmap_cwt = get_truncated_colormap(COLORBAR_START_PERCENT)
        im = ax.pcolormesh(
            t,
            freqs,
            np.abs(Wx),
            cmap=cmap_cwt,
            shading="gouraud",
            vmin=0,
            vmax=CWTENERGY_VMAX,
        )
        ax.set_yscale("log")
        ax.set_ylim(2, 100)
        ax.set_xlim(t[0], t[-1])
        ax.set_yticks([2, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.tick_params()
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

        ax.text(
            0.5,
            0.15,
            "Continuous Wavelet Transform (CWT)\nTime-Frequency Blurring",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            ha="center",
            va="bottom",
        )

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Amplitude (a.u.)")
        cbar.ax.tick_params()

        if save_path:
            plt.savefig(save_path, format="png")

        return fig, Wx, freqs

    except Exception as e:
        return None, None, None


def plot_sst_single_beat(scg_signal, fs, t_start=0, save_path=None):
    # SST Time-Frequency Analysis
    # Wavelet: Gaussian Morlet (gamma=3, beta=20/3)
    # Frequency range: 2-100Hz (logarithmic scale)
    # Better time-frequency resolution than CWT
    try:
        from ssqueezepy import ssq_cwt

        wavelet = ("gmw", {"gamma": 3, "beta": 20 / 3, "norm": "bandpass"})
        Tx, _, ssq_freqs, *_ = ssq_cwt(
            scg_signal, wavelet=wavelet, scales="log-piecewise", fs=fs, nv=32
        )

        t = t_start + np.arange(len(scg_signal)) / fs

        fig = plt.figure()

        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[35, 1],
        )

        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        cmap_sst = get_truncated_colormap(COLORBAR_START_PERCENT)
        im = ax.pcolormesh(
            t,
            ssq_freqs,
            np.abs(Tx),
            cmap=cmap_sst,
            shading="gouraud",
            vmin=0,
            vmax=ENERGY_VMAX,
        )
        ax.set_yscale("log")
        ax.set_ylim(2, 100)
        ax.set_xlim(t[0], t[-1])
        ax.set_yticks([2, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.tick_params()
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

        ax.text(
            0.5,
            0.15,
            "Synchrosqueezing Wavelet Transform (SSWT)\nHigh Time-Frequency Resolution",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            ha="center",
            va="bottom",
        )

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Amplitude (a.u.)")
        cbar.ax.tick_params()

        if save_path:
            plt.savefig(save_path, format="png")

        return fig, Tx, ssq_freqs

    except Exception as e:
        return None, None, None


def plot_signal_only(
    signal_fslice,
    ecg_slice,
    r_peaks_indices_in_range,
    t_start,
    t_end,
    fs_ecg,
    save_path=None,
):
    # Plot Synchronized Signals (SCG + ECG + R-peaks)
    fig, ax_sig = plt.subplots()

    # Signal plot
    # Convert raw to mm/s^2: m/s^2 * 1000
    signal_fslice_mms2 = convert_scg_unit(signal_fslice, unit="ms2") * 1000
    t_scg_slice = t_start + np.arange(len(signal_fslice)) / fs_scg_g
    ax_sig.plot(t_scg_slice, signal_fslice_mms2, label="SCG", color=COLOR_SCG, lw=1)
    ax_sig.set_xlabel("Time(s)")
    ax_sig.set_ylabel("SCG Amplitude(mm/s$^2$)", color=COLOR_SCG)

    # Customizing Left Axis (SCG)
    ax_sig.tick_params(axis="y", colors=COLOR_SCG)
    ax_sig.spines["left"].set_color(COLOR_SCG)
    ax_sig.spines["left"].set_linewidth(1.5)

    # Remove top and right spines from the main axis (right spine will be added by twinx)
    ax_sig.spines["top"].set_visible(False)
    ax_sig.spines["right"].set_visible(False)  # twinx will handle this side

    ax_ecg = ax_sig.twinx()
    # Convert raw to uV using new function
    ecg_slice_uV = convert_ecg_unit(ecg_slice, unit="mV")
    t_ecg_slice = t_start + np.arange(len(ecg_slice)) / fs_ecg
    ax_ecg.plot(t_ecg_slice, ecg_slice_uV, label="ECG", color=COLOR_ECG, lw=1)

    ax_ecg.set_ylabel("ECG Amplitude(mV)", color=COLOR_ECG)
    ax_ecg.text(
        0.5,
        0.85,
        "Time-domain lacks frequency information",
        transform=ax_ecg.transAxes,
        fontsize=10,
        color="black",
    )
    # Customizing Right Axis (ECG)
    ax_ecg.tick_params(axis="y", colors=COLOR_ECG)
    ax_ecg.spines["right"].set_color(COLOR_ECG)
    ax_ecg.spines["right"].set_linewidth(1.5)
    ax_ecg.spines["left"].set_visible(False)
    ax_ecg.spines["top"].set_visible(False)

    lines_1, labels_1 = ax_sig.get_legend_handles_labels()
    lines_2, labels_2 = ax_ecg.get_legend_handles_labels()
    ax_sig.legend(
        lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True
    )

    if save_path:
        plt.savefig(save_path, format="png")

    return fig


def plot_comprehensive_analysis(
    signal_fslice,
    signal_slice,
    ecg_slice,
    r_peaks_indices_in_range,
    t_start,
    t_end,
    fs_scg,
    fs_ecg,
    save_path=None,
):
    """
    Generate and return SST analysis figure and signal plot.
    """
    if len(signal_slice) < 50:
        return None, None

    wavelet_config = ("gmw", {"gamma": 3, "beta": 20 / 3, "norm": "bandpass"})
    nv = 32

    fig = plt.figure()

    # 6. Configure GridSpec
    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.05,
        height_ratios=[7, 3],
        width_ratios=[35, 1],
    )

    # 7. Generate Subplot Objects
    ax_sst = fig.add_subplot(gs[0, 0])
    ax_sig = fig.add_subplot(gs[1, 0], sharex=ax_sst)
    cax = fig.add_subplot(gs[0, 1])
    ax_dummy = fig.add_subplot(gs[1, 1])
    ax_dummy.axis("off")

    # SST analysis
    Tx, ssq_freqs = None, None
    try:
        Tx, _, ssq_freqs, *_ = ssq_cwt(
            signal_slice,
            wavelet=wavelet_config,
            scales="log-piecewise",
            fs=fs_scg,
            nv=nv,
        )
        t_sst = t_start + np.arange(Tx.shape[1]) / fs_scg
        cmap_comprehensive = get_truncated_colormap(COLORBAR_START_PERCENT)
        # Use ENERGY_VMAX as colorbar maximum (actual range of SST amplitude)
        im = ax_sst.pcolormesh(
            t_sst,
            ssq_freqs,
            np.abs(Tx),
            cmap=cmap_comprehensive,
            shading="gouraud",
            vmin=0,
            vmax=ENERGY_VMAX,
        )

        ax_sst.set_ylabel("Frequency [Hz]")

        # Hide x-axis labels and ticks for the upper plot
        ax_sst.set_xlabel("")
        plt.setp(ax_sst.get_xticklabels(), visible=False)
        ax_sst.tick_params(axis="x", which="both", length=0)

        ax_sst.set_ylim(2, 100)
        ax_sst.set_yscale("log")
        ax_sst.set_yticks([2, 5, 10, 20, 50, 100])
        ax_sst.set_yticklabels(["2", "5", "10", "20", "50", "100"])
        ax_sst.tick_params()

        # Colorbar settings, number of ticks controlled by GRID_Z_NTICKS
        cbar = fig.colorbar(im, cax=cax)
        tick_step = (
            ENERGY_VMAX / (GRID_Z_NTICKS - 1) if GRID_Z_NTICKS > 1 else ENERGY_VMAX
        )
        if tick_step >= 1:
            tick_step = np.ceil(tick_step)
        ticks = np.arange(0, ENERGY_VMAX + tick_step * 0.5, tick_step)
        ticks = ticks[ticks <= ENERGY_VMAX * 1.05]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{int(v)}" if v == int(v) else f"{v:.1f}" for v in ticks])
        cbar.set_label("Amplitude (a.u.)", labelpad=10)
        cbar.ax.tick_params()

    except Exception as e:
        ax_sst.text(
            0.5,
            0.5,
            f"SST calculation failed:\n{e}",
            ha="center",
            va="center",
            color="red",
            transform=ax_sst.transAxes,
        )

    # Signal plot

    # Signal plot
    # Convert raw to mm/s^2: m/s^2 * 1000
    signal_fslice_mms2 = convert_scg_unit(signal_fslice, unit="ms2") * 1000
    t_scg_slice = t_start + np.arange(len(signal_slice)) / fs_scg
    ax_sig.plot(t_scg_slice, signal_fslice_mms2, label="SCG", color=COLOR_SCG, lw=1)
    ax_sig.set_xlabel("Time(s)")
    ax_sig.set_ylabel(
        "SCG Amplitude (mm/s$^2$)", color="dodgerblue", fontsize=FIG5_2FONT_LABEL
    )
    ax_sig.tick_params(axis="y", labelcolor="dodgerblue", labelsize=FIG5_2FONT_TICK)
    ax_sig.yaxis.set_major_locator(MultipleLocator(100))

    # Remove overlapping sub-titles
    ax_sig.grid(True, alpha=0.5)
    # Customizing Left Axis (SCG)
    ax_sig.spines["left"].set_color("dodgerblue")
    ax_sig.spines["left"].set_linewidth(1.5)
    ax_sig.spines["top"].set_visible(False)
    ax_sig.spines["right"].set_visible(False)

    ax_ecg = ax_sig.twinx()
    # Convert raw to uV
    ecg_slice_uV = convert_ecg_unit(ecg_slice, unit="mV")
    t_ecg_slice = t_start + np.arange(len(ecg_slice)) / fs_ecg
    ax_ecg.plot(t_ecg_slice, ecg_slice_uV, label="ECG", color="tomato", lw=0.5)

    # --- R-wave Amplitude Extraction Logic ---

    # 1. Assume r_peaks_indices_in_range are absolute indices of the original signal
    # 2. Determine start index of current ecg_slice corresponding to original signal
    start_index_ecg = int(t_start * fs_ecg)
    end_index_ecg = int(t_end * fs_ecg)

    # 3. Filter R-peak indices falling within the current time range
    r_peaks_in_range_global = r_peaks_indices_in_range[
        (r_peaks_indices_in_range >= start_index_ecg)
        & (r_peaks_indices_in_range < end_index_ecg)
    ]

    # 4. Calculate local indices of these R-peaks relative to ecg_slice
    local_r_peak_indices = r_peaks_in_range_global - start_index_ecg

    # 5. Extract R-peak times (relative to t=0)
    r_peaks_times = r_peaks_in_range_global / fs_ecg

    # 6. Extract R-peak amplitudes (using local indices from ecg_slice)
    r_peaks_amps = ecg_slice_uV[local_r_peak_indices]

    # --- R-wave Amplitude Extraction Logic End ---

    # Ensure plotting only if valid R-peaks are found
    if len(r_peaks_times) > 0:
        ax_ecg.plot(
            r_peaks_times,
            r_peaks_amps,
            "x",
            color="black",
            markersize=5,
            label="R-peaks",
        )

    ax_ecg.set_ylabel(
        "ECG Amplitude [mV]", color="tomato", fontsize=FIG5_2FONT_LABEL, labelpad=25
    )
    ax_ecg.tick_params(axis="y", labelcolor="tomato", labelsize=FIG5_2FONT_TICK)
    ax_ecg.yaxis.set_major_locator(MultipleLocator(1))

    # Set x-axis range and R-peak markers for all subplots
    for ax in [ax_sst, ax_sig]:
        ax.set_xlim(t_start, t_end)
        # SST plot uses white dashed line, signal plot uses black dashed line
        linestyle, color = ("--", "black") if ax == ax_sig else ("--", "white")
        for r_time in r_peaks_times:
            ax.axvline(
                x=r_time, color=color, linestyle=linestyle, linewidth=0.8, alpha=0.9
            )

    # Add Beat Labels to SST plot
    if len(r_peaks_times) >= 2:
        for i in range(len(r_peaks_times) - 1):
            # Calculate center time between two R-peaks
            center_time = (r_peaks_times[i] + r_peaks_times[i + 1]) / 2

            ax_sst.text(
                center_time,
                60,
                f"Beat {i + 1}",
                color="white",
                ha="center",
                va="center",
                fontsize=12,
            )

    # Add "Plausibly Similar But Distinct" text
    ax_sst.text(
        0.5,
        0.15,
        "Plausibly Similar But Distinct",
        transform=ax_sst.transAxes,
        color="white",
        ha="center",
        va="center",
        fontsize=12,
    )

    ax_sst.set_ylabel("Frequency (Hz)")

    if save_path:
        plt.savefig(save_path, format="png")

    return fig, Tx


def calculate_average_beat_atlas(scg_full, r_peaks_indices_ecg, fs_scg, fs_ecg):
    # Calculate average cardiac cycle energy feature map
    # Processing steps:
    #   1. Perform SST analysis beat by beat
    #   2. Interpolate to uniform frequency grid (256 frequency points)
    #   3. Time axis normalization (1000 time points)
    #   4. Calculate average of all beats
    # Output: 256x1000 energy matrix
    if len(r_peaks_indices_ecg) < 2:
        return None

    # Convert R-peak indices from ECG sampling rate to SCG sampling rate
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
            # Interpolate to uniform frequency grid
            interp_Tx_beat = np.apply_along_axis(
                lambda col: np.interp(final_freqs, ssq_freqs_beat[sort_indices], col),
                axis=0,
                arr=abs_Tx_beat[sort_indices, :],
            )
            # Time axis normalization
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
    # Calculate background threshold based on energy accumulation percentile
    # Args: percentile - Energy accumulation percentile (default 50%)
    # Returns: Dictionary containing threshold and statistics
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


def plot_energy_atlas(
    atlas_matrix,
    title_prefix="",
    apply_filter=False,
    bg_stats=None,
    vmax_unified=None,
    r_peaks_in_range=None,
):
    # Ensure non-negative values
    atlas_matrix = np.maximum(atlas_matrix, 0)

    # Normalize first (using original data sum), ensure Fig1_h and Fig1_i use same normalization baseline
    total_sum = np.sum(atlas_matrix)
    normalized_atlas_matrix = (
        atlas_matrix / total_sum if total_sum > 1e-9 else atlas_matrix
    )

    # Apply background filtering (after normalization)
    if apply_filter and bg_stats is not None:
        # Normalize threshold to the same scale
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

    # Select colormap based on whether filtering is applied - Using COLOR_SCHEME
    # Get custom colormap (use custom gradient for spectral)
    base_cmap = get_truncated_colormap(0)  # Get base colormap

    if apply_filter:
        # Filtered version: directly use original colormap, but handle transparency with masked array
        cmap = base_cmap
    else:
        # Unfiltered version: directly use original colormap (consistent with Fig1_e, 0 is not white)
        cmap = base_cmap

    # 4. Create Figure
    fig = plt.figure()

    # 6. Configure GridSpec
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[35, 1],
    )

    # 7. Generate subplot objects
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    n_freqs, n_times = atlas_matrix.shape

    if not apply_filter and COLORBAR_START_PERCENT > 0:
        cmap = get_truncated_colormap(COLORBAR_START_PERCENT)

    # Draw Image
    if apply_filter:
        # Use masked array to set 0 values to transparent
        masked_atlas = np.ma.masked_less_equal(normalized_atlas_matrix, 1e-10)
        im = ax.imshow(
            masked_atlas,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=[0, 1000, 0, n_freqs],
            vmin=0,
            vmax=PJENERGY_VMAX,
        )
    else:
        im = ax.imshow(
            normalized_atlas_matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=[0, 1000, 0, n_freqs],
            vmin=0,
            vmax=PJENERGY_VMAX,
        )

    if title_prefix:
        ax.set_title(f"{title_prefix}", pad=5)

    # Set ticks - Use percentage for x-axis
    x_tick_positions = [0, 200, 400, 600, 800, 1000]
    x_tick_labels = ["0", "20", "40", "60", "80", "100"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    # Y-axis uses logarithmic frequency scale (2 to 100Hz), fixed ticks at 2, 5, 10, 20, 50, 100
    freq_bins = np.logspace(np.log10(2), np.log10(100), n_freqs)
    target_freqs = [2, 5, 10, 20, 50, 100]
    y_tick_positions = []
    for freq in target_freqs:
        # Find index closest to target frequency
        idx = np.argmin(np.abs(freq_bins - freq))
        y_tick_positions.append(idx)
    y_tick_labels = ["2", "5", "10", "20", "50", "100"]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel("Normalized Cardiac Cycle (0-100%)")
    ax.set_ylabel("Frequency(Hz)")

    # Add Text Annotations
    if not apply_filter:
        # Top Right
        ax.text(
            0.95,
            0.9,
            f"Signal-Averaged Processing({len(r_peaks_in_range) - 1} Beats)\nStable Energy Retention\nStochastic Noise Reduction",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            ha="right",
            va="top",
        )
        # Bottom Right
        ax.text(
            0.95,
            0.1,
            "Main Skeleton Surrounded by\nLow-energy Vibration or Jitter",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            ha="right",
            va="bottom",
        )

    # Add colorbar - Consistent style with Fig1_e
    # Set unit conversion factor based on ENERGY_UNIT_SCALE
    if ENERGY_UNIT_SCALE == "e-5":
        unit_factor = 1e-5
        unit_label = r"Normalized Energy (a.u.)"
    else:  # Default 'e-6'
        unit_factor = 1e-6
        unit_label = r"Normalized Energy (a.u.)"

    cbar = fig.colorbar(im, cax=cax)

    # Generate exact number of ticks based on GRID_Z_NTICKS
    max_scaled = PJENERGY_VMAX / unit_factor  # Convert to display unit
    # Generate specific tick points (including 0 and max)
    ticks_scaled = np.linspace(0, max_scaled, GRID_Z_NTICKS)
    # Round tick values to integers (concise)
    ticks_scaled = np.round(ticks_scaled).astype(int)
    ticks = ticks_scaled * unit_factor  # Convert back to original units
    cbar.set_ticks(ticks)
    # Display ticks in concise integer format (e.g. 0, 20, 40, 60)
    cbar.set_ticklabels([f"{int(tick)}" for tick in ticks_scaled])
    cbar.set_label(unit_label)
    cbar.ax.tick_params()

    return fig, normalized_atlas_matrix


def plot_Fig1_i_with_marginals(
    atlas_matrix, title_prefix="", apply_filter=False, bg_stats=None, vmax_unified=None
):
    """
    Fig1_i: Energy Atlas with Ridgeline Plot (Ridgeline)

    Layout:
      Main: Energy Atlas (Heatmap) with Ridgeline overlay
    """
    # 1. Data Preparation
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

    # 6. Configure GridSpec
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[35, 1],
    )  # Width ratio: Main plot vs Colorbar

    # 7. Generate subplot objects
    ax_main = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    # 3. Plot Main Heatmap
    n_freqs, n_times = normalized_atlas_matrix.shape

    # Colormap
    if not apply_filter and COLORBAR_START_PERCENT > 0:
        cmap = get_truncated_colormap(COLORBAR_START_PERCENT)
    else:
        cmap = get_truncated_colormap(0)

    if apply_filter:
        # Filtered version: use base cmap (no white override)
        pass

    # Set white background
    ax_main.set_facecolor("white")

    # 4. Add Ridgeline Plot
    # Create logarithmic frequency axis
    freq_bins = np.logspace(np.log10(2), np.log10(100), n_freqs)

    # Select frequency layers to plot (uniformly distributed in log space)
    # Use RIDGELINE_COUNT parameter to control number of ridge lines
    ridgeline_indices = np.linspace(0, n_freqs - 1, RIDGELINE_COUNT, dtype=int)

    # Time axis
    t_axis = np.linspace(0, 1000, n_times)

    # Calculate global max energy for normalization
    global_max_energy = np.max(normalized_atlas_matrix)

    # Get colormap object for ridgeline coloring
    if isinstance(cmap, str):
        cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Preprocess all ridge data grouped by frequency layer
    # ridges_by_freq[freq_idx] = [ridge1, ridge2, ...] (Multiple discontinuous regions may exist in same frequency layer)
    ridges_by_freq = {}
    for freq_idx in ridgeline_indices[::-1]:  # From high frequency to low frequency
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

    # Get list of frequency layers in drawing order (High to Low)
    freq_order = list(ridges_by_freq.keys())

    for i, freq_idx in enumerate(freq_order):
        ridges = ridges_by_freq[freq_idx]
        base_zorder = 10 + i  # Lower frequency has larger i, thus higher zorder

        for ridge in ridges:
            baseline = ridge["baseline"]
            t_segment = ridge["t_segment"]
            energy_segment = ridge["energy_segment"]
            current_max_energy = ridge["current_max_energy"]
            current_max_height = ridge["current_max_height"]

            # Construct polygon
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

            # Rectangular range
            min_x, max_x = t_segment[0], t_segment[-1]
            min_y, max_y = baseline, baseline + current_max_height

            # Construct gradient data
            num_y_pixels = 50
            gradient_data = np.linspace(0, current_max_energy, num_y_pixels).reshape(
                -1, 1
            )

            # Draw fill, set zorder
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

            # Draw ridgeline
            if RIDGELINE_LINE_WIDTH > 0 and RIDGELINE_LINE_COLOR.lower() != "none":
                if RIDGELINE_REVERSE_ORDER:
                    # True: Lines on top layer
                    line_zorder = 10 + len(freq_order) + 1
                else:
                    # False: Lines slightly higher than corresponding fill, but lower than next layer fill
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

    # 5. Axis Formatting (Main Plot)
    # X-axis
    x_tick_positions = [0, 200, 400, 600, 800, 1000]
    x_tick_labels = ["0", "20", "40", "60", "80", "100"]
    ax_main.set_xticks(x_tick_positions)
    ax_main.set_xticklabels(x_tick_labels)
    ax_main.set_xlabel("Normalized Cardiac Cycle (0-100%)")

    # Add Text Annotation
    ax_main.text(
        0.95,
        0.9,
        "SSTF Atlas\n50% Core Energy Pattern",
        transform=ax_main.transAxes,
        color="black",
        fontsize=12,
        ha="right",
        va="top",
    )

    # Y-axis (Log Frequency)
    target_freqs = [2, 5, 10, 20, 50, 100]
    y_tick_positions = []
    for freq in target_freqs:
        idx = np.argmin(np.abs(freq_bins - freq))
        y_tick_positions.append(idx)
    y_tick_labels = ["2", "5", "10", "20", "50", "100"]
    ax_main.set_yticks(y_tick_positions)
    ax_main.set_yticklabels(y_tick_labels)
    ax_main.set_ylabel("Frequency (Hz)")

    # IMPORTANT: Set limits to ensure the whole plot is visible
    ax_main.set_xlim(0, 1000)
    ax_main.set_ylim(0, n_freqs)

    if title_prefix:
        ax_main.set_title(title_prefix, pad=10)

    # 6. Colorbar - Use ScalarMappable to create colorbar (Ensure starting from 0)
    if ENERGY_UNIT_SCALE == "e-5":
        unit_factor = 1e-5
        unit_label = r"Normalized Energy (a.u.)"
    else:
        unit_factor = 1e-6
        unit_label = r"Normalized Energy (a.u.)"

    # Use ScalarMappable to create colorbar, ensure vmin=0
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=0, vmax=PJENERGY_VMAX)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)

    # Generate tick count accurately based on GRID_Z_NTICKS (Consistent with Fig1_h)
    max_scaled = PJENERGY_VMAX / unit_factor  # Convert to display units
    # Generate GRID_Z_NTICKS tick points accurately (including 0 and max)
    ticks_scaled = np.linspace(0, max_scaled, GRID_Z_NTICKS)
    # Round tick values to integers (concise)
    ticks_scaled = np.round(ticks_scaled).astype(int)
    ticks = ticks_scaled * unit_factor  # Convert back to original units
    cbar.set_ticks(ticks)
    # # Use concise integer format for ticks (e.g.,0, 20, 40, 60)
    cbar.set_ticklabels([f"{int(tick)}" for tick in ticks_scaled])
    cbar.set_label(unit_label)
    cbar.ax.tick_params()

    return fig, normalized_atlas_matrix


# ==============================
#   4) Result Saving Function
# ==============================


def save_analysis_results(
    base_filename,
    time_range,
    figures,
    data_to_save,
    output_folder=None,
):
    """Save analysis results to a folder named after the file (PNG and HTML only)"""
    try:
        # 1. Create unified filename prefix
        file_prefix = f"{os.path.splitext(base_filename)[0]}_{time_range[0]:.2f}s-{time_range[1]:.2f}s"

        if output_folder:
            folder_name = output_folder
        else:
            file_prefix1 = f"{os.path.splitext(base_filename)[0]}"
            folder_name = f"{file_prefix1}_results"

        os.makedirs(folder_name, exist_ok=True)

        for name, fig in figures.items():
            if fig is None:
                continue

            if name.startswith("fig") and any(c.isdigit() for c in name[:4]):
                continue

            if hasattr(fig, "screenshot"):
                save_filename = f"{file_prefix}_{name}"
                png_path = os.path.join(folder_name, f"{save_filename}.png")

    except Exception as e:
        pass


def run_full_analysis(
    corrected_r_peaks, xmin, xmax, filter_percentile=50, show_3d=False
):
    # Execute full SST analysis and energy feature map generation
    print(f"[Analysis] Time range: {xmin:.2f}s - {xmax:.2f}s")

    start_idx_scg = int(xmin * fs_scg_g)
    end_idx_scg = int(xmax * fs_scg_g)
    signal_slice = data_scg_g[start_idx_scg:end_idx_scg]
    signal_fslice = datafiltered_scg_g[start_idx_scg:end_idx_scg]
    start_idx_ecg = int(xmin * fs_ecg_g)
    end_idx_ecg = int(xmax * fs_ecg_g)
    ecg_slice = ecg_filtered_g[start_idx_ecg:end_idx_ecg]

    r_peaks_in_range = corrected_r_peaks[
        (corrected_r_peaks >= start_idx_ecg) & (corrected_r_peaks <= end_idx_ecg)
    ]

    print(f"  Number of R-peaks: {len(r_peaks_in_range)}")

    if len(r_peaks_in_range) < 2:
        pass

    figures_to_save = {}
    data_to_save = {}

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    base_name = os.path.splitext(os.path.basename(filename_path_g))[0]
    folder_name = os.path.join(script_dir, f"{base_name}_results")
    os.makedirs(folder_name, exist_ok=True)

    # --- Prepare Data Slices ---
    # Default to full slice if logic fails
    sig_fslice_1beat = signal_fslice
    sig_slice_1beat = signal_slice
    ecg_slice_1beat = ecg_slice
    r_peaks_1beat = r_peaks_in_range
    t_start_1beat = xmin
    t_end_1beat = xmax

    sig_fslice_4beats = signal_fslice
    sig_slice_4beats = signal_slice
    ecg_slice_4beats = ecg_slice
    r_peaks_4beats = r_peaks_in_range
    t_start_4beats = xmin
    t_end_4beats = xmax

    # Padded versions for Fig1_b
    sig_fslice_1beat_pad = signal_fslice
    ecg_slice_1beat_pad = ecg_slice
    t_start_1beat_pad = xmin
    t_end_1beat_pad = xmax

    if len(r_peaks_in_range) >= 2:
        # Indices relative to the start of the selection
        r_start_idx_ecg = r_peaks_in_range[0] - start_idx_ecg

        # 1. First Beat (for Fig1_b-4)
        print(
            f"Cardiac cycle used for Fig. 1b-1e: The 1st one (R-peak positions: {r_peaks_in_range[0]} - {r_peaks_in_range[1]}, Time: {r_peaks_in_range[0] / fs_ecg_g:.2f}s - {r_peaks_in_range[1] / fs_ecg_g:.2f}s)"
        )
        r_end_idx_ecg_1 = r_peaks_in_range[1] - start_idx_ecg

        # Convert to SCG indices
        r_start_idx_scg = int(r_start_idx_ecg * fs_scg_g / fs_ecg_g)
        r_end_idx_scg_1 = int(r_end_idx_ecg_1 * fs_scg_g / fs_ecg_g)

        if 0 <= r_start_idx_scg < r_end_idx_scg_1 <= len(signal_fslice):
            sig_fslice_1beat = signal_fslice[r_start_idx_scg:r_end_idx_scg_1]
            sig_slice_1beat = signal_slice[r_start_idx_scg:r_end_idx_scg_1]
            ecg_slice_1beat = ecg_slice[r_start_idx_ecg:r_end_idx_ecg_1]
            t_start_1beat = xmin + r_start_idx_ecg / fs_ecg_g
            t_end_1beat = xmin + r_end_idx_ecg_1 / fs_ecg_g
            r_peaks_1beat = r_peaks_in_range[0:1]  # Just start peak

            # --- Fig1_b Padding Logic (0.2s) ---
            pad_sec = 0.2
            pad_samples_ecg = int(pad_sec * fs_ecg_g)

            idx_start_ecg_pad = max(0, r_start_idx_ecg - pad_samples_ecg)
            idx_end_ecg_pad = min(len(ecg_slice), r_end_idx_ecg_1 + pad_samples_ecg)

            idx_start_scg_pad = int(idx_start_ecg_pad * fs_scg_g / fs_ecg_g)
            idx_end_scg_pad = int(idx_end_ecg_pad * fs_scg_g / fs_ecg_g)

            if 0 <= idx_start_scg_pad < idx_end_scg_pad <= len(signal_fslice):
                sig_fslice_1beat_pad = signal_fslice[idx_start_scg_pad:idx_end_scg_pad]
                ecg_slice_1beat_pad = ecg_slice[idx_start_ecg_pad:idx_end_ecg_pad]
                t_start_1beat_pad = xmin + idx_start_ecg_pad / fs_ecg_g
                t_end_1beat_pad = xmin + idx_end_ecg_pad / fs_ecg_g

        # 2. First 4 Beats (for Fig1_f)
        num_beats_Fig1_f = min(len(r_peaks_in_range) - 1, 4)
        print(
            f"Cardiac cycles used for Fig. 1f: The first {num_beats_Fig1_f} (R-peak positions: {r_peaks_in_range[0]} - {r_peaks_in_range[num_beats_Fig1_f]}, Time: {r_peaks_in_range[0] / fs_ecg_g:.2f}s - {r_peaks_in_range[num_beats_Fig1_f] / fs_ecg_g:.2f}s)"
        )
        if num_beats_Fig1_f > 0:
            idx_end_peak = num_beats_Fig1_f
            r_end_idx_ecg_4 = r_peaks_in_range[idx_end_peak] - start_idx_ecg

            # --- Fig1_f Padding Logic (0.2s) ---
            pad_sec_f = 0.2
            pad_samples_ecg_f = int(pad_sec_f * fs_ecg_g)

            # Apply padding to indices
            idx_start_ecg_f_pad = max(0, r_start_idx_ecg - pad_samples_ecg_f)
            idx_end_ecg_f_pad = min(len(ecg_slice), r_end_idx_ecg_4 + pad_samples_ecg_f)

            # Convert to SCG indices
            idx_start_scg_f_pad = int(idx_start_ecg_f_pad * fs_scg_g / fs_ecg_g)
            idx_end_scg_f_pad = int(idx_end_ecg_f_pad * fs_scg_g / fs_ecg_g)

            if 0 <= idx_start_scg_f_pad < idx_end_scg_f_pad <= len(signal_fslice):
                sig_fslice_4beats = signal_fslice[idx_start_scg_f_pad:idx_end_scg_f_pad]
                sig_slice_4beats = signal_slice[idx_start_scg_f_pad:idx_end_scg_f_pad]
                ecg_slice_4beats = ecg_slice[idx_start_ecg_f_pad:idx_end_ecg_f_pad]
                t_start_4beats = xmin + idx_start_ecg_f_pad / fs_ecg_g
                t_end_4beats = xmin + idx_end_ecg_f_pad / fs_ecg_g
                r_peaks_4beats = r_peaks_in_range[0 : idx_end_peak + 1]

    # Fig1_b: Signal plot (1st Beat + Padding)
    print(f"  Generating figures...")
    Fig1_b = plot_signal_only(
        sig_fslice_1beat_pad,
        ecg_slice_1beat_pad,
        r_peaks_1beat,
        t_start_1beat_pad,
        t_end_1beat_pad,
        fs_ecg_g,
        save_path=os.path.join(folder_name, f"Fig. 1b.png"),
    )
    if Fig1_b:
        figures_to_save["Fig1_b_signals"] = Fig1_b
        print(f"    [OK] Fig1_b: Signal Synchronization Plot")

    # Fig1_c: FFT (1st Beat)
    Fig1_c = plot_fft_spectrum(
        sig_fslice_1beat,
        fs_scg_g,
        save_path=os.path.join(folder_name, f"Fig. 1c.png"),
    )
    if Fig1_c:
        figures_to_save["Fig1_c_fft"] = Fig1_c
        print(f"    [OK] Fig1_c: FFT Spectrum")

    # Fig1_d: CWT (1st Beat)
    # Note: plot_cwt_timefreq uses t_start for axis.
    Fig1_d, _, _ = plot_cwt_timefreq(
        sig_slice_1beat,  # Use broadband for CWT
        fs_scg_g,
        t_start=t_start_1beat,
        save_path=os.path.join(folder_name, f"Fig. 1d.png"),
    )
    if Fig1_d:
        figures_to_save["Fig1_d_cwt"] = Fig1_d
        print(f"    [OK] Fig1_d: CWT Time-Frequency Plot")

    # Fig1_e: SST (1st Beat)
    Fig1_e, _, _ = plot_sst_single_beat(
        sig_slice_1beat,
        fs_scg_g,
        t_start=t_start_1beat,
        save_path=os.path.join(folder_name, f"Fig. 1e.png"),
    )
    if Fig1_e:
        figures_to_save["Fig1_e_sst"] = Fig1_e
        print(f"    [OK] Fig1_e: SST Time-Frequency Plot")

    # Fig1_f: Comprehensive (First 4 Beats)
    Fig1_f, Tx = plot_comprehensive_analysis(
        sig_fslice_4beats,
        sig_slice_4beats,
        ecg_slice_4beats,
        r_peaks_4beats,
        t_start_4beats,
        t_end_4beats,
        fs_scg_g,
        fs_ecg_g,
        save_path=os.path.join(folder_name, f"Fig. 1f.png"),
    )
    if Fig1_f:
        figures_to_save["Fig1_f_comprehensive"] = Fig1_f
        print(f"    [OK] Fig1_f: Comprehensive Analysis")

    # Fig1_h & Fig1_i: Energy Atlas
    print(f"  Calculating Energy Atlas...")
    print(
        f"Cardiac cycles used for Fig. 1h-1i: All {len(r_peaks_in_range) - 1} complete cycles within the selected range"
    )
    avg_atlas = calculate_average_beat_atlas(
        data_scg_g, r_peaks_in_range, fs_scg_g, fs_ecg_g
    )

    if avg_atlas is not None:
        # Save original matrix
        # Normalize matrix
        avg_atlas = np.maximum(avg_atlas, 0)
        total_energy = np.sum(avg_atlas)
        normalized_atlas = (
            avg_atlas / total_energy if total_energy > 1e-9 else avg_atlas
        )
        normalized_atlas = (
            avg_atlas / total_energy if total_energy > 1e-9 else avg_atlas
        )

        # Background Energy Stats
        bg_stats = calculate_percentile_threshold(
            normalized_atlas, percentile=filter_percentile
        )

        print(f"    [OK] Energy Atlas Calculation Complete")
        print(f"      - Threshold: {bg_stats['threshold']:.2e}")
        print(f"      - Foreground Energy: {bg_stats['foreground_energy_sum']:.2e}")

        # Apply background filtering
        threshold = bg_stats["threshold"]
        filtered_atlas = normalized_atlas.copy()
        filtered_atlas[filtered_atlas < threshold] = 0
        # Calculate unified vmax
        unified_vmax = np.max(normalized_atlas)

        # Fig1_h: Energy Atlas (Unfiltered)
        Fig1_h, _ = plot_energy_atlas(
            normalized_atlas,
            apply_filter=False,
            bg_stats=bg_stats,
            vmax_unified=unified_vmax,
            r_peaks_in_range=r_peaks_in_range,
        )
        if Fig1_h:
            Fig1_h.savefig(os.path.join(folder_name, f"Fig. 1h.png"))
            figures_to_save["Fig1_h_energy_atlas"] = Fig1_h
            print(f"    [OK] Fig1_h: Energy Atlas")

        Fig1_i, _ = plot_Fig1_i_with_marginals(
            normalized_atlas,
            apply_filter=True,
            bg_stats=bg_stats,
            vmax_unified=unified_vmax,
        )
        if Fig1_i:
            Fig1_i.savefig(os.path.join(folder_name, f"Fig. 1i.png"))
            figures_to_save["Fig1_i_energy_marginals"] = Fig1_i
            print(f"    [OK] Fig1_i: Energy Atlas + Ridgelines")

        stats_dict = {
            "filter_method": bg_stats["filter_method"],
            "threshold": float(bg_stats["threshold"]),
            "background_energy_sum": float(bg_stats["background_energy_sum"]),
            "foreground_energy_sum": float(bg_stats["foreground_energy_sum"]),
            "total_energy": float(bg_stats["total_energy"]),
        }

        # Add specific fields based on filter method
        if bg_stats["filter_method"] == "percentile":
            stats_dict["percentile"] = float(bg_stats["percentile"])
        else:  # reference_region
            stats_dict["reference_mean"] = float(bg_stats["reference_mean"])
            stats_dict["reference_std"] = float(bg_stats["reference_std"])

        data_to_save["background_statistics"] = stats_dict
    else:
        pass

    t_scg_slice = np.linspace(xmin, xmax, len(signal_fslice))
    data_to_save["scg_signal"] = pd.DataFrame(
        {"Time(s)": t_scg_slice, "Amplitude": signal_fslice}
    )
    t_ecg_slice = np.linspace(xmin, xmax, len(ecg_slice))
    data_to_save["ecg_signal"] = pd.DataFrame(
        {"Time(s)": t_ecg_slice, "Amplitude": ecg_slice}
    )
    if len(r_peaks_in_range) > 0:
        r_peaks_times = t_ecg_g[r_peaks_in_range]
        r_peaks_amps = ecg_filtered_g[r_peaks_in_range]
        data_to_save["r_peaks"] = pd.DataFrame(
            {
                "Index": r_peaks_in_range,
                "Time(s)": r_peaks_times,
                "Amplitude": r_peaks_amps,
            }
        )

    save_analysis_results(
        os.path.basename(filename_path_g),
        (xmin, xmax),
        figures_to_save,
        data_to_save,
        output_folder=folder_name,
    )

    print(f"[OK] Results saved to: {folder_name}")

    if SHOW_PLOTS_AFTER_ANALYSIS:
        plt.show()
    else:
        for fig_name, fig_obj in figures_to_save.items():
            if fig_obj is not None and not isinstance(fig_obj, type(None)):
                # Close matplotlib figures
                if hasattr(fig_obj, "savefig"):
                    plt.close(fig_obj)
                # Close PyVista figures
                elif hasattr(fig_obj, "close"):
                    fig_obj.close()


def onselect_initial(xmin, xmax):
    """SpanSelector callback function: directly perform analysis without R-peak editing"""
    span_selector.active = False

    r_peaks_in_range = r_peaks_g[
        (t_ecg_g[r_peaks_g] >= xmin) & (t_ecg_g[r_peaks_g] <= xmax)
    ]

    filter_percentile = 50.0
    show_3d = False

    run_full_analysis(r_peaks_in_range, xmin, xmax, filter_percentile, show_3d)

    span_selector.active = True


# ==============================
#       6) Main Program
# ==============================
if __name__ == "__main__":
    if not SSQUEEZEPY_AVAILABLE:
        exit()
    root = tk.Tk()
    root.withdraw()
    filename_path_g = filedialog.askopenfilename(
        title="Select Data File",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    if not filename_path_g:
        exit()

    fs_ecg_g, fs_scg_g = 250, 500
    len_ecg, len_accz = [181, 240], [1, 120]
    cut_ecg, cut_accz = [0.5, 100], [2, 100]
    cut_scg = [20, 50]
    len_acc = [121, 180]
    fs_acc = 50

    cut_acc_1 = 20
    cut_acc_2 = 2
    try:
        data_raw = []
        with open(filename_path_g, "r") as file:
            for line in file:
                try:
                    values = [float(x) for x in line.strip().split(",")]
                    if len(values) >= 241:
                        data_raw.append(values[:240])
                except ValueError:
                    continue
        data_raw = np.array(data_raw)
        data_ecg_g = data_raw[:, len_ecg[0] - 1 : len_ecg[1]].flatten()
        ecg_filtered_g = fil(data_ecg_g, fs_ecg_g, cut_ecg)
        t_ecg_g = np.arange(len(ecg_filtered_g)) / fs_ecg_g
        data_accz_raw = data_raw[:, len_accz[0] - 1 : len_accz[1]].flatten()
        data_scg_g = fil(data_accz_raw, fs_scg_g, cut_accz, btype="bandpass")
        datafiltered_scg_g = fil(data_accz_raw, fs_scg_g, cut_scg, btype="bandpass")
        data_acc = data_raw[:, len_acc[0] - 1 : len_acc[1]].flatten()
        data_accy = fil(data_acc[1::5], fs_acc, cut_acc_1, btype="low")
        data_accy = fil(data_accy, fs_acc, cut_acc_2, btype="low")
        t_acc = np.arange(len(data_accy)) / fs_acc
        t_scg_g = np.arange(len(data_scg_g)) / fs_scg_g

    except Exception as e:
        exit()

    r_peaks_g = detect_r_peaks(data_ecg_g, ecg_filtered_g, fs_ecg_g)
    print(f"[OK] R-peak Detection Complete: {len(r_peaks_g)} R-peaks detected")
    if len(r_peaks_g) < 3:
        exit()

    fig_main, ax_main1 = plt.subplots()
    plt.suptitle(
        "Please drag with left mouse button to select a region for analysis (20~30s recommended)\nThe data used in the paper is start from 12s ",
    )

    ax_main1.plot(
        t_acc, data_accy, label="ACC (Low Frequency)", color="orange", alpha=0.9
    )
    ax_main1.set_ylim(-5000, 23800)
    ax_main = ax_main1.twinx()
    ax_main.plot(
        t_scg_g,
        datafiltered_scg_g,
        label="SCG (5-100Hz)",
        color="dodgerblue",
        alpha=0.9,
    )
    ax_main.set_xlabel("Time [s]")
    ax_main.set_ylabel("SCG Amplitude [mg]", color="dodgerblue")
    ax_ecg = ax_main  # Use twinx to create independent Y-axis
    ax_ecg.plot(
        t_ecg_g, ecg_filtered_g, label="ECG (0.5-100Hz)", color="tomato", alpha=0.8
    )
    ax_ecg.plot(
        t_ecg_g[r_peaks_g],
        ecg_filtered_g[r_peaks_g],
        "x",
        color="black",
        markersize=5,
        label="R-peaks",
    )

    ax_ecg.set_ylabel("ECG Amplitude [mV]", color="tomato")
    ax_ecg.tick_params(axis="y", labelcolor="tomato")
    ax_ecg.set_ylim(-4000, 4000)
    lines, labels = ax_main.get_legend_handles_labels()
    ax_ecg.legend(lines, labels, loc="upper right")
    ax_main.grid(True)

    span_selector = SpanSelector(
        ax_main,
        onselect_initial,
        "horizontal",
        useblit=False,
        interactive=True,
        props=dict(alpha=0.3, facecolor="lightgreen"),
    )

    try:
        pass
    except AttributeError:
        pass
    plt.show()
