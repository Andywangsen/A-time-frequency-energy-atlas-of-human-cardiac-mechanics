"""
==============================================================================
Time-Frequency Analysis and Ridge Plot Visualization for Cardiac Signals

This module processes ECG and SCG (seismocardiogram) signals to generate
time-frequency atlases and ridge plot visualizations. It implements:
- R-peak detection from ECG signals
- Synchrosqueezed Stockwell Transform (SSTF) for time-frequency analysis
- Energy-based background filtering
- Ridge plot generation with normalized cardiac cycle representation

The pipeline processes multiple data files and generates publication-quality
visualizations of cardiac signal energy distribution across time and frequency.
==============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import zoom

mplstyle.use("fast")

# Visualization and processing configuration
SHOW_PLOTS_AFTER_ANALYSIS = False

# Time-frequency grid normalization parameters
NORMALIZED_TIME_POINTS = 1000  # Normalized time resolution for atlas
NORMALIZED_FREQ_BINS = 256  # Normalized frequency resolution for atlas

# Energy visualization scaling parameters
ENERGY_VMAX = 15  # Maximum energy for standard atlas
CWTENERGY_VMAX = 140  # Maximum energy for CWT visualization
PJENERGY_VMAX = 0.0000600  # Maximum energy for ridge plot display

# Energy unit scaling for colorbar labels
ENERGY_UNIT_SCALE = "e-6"

# Colorbar configuration
COLORBAR_START_PERCENT = 0  # Colormap truncation percentage

# Ridge plot grid configuration
GRID_X_DTICK = 200  # X-axis tick spacing
GRID_Z_NTICKS = 4  # Number of colorbar ticks

# Font sizes for figure elements
FIG5_2FONT_LABEL = 12
FIG5_2FONT_TICK = 10
FIG5_2FONT_LEGEND = 10

# Ridge plot visualization parameters
RIDGELINE_COUNT = 160  # Number of frequency ridges to display
RIDGELINE_Y_SPACING = 60  # Vertical spacing between ridges
RIDGELINE_ALPHA = 1  # Ridge transparency
RIDGELINE_LINE_COLOR = "black"  # Ridge line color
RIDGELINE_LINE_WIDTH = 0.3  # Ridge line width
RIDGELINE_LINE_ALPHA = 0.2  # Ridge line transparency
RIDGELINE_REVERSE_ORDER = False  # Ridge drawing order

# Check for optional SSTF library availability
try:
    from ssqueezepy import ssq_cwt

    SSQUEEZEPY_AVAILABLE = True
except ImportError:
    SSQUEEZEPY_AVAILABLE = False

mplstyle.use("fast")

# Configure matplotlib for publication-quality output
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.autolayout"] = True

# Signal color scheme for visualization
COLOR_ECG = "#333333"  # Dark gray for ECG
COLOR_SCG = "#009E73"  # Green for SCG
COLOR_R_PEAK = "black"  # Black for R-peak markers


def convert_scg_unit(raw_scg, unit="ms2"):
    """
    Convert raw SCG acceleration data to physical units.

    Parameters:
        raw_scg: Raw SCG ADC values (array-like)
        unit: Target unit - "g" (gravitational), "ms2" (m/s^2), or "mg" (milligravity)

    Returns:
        Converted acceleration values in specified unit
    """
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
    """
    Convert raw ECG ADC values to voltage units.

    Parameters:
        raw_ecg: Raw ECG ADC values (array-like)
        unit: Target unit - "mV" (millivolts) or "uV" (microvolts)

    Returns:
        Converted voltage values in specified unit
    """
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
    """
    Create custom spectral colormap for time-frequency visualization.

    Parameters:
        start_percent: Colormap truncation percentage (0-100)

    Returns:
        LinearSegmentedColormap object with purple-to-red gradient
    """
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
    """
    Apply zero-phase digital filter to signal.

    Parameters:
        data: Input signal (array-like)
        fs: Sampling frequency (Hz)
        f_cut: Cutoff frequency or [low, high] for bandpass (Hz)
        btype: Filter type - "bandpass", "highpass", or "lowpass"

    Returns:
        Filtered signal with zero phase distortion
    """
    data = np.array(data, dtype=float)
    order = 2
    b, a = butter(order, np.array(f_cut) / (fs / 2), btype=btype)
    return filtfilt(b, a, data)


def detect_r_peaks(data_ecg_raw, ecg_filtered, fs_ecg):
    """
    Detect R-peaks in ECG signal using multi-stage algorithm.

    Parameters:
        data_ecg_raw: Raw ECG signal (array-like)
        ecg_filtered: Pre-filtered ECG signal (array-like)
        fs_ecg: ECG sampling frequency (Hz)

    Returns:
        Array of R-peak indices in the ECG signal

    Algorithm:
        Stage 1: Derivative-based peak detection with windowed integration
        Stage 2: Candidate peak finding with distance constraints
        Stage 3: Refinement with height and distance thresholds
        Stage 4: Final validation with amplitude thresholds
    """
    # Stage 1: Compute derivative and squared signal for peak detection
    data_ecg_for_deriv = fil(data_ecg_raw, fs_ecg, [10, 30])
    deriv_ecg = np.convolve(
        data_ecg_for_deriv, np.array([1, 2, 0, -2, -1]) * (fs_ecg / 8.0), mode="same"
    )
    squared_ecg = deriv_ecg**2

    # Stage 2: Windowed integration for peak prominence
    window_size = int(0.15 * fs_ecg)
    cumsum = np.cumsum(np.insert(squared_ecg, 0, 0))
    integrated_ecg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    pad_before = (window_size - 1) // 2
    integrated_ecg = np.pad(integrated_ecg, (pad_before, window_size // 2), mode="edge")

    # Stage 3: Candidate peak detection in sliding windows
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

    # Stage 4: Final validation and deduplication
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
    """
    Calculate average time-frequency atlas for cardiac cycles.

    Parameters:
        scg_full: Complete SCG signal (array-like)
        r_peaks_indices_ecg: R-peak indices from ECG (array-like)
        fs_scg: SCG sampling frequency (Hz)
        fs_ecg: ECG sampling frequency (Hz)

    Returns:
        Normalized average time-frequency atlas matrix or None if insufficient data

    Process:
        1. Extract individual cardiac cycles between consecutive R-peaks
        2. Apply Synchrosqueezed Stockwell Transform (SSTF) to each cycle
        3. Normalize frequency bins to logarithmic scale (2-100 Hz)
        4. Interpolate and resample to standard time-frequency grid
        5. Average all normalized beats to create atlas
    """
    if len(r_peaks_indices_ecg) < 2:
        return None

    r_peaks_indices_scg = (r_peaks_indices_ecg / fs_ecg * fs_scg).astype(int)
    final_freqs = np.logspace(np.log10(2), np.log10(100), NORMALIZED_FREQ_BINS)
    normalized_beats_sst = []

    # Extract and process individual cardiac cycles
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
            # Apply SSTF to extract time-frequency representation
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
        except Exception as e_beat:
            continue

    if not normalized_beats_sst:
        return None

    # Average all normalized beats to create atlas
    average_beat_atlas = np.mean(np.array(normalized_beats_sst), axis=0)
    return average_beat_atlas


def calculate_percentile_threshold(atlas_matrix, percentile=50):
    """
    Calculate energy-based threshold for background/foreground separation.

    Parameters:
        atlas_matrix: 2D time-frequency energy matrix
        percentile: Energy percentile threshold (0-100)

    Returns:
        Dictionary containing threshold statistics and energy split information

    Method:
        Sorts all energy values and finds threshold where cumulative energy
        reaches the specified percentile of total energy.
    """
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


def plot_Fig1_i_with_marginals(
    atlas_matrix,
    title_prefix="",
    apply_filter=False,
    bg_stats=None,
    vmax_unified=None,
    annotation_text=None,
):
    """
    Create ridge plot visualization of time-frequency atlas.

    Parameters:
        atlas_matrix: 2D time-frequency energy matrix
        title_prefix: Title text for the plot
        apply_filter: Whether to apply energy-based background filtering
        bg_stats: Background statistics dictionary from calculate_percentile_threshold
        vmax_unified: Maximum energy value for colorbar scaling

    Returns:
        (figure, normalized_atlas_matrix) - matplotlib figure and normalized data

    Visualization:
        Creates ridge plot with frequency ridges colored by energy intensity,
        normalized cardiac cycle on x-axis (0-100%), frequency on y-axis (2-100 Hz).
    """
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

    # Generate ridge line indices for visualization
    ridgeline_indices = np.linspace(0, n_freqs - 1, RIDGELINE_COUNT, dtype=int)

    t_axis = np.linspace(0, 1000, n_times)

    global_max_energy = np.max(normalized_atlas_matrix)

    if isinstance(cmap, str):
        cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Extract ridge segments for each frequency
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

    # Render ridge polygons with gradient coloring
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

    # Configure x-axis (normalized cardiac cycle)
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

    # Configure y-axis (frequency)
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

    # Configure colorbar with energy scale
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
    # Verify SSTF library availability
    if not SSQUEEZEPY_AVAILABLE:
        print(
            "Error: ssqueezepy library not available. Install with: pip install ssqueezepy"
        )
        exit()

    # List of data files to process
    file_list = [
        "Fig. 2c.txt",
        "Fig. 2d.txt",
        "Fig. 2e.txt",
        "Fig. 2f.txt",
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Time window for analysis (seconds)
    xmin, xmax = 5, 30
    filter_percentile = 50.0

    # Process each data file
    for filename in file_list:
        file_path = os.path.join(current_dir, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {filename}: file not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {filename}")
        print(f"{'=' * 60}")

        # Signal processing parameters
        fs_ecg_g, fs_scg_g = 250, 500
        len_ecg, len_accz = [181, 240], [1, 120]
        cut_ecg, cut_accz = [0.5, 100], [2, 100]
        cut_scg = [20, 50]
        len_acc = [121, 180]
        fs_acc = 50

        cut_acc_1 = 20
        cut_acc_2 = 2

        try:
            # Load raw data from text file
            print(f"  Loading data from {filename}...")
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
                print(f"  Warning: No valid data loaded from {filename}")
                continue

            print(f"  Data shape: {data_raw.shape}")

            # Extract and filter ECG signal
            data_ecg_g = data_raw[:, len_ecg[0] - 1 : len_ecg[1]].flatten()
            ecg_filtered_g = fil(data_ecg_g, fs_ecg_g, cut_ecg)
            t_ecg_g = np.arange(len(ecg_filtered_g)) / fs_ecg_g

            # Extract and filter SCG signal
            data_accz_raw = data_raw[:, len_accz[0] - 1 : len_accz[1]].flatten()
            data_scg_g = fil(data_accz_raw, fs_scg_g, cut_accz, btype="bandpass")
            datafiltered_scg_g = fil(data_accz_raw, fs_scg_g, cut_scg, btype="bandpass")

            print(f"  ECG signal length: {len(data_ecg_g)} samples")
            print(f"  SCG signal length: {len(data_scg_g)} samples")

        except Exception as e:
            print(f"  Error loading data: {e}")
            continue

        # Detect R-peaks in ECG signal
        print(f"  Detecting R-peaks...")
        r_peaks_g = detect_r_peaks(data_ecg_g, ecg_filtered_g, fs_ecg_g)
        print(f"  R-peaks detected: {len(r_peaks_g)}")

        if len(r_peaks_g) < 3:
            print(f"  Warning: Insufficient R-peaks detected")
            continue

        # Filter R-peaks within analysis window
        r_peaks_in_range = r_peaks_g[
            (t_ecg_g[r_peaks_g] >= xmin) & (t_ecg_g[r_peaks_g] <= xmax)
        ]
        print(f"  R-peaks in analysis window ({xmin}-{xmax}s): {len(r_peaks_in_range)}")

        if len(r_peaks_in_range) < 2:
            print(f"  Warning: Insufficient R-peaks in analysis window")
            continue

        start_idx_scg = int(xmin * fs_scg_g)
        end_idx_scg = int(xmax * fs_scg_g)

        # Calculate average time-frequency atlas
        print(f"  Calculating time-frequency atlas...")
        avg_atlas = calculate_average_beat_atlas(
            data_scg_g, r_peaks_in_range, fs_scg_g, fs_ecg_g
        )

        if avg_atlas is not None:
            avg_atlas = np.maximum(avg_atlas, 0)
            total_energy = np.sum(avg_atlas)
            normalized_atlas = (
                avg_atlas / total_energy if total_energy > 1e-9 else avg_atlas
            )

            # Calculate energy threshold for background filtering
            print(
                f"  Calculating energy threshold ({filter_percentile}% percentile)..."
            )
            bg_stats = calculate_percentile_threshold(
                normalized_atlas, percentile=filter_percentile
            )

            unified_vmax = np.max(normalized_atlas)

            # Generate ridge plot visualization
            print(f"  Generating ridge plot...")

            annotation_text = None
            title_text = ""

            if "Fig. 2c.txt" in filename:
                title_text = "Healthy Individual Atlas #1"
            elif "Fig. 2d.txt" in filename:
                title_text = "HF Individual Atlas #1"
                annotation_text = "Sinus Rhythm"
            elif "Fig. 2e.txt" in filename:
                title_text = "Healthy Individual Atlas #2"
            elif "Fig. 2f.txt" in filename:
                title_text = "HF Individual Atlas #2"
                annotation_text = "Atrial Fibrillation"

            Fig1_i, _ = plot_Fig1_i_with_marginals(
                normalized_atlas,
                title_prefix=title_text,
                apply_filter=True,
                bg_stats=bg_stats,
                vmax_unified=unified_vmax,
                annotation_text=annotation_text,
            )

            if Fig1_i:
                base_name = os.path.splitext(filename)[0]
                output_png = os.path.join(current_dir, f"{base_name}.png")

                print(f"  Saving plot to {output_png}...")
                Fig1_i.savefig(output_png)
                plt.close(Fig1_i)
                print(f"  [OK] Successfully saved {base_name}.png")
        else:
            print(f"  Warning: Failed to calculate atlas for {filename}")

    print(f"\n{'=' * 60}")
    print("Processing complete")
    print(f"{'=' * 60}")
