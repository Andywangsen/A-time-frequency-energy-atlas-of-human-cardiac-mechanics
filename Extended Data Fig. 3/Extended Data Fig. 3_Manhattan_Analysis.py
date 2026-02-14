# -*- coding: utf-8 -*-
"""
Extended Data Fig. 3: Manhattan Plotting Tool

Data Privacy and Reproducibility Statement:
Due to the patient privacy and data sensitivity of the original clinical data
(such as high-precision ECG/SCG waveforms), and in accordance with ethical review
and privacy protection regulations, the original dataset cannot be publicly shared.
To ensure the transparency and reproducibility of the research results, we have provided
the original analysis code and the code for generating the figures consistent with
the paper using the intermediate result files (.pkl) after processing and feature
extraction.

Generate Manhattan plots for multiple comparisons to display feature significance.
Supports processing data from multiple folders, calculating statistical significance and AUC.

Main Features:
- Feature classification and statistical significance calculation
- Process Excel data from multiple folders
- Generate high-quality Manhattan plots
- Highlight key features and color strips
- Generate publication-quality PNG images
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.metrics import roc_auc_score

PLOT_ORDER = [
    "Ctrl-H vs HF",
    "CVD vs HF",
    "CVD-LRRCV vs HF-LRRCV",
    "CVD-HRRCV vs HF-HRRCV",
    "HFpEF vs HFrEF",
]

HIGHLIGHTS = {
    "Ctrl-H vs HF": {
        "H_{15-80Hz}[37-70%]": "H$_{{15-80Hz}}$[37-70%]",
        "H_{2-15Hz}[40-100%]": "H$_{{2-15Hz}}$[40-100%]",
        "ER_{2-100Hz}[0-20/0-30%]": "ER$_{{2-100Hz}}$[0-20/0-30%]",
    },
    "CVD vs HF": {
        "E_{2-100Hz}[0-10%]": "E$_{{2-100Hz}}$[0-10%]",
        "H_{15-80Hz}[39-70%]": "H$_{{15-80Hz}}$[39-70%]",
        "ER_{2-100Hz}[0-18/0-35%]": "ER$_{{2-100Hz}}$[0-18/0-35%]",
    },
    "HFpEF vs HFrEF": {
        "E_{2-12Hz}[40-100%]": "E$_{{2-12Hz}}$[40-100%]",
        "H_{10-80Hz}[39-70%]": "H$_{{10-80Hz}}$[39-70%]",
        "E_{2-100Hz}[0-10%]": "E$_{{2-100Hz}}$[0-10%]",
    },
    "CVD-HRRCV vs HF-HRRCV": {
        "E_{2-100Hz}[0-30%]": "E$_{{2-100Hz}}$[0-30%]",
        "E_{12-80Hz}[39-70%]": "E$_{{12-80Hz}}$[39-70%]",
        "ER_{2-100Hz}[0-21/0-28%]": "ER$_{{2-100Hz}}$[0-21/0-28%]",
    },
    "CVD-LRRCV vs HF-LRRCV": {
        "E_{2-100Hz}[0-10%]": "E$_{{2-100Hz}}$[0-10%]",
        "ER_{2-100Hz}[0-16/0-34%]": "ER$_{{2-100Hz}}$[0-16/0-34%]",
        "H_{2-13Hz}[36-100%]": "H$_{{2-13Hz}}$[36-100%]",
    },
}

HIGH_CONTRAST_COLORS = [
    "#E6194B",
    "#3CB44B",
    "#B58900",
    "#4363D8",
    "#F58231",
    "#911EB4",
    "#42D4F4",
    "#F032E6",
    "#BFEF45",
    "#FABED4",
    "#469990",
    "#DCBEFF",
    "#9A6324",
    "#FFFAC8",
    "#800000",
    "#AAFFC3",
    "#808000",
]


try:
    plt.style.use("seaborn-whitegrid")
except:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("ggplot")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


def get_feature_category_15(feature_name):
    """
    Return category based on feature name.

    Args:
        feature_name: Feature name string

    Returns:
        tuple: (Category_ID, Category_Name)
    """
    name = feature_name.strip()
    clean_name = name.replace("_grid", "").replace("_sensitivity", "")

    # 1. ER Class (Must be checked before E because ER starts with E)
    if clean_name.startswith("ER"):
        return 2, "ER$_{burst}$\n(Sys.ER)"

    # 2. All Freq Class (2-100Hz)
    if "{2-100Hz}" in clean_name:
        if clean_name.startswith("E"):
            return 1, "E$_{impulse}$\n(Sys.Mag.)"
        if clean_name.startswith("H"):
            return 3, "H$_{impulse}$\n(Sys.Mag.)"

    # 3. Stiffness (Mid-High Freq, e.g. 15-80Hz, 12-80Hz, 10-80Hz)
    # Check for 80Hz closing bracket
    if "80Hz}" in clean_name:
        if clean_name.startswith("E"):
            return 6, "E$_{stiff}$\n(Dias.Mid-High)"
        if clean_name.startswith("H"):
            return 7, "H$_{stiff}$\n(Dias.Mid-High)"

    # 4. Load (Low Freq / Remaining)
    if clean_name.startswith("E"):
        return 5, "E$_{load}$\n(Dias.Low)"

    if clean_name.startswith("H"):
        return 4, "H$_{load}$\n(Dias.Low)"

    return 8, "8. Others"


def calculate_significance(df, feature_cols, label_col):
    """
    Calculate P-value and AUC for each feature.

    Use Mann-Whitney U test for statistical significance testing,
    calculate Area Under ROC Curve (AUC) as feature discrimination metric.

    Args:
        df: DataFrame containing features and labels
        feature_cols: List of feature column names
        label_col: Label column name

    Returns:
        DataFrame: Statistical results including Feature, P-value, AUC, etc.
    """
    results = []
    classes = df[label_col].unique()

    if len(classes) != 2:
        if len(classes) > 2:
            classes = classes[:2]
        else:
            return pd.DataFrame()

    group1 = df[df[label_col] == classes[0]]
    group2 = df[df[label_col] == classes[1]]

    for feature in feature_cols:
        try:
            val1 = pd.to_numeric(group1[feature], errors="coerce").dropna()
            val2 = pd.to_numeric(group2[feature], errors="coerce").dropna()

            if len(val1) == 0 or len(val2) == 0:
                continue

            stat, p_val = stats.mannwhitneyu(val1, val2, alternative="two-sided")

            y_true = np.concatenate([np.zeros(len(val1)), np.ones(len(val2))])
            y_scores = np.concatenate([val1, val2])
            try:
                auc = roc_auc_score(y_true, y_scores)
            except:
                auc = 0.5

            if auc < 0.5:
                auc = 1 - auc

            if p_val == 0:
                log_p = 300
            else:
                log_p = -np.log10(p_val)

            cat_id, cat_name = get_feature_category_15(feature)

            results.append(
                {
                    "Feature": feature,
                    "Category_ID": cat_id,
                    "Category_Name": cat_name,
                    "P-value": p_val,
                    "-log10(P)": log_p,
                    "AUC": auc,
                }
            )
        except:
            pass

    return pd.DataFrame(results)


def rename_feature_on_load(name):
    """
    Standardize feature names to Feature_{Freq}[Range] format.
    """
    if "_{" in name and "}[" in name:
        return name

    # ER pattern: ER_{Freq}(Range) -> ER_{Freq}[Range]
    match_er = re.match(r"ER_\{(.+?)\}\((.+?)\)", name)
    if match_er:
        freq = match_er.group(1)
        rng_raw = match_er.group(2)
        parts = rng_raw.split("/")
        new_parts = []
        for part in parts:
            try:
                subparts = part.split("-")
                if len(subparts) == 2:
                    s, e = float(subparts[0]), float(subparts[1])
                    if s > 100 or e > 100:
                        new_parts.append(f"{s / 10:g}-{e / 10:g}")
                    else:
                        new_parts.append(f"{s:g}-{e:g}")
                else:
                    new_parts.append(part)
            except:
                new_parts.append(part)
        rng = "/".join(new_parts) + "%"
        freq_new = "2-100Hz" if freq == "0-100Hz" else freq
        return f"ER_{{{freq_new}}}[{rng}]"

    # E/H pattern: E(Range, Freq) -> E_{Freq}[Range]
    match_eh = re.match(r"([EH])\((.+?), (.+?)\)", name)
    if match_eh:
        type_prefix = match_eh.group(1)
        rng_raw = match_eh.group(2)
        freq_raw = match_eh.group(3)

        freq_new = freq_raw
        if freq_raw == "All Freq":
            freq_new = "2-100Hz"
        elif not freq_raw.endswith("Hz"):
            if re.match(r"^\d+-\d+$", freq_raw):
                freq_new = f"{freq_raw}Hz"

        rng_new = rng_raw
        try:
            parts = rng_raw.split("-")
            if len(parts) == 2:
                start = float(parts[0])
                end = float(parts[1])
                if start > 100 or end > 100:
                    rng_new = f"{start / 10:g}-{end / 10:g}%"
                else:
                    rng_new = f"{start / 10:g}-{end / 10:g}%"
        except:
            pass

        return f"{type_prefix}_{{{freq_new}}}[{rng_new}]"

    return name


def process_folder(folder_name):
    """
    Process data from a single folder.

    Workflow:
    1. Locate folder
    2. Find Excel files
    3. Read two Excel files
    4. Merge data and label classes
    5. Identify feature columns
    6. Data cleaning
    7. Calculate statistics

    Args:
        folder_name: Folder name

    Returns:
        DataFrame: Statistics results, or None if failed
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Search recursively for the folder
    folder_path = None
    # Search in current dir and parent dir, and their subdirs
    search_roots = [current_dir, parent_dir]

    for root_dir in search_roots:
        for root, dirs, files in os.walk(root_dir):
            if folder_name in dirs:
                folder_path = os.path.join(root, folder_name)
                break
        if folder_path:
            break

    if not folder_path or not os.path.exists(folder_path):
        # Last resort: try just joining with current dir (cwd)
        cwd_path = os.path.join(os.getcwd(), folder_name)
        if os.path.exists(cwd_path):
            folder_path = cwd_path
        else:
            print(
                f"! Folder not found: {folder_name} (searched in {current_dir} and {parent_dir})"
            )
            return None

    xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith("~$")]

    if len(xlsx_files) < 2:
        print(f"! Less than 2 Excel files found in {folder_name}, skipping.")
        return None

    file1, file2 = xlsx_files[:2]
    print(
        f"  Processing {folder_name}: {os.path.basename(file1)} vs {os.path.basename(file2)}"
    )

    try:
        df1 = pd.read_excel(file1)
        df1["Label"] = "Class0"
        df2 = pd.read_excel(file2)
        df2["Label"] = "Class1"

        # Rename columns on load
        df1.columns = [rename_feature_on_load(c) for c in df1.columns]
        df2.columns = [rename_feature_on_load(c) for c in df2.columns]

        df_all = pd.concat([df1, df2], ignore_index=True)

        exclude_cols = {
            "Label",
        }
        feature_cols = [
            c
            for c in df_all.columns
            if c not in exclude_cols and not c.startswith("Unnamed")
        ]

        df_features = df_all[feature_cols].apply(pd.to_numeric, errors="coerce")
        df_clean = df_all[df_features.notna().all(axis=1)].copy()

        stats_df = calculate_significance(df_clean, feature_cols, "Label")
        return stats_df

    except Exception as e:
        print(f"[X] Error processing {folder_name}: {e}")
        return None


def main():
    """
    Main Entry: Execute Manhattan plot generation workflow.

    Workflow:
    1. Process data from all folders
    2. Save data to pickle file
    3. Prepare color mapping
    4. Create subplots
    5. Plot scatter plots and highlight features
    6. Add color strips and legends
    7. Save chart
    8. Feature filtering and statistics
    """
    print("\n" + "=" * 60)
    print("Starting Manhattan plot generation...")
    print("=" * 60)

    plot_data = []

    for folder in PLOT_ORDER:
        df_stats = process_folder(folder)
        if df_stats is not None and not df_stats.empty:
            plot_data.append((folder, df_stats))

    if not plot_data:
        print("[X] No valid data generated. Please check folders and Excel files.")
        return

    print(f"\n[OK] Successfully processed {len(plot_data)} folders")

    import pickle

    pkl_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Extended_Data_Fig. 3_plot_data.pkl"
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(plot_data, f)
    print(f"[OK] Data saved: {pkl_path}")

    all_cat_ids = set()
    for _, df in plot_data:
        all_cat_ids.update(df["Category_ID"].dropna().unique())

    unique_cats = sorted(list(all_cat_ids))
    color_map = {
        cat_id: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)]
        for i, cat_id in enumerate(unique_cats)
    }
    print(f"[OK] Color mapping complete: {len(unique_cats)} categories")

    num_plots = len(plot_data)
    fig, axes = plt.subplots(
        nrows=num_plots,
        ncols=1,
        sharex=True,
        figsize=(20, 4 * num_plots),
        gridspec_kw={"hspace": 0.4},
    )
    if num_plots == 1:
        axes = [axes]

    legend_handles = {}

    print(f"[OK] Created {num_plots} subplots")

    for idx, (label, df) in enumerate(plot_data):
        ax = axes[idx]

        df = df.sort_values(by=["Category_ID", "Feature"]).reset_index(drop=True)
        df["x_coord"] = df.index

        for cat_id in sorted(df["Category_ID"].unique()):
            data_cat = df[df["Category_ID"] == cat_id]
            cat_name = str(data_cat["Category_Name"].iloc[0])
            color = color_map.get(cat_id, "grey")

            sc = ax.scatter(
                data_cat["x_coord"],
                data_cat["-log10(P)"],
                c=[color],
                label=cat_name,
                alpha=1,
                s=20,
                edgecolors="none",
                zorder=1,
            )
            if cat_name not in legend_handles:
                legend_handles[cat_name] = sc

        specified_features_config = HIGHLIGHTS.get(label, {})
        target_indices = []

        if specified_features_config:
            if isinstance(specified_features_config, dict):
                specified_features = list(specified_features_config.keys())
            else:
                specified_features = specified_features_config

            target_data = df[df["Feature"].isin(specified_features)]
            if target_data.empty:
                print(
                    f"  Warning: Specified highlight feature not found, stars will not be shown."
                )
            else:
                target_indices = target_data.index
        else:
            top_per_cat = df.loc[df.groupby("Category_ID")["-log10(P)"].idxmax()]
            top_features = top_per_cat.nlargest(3, "-log10(P)")
            target_indices = top_features.index

        targets = df.loc[target_indices]
        for i, (_, row) in enumerate(targets.iterrows()):
            feature_name = row["Feature"]
            display_label = feature_name
            if (
                isinstance(specified_features_config, dict)
                and feature_name in specified_features_config
            ):
                display_label = specified_features_config[feature_name]

            ax.annotate(
                display_label,
                xy=(row["x_coord"], row["-log10(P)"]),
                xytext=(
                    row["x_coord"],
                    row["-log10(P)"] + (df["-log10(P)"].max() * 0.15),
                ),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
                fontsize=11,
                fontweight="bold",
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="gold", lw=2
                ),
                zorder=10,
            )
            ax.scatter(
                row["x_coord"],
                row["-log10(P)"],
                c="gold",
                s=300,
                edgecolors="black",
                linewidth=1.5,
                marker="*",
                zorder=10,
            )

        threshold = -np.log10(0.001)
        ax.axhline(
            y=threshold,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=0,
        )

        ylabel = f"{label}\n-log$_{{10}}$(P-value)"
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ymax = df["-log10(P)"].max() if not df.empty else 10
        ax.set_ylim(0, ymax * 1.5)

        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)
        ax.grid(False, axis="x")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1)

        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if idx == num_plots - 1:
            cat_groups = df.groupby("Category_ID")["x_coord"].agg(["min", "max"])
            strip_height = ymax * 0.08
            ax.set_ylim(-strip_height, ymax * 1.5)

            for cat_id in unique_cats:
                if cat_id not in cat_groups.index:
                    continue
                xmin = cat_groups.loc[cat_id, "min"]
                xmax = cat_groups.loc[cat_id, "max"]
                width = xmax - xmin + 1
                center = (xmin + xmax) / 2
                color = color_map[cat_id]

                rect = mpatches.Rectangle(
                    (xmin - 0.5, -strip_height),
                    width,
                    strip_height,
                    facecolor=color,
                    edgecolor="none",
                    alpha=1.0,
                    zorder=0,
                )
                ax.add_patch(rect)

                display_num = unique_cats.index(cat_id) + 1

                rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
                brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
                txt_color = "black" if brightness > 128 else "white"

                ax.text(
                    center,
                    -strip_height / 2,
                    str(display_num),
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontweight="bold",
                    fontsize=10,
                )

            ax.set_xlabel("Feature Categories (Color Strips)", fontsize=14, labelpad=10)

    id_name_map = {}
    for _, df in plot_data:
        for cat_id in df["Category_ID"].unique():
            if cat_id not in id_name_map:
                name = df[df["Category_ID"] == cat_id]["Category_Name"].iloc[0]
                id_name_map[cat_id] = name

    custom_handles = []
    custom_labels = []
    for i, cat_id in enumerate(unique_cats):
        color = color_map[cat_id]
        name = id_name_map.get(cat_id, f"Cat {cat_id}")
        clean_name = str(name).split(". ", 1)[-1]

        patch = mpatches.Patch(color=color, label=clean_name)
        custom_handles.append(patch)
        custom_labels.append(f"[{i + 1}] {clean_name}")

    fig.legend(
        custom_handles,
        custom_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=11,
        frameon=False,
        title="Feature Categories",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = "Extended Data Fig. 3_Combined_Manhattan_Standalone.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Chart saved: {os.path.abspath(save_path)}")

    print("\n" + "=" * 60)
    print("Manhattan plot generation complete!")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
