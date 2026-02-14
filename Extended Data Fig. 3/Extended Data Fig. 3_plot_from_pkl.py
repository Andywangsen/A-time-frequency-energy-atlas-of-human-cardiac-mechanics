# -*- coding: utf-8 -*-
"""
==============================================================================
Extended Data Fig. 3: Manhattan Plotting Tool (From PKL)

Data Privacy and Reproducibility Statement:
Due to the patient privacy and data sensitivity of the original clinical data
(such as high-precision ECG/SCG waveforms), and in accordance with ethical review
and privacy protection regulations, the original dataset cannot be publicly shared.
To ensure the transparency and reproducibility of the research results, we have provided
the original analysis code and the code for generating the figures consistent with
the paper using the intermediate result files (.pkl) after processing and feature
extraction.

Load pre-calculated chart data from pickle file and generate publication-quality Manhattan plots.
Supports highlighting key features, color coding, and multi-panel display.

Main Features:
- Load pre-calculated data from pickle file
- Generate multi-panel Manhattan plots
- Highlight key features and color coding
- Generate publication-quality PNG images
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle

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
    "#E6194B",  # Red
    "#3CB44B",  # Green
    "#B58900",  # Dark Yellow/Gold
    "#4363D8",  # Blue
    "#F58231",  # Orange
    "#911EB4",  # Purple
    "#42D4F4",  # Cyan
    "#F032E6",  # Magenta
    "#BFEF45",  # Lime
    "#FABED4",  # Pink
    "#469990",  # Teal
    "#DCBEFF",  # Lavender
    "#9A6324",  # Brown
    "#FFFAC8",  # Beige
    "#800000",  # Maroon
    "#AAFFC3",  # Mint
    "#808000",  # Olive
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


def main():
    """
    Generate Manhattan plot from pickle file.

    Workflow:
    1. Load pre-calculated chart data
    2. Prepare color mapping and category information
    3. Create multi-panel chart
    4. Plot scatter plots and highlight features
    5. Add color strips and legends
    6. Save chart
    """
    print("\n" + "=" * 60)
    print("Starting Manhattan plot generation...")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, "Extended Data Fig. 3_plot_data.pkl")

    if not os.path.exists(pkl_path):
        print(f"! Pickle file not found: {pkl_path}")
        return

    print("Loading data from pickle file...")
    with open(pkl_path, "rb") as f:
        plot_data = pickle.load(f)

    if not plot_data:
        print("! Loaded data is empty")
        return

    all_cat_ids = set()
    for _, df in plot_data:
        all_cat_ids.update(df["Category_ID"].dropna().unique())

    unique_cats = sorted(list(all_cat_ids))
    print(
        f"[OK] Loaded {len(plot_data)} comparisons, containing {len(unique_cats)} feature categories"
    )

    color_map = {
        cat_id: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)]
        for i, cat_id in enumerate(unique_cats)
    }

    num_plots = len(plot_data)
    print(f"[OK] Creating {num_plots} subplots...")

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

    for idx, (label, df) in enumerate(plot_data):
        ax = axes[idx]
        print(f"  Plotting {idx + 1}/{num_plots}: {label}")

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
            if not target_data.empty:
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
        ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
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

    save_path = os.path.join(
        script_dir, "Extended Data Fig. 3_Combined_Manhattan_Standalone.png"
    )
    print(f"\n[OK] Saving chart to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print("\n" + "=" * 60)
    print("Manhattan plot generation complete!")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
