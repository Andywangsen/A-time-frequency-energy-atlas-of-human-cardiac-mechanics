"""
==============================================================================
Figure 3-c: SHAP Swarm Plot for Feature Importance Analysis

Generates SHAP (SHapley Additive exPlanations) swarm plots to visualize
feature importance and impact on model predictions. Compares cardiac features
between healthy controls and heart failure patients using Random Forest
classification with TreeExplainer for interpretability.

Key components:
- Loads data from Excel files (Ctrl-H and HF groups)
- Trains Random Forest classifier on standardized features
- Computes SHAP values using TreeExplainer (optimized for tree models)
- Visualizes feature contributions with swarm plot
- Displays model accuracy and feature importance rankings
==============================================================================
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import os

FIG_WIDTH = 8.6
FIG_HEIGHT = 7.6

FEATURE_1 = "ER_{2-100Hz}[0-20/0-30%]"
FEATURE_2 = "H_{2-15Hz}[40-100%]"
FEATURE_3 = "H_{15-80Hz}[37-70%]"

FEATURE_DISPLAY_NAMES = {
    "ER_{2-100Hz}[0-20/0-30%]": "Systolic Ejection Burst\n$ER_{{2-100Hz}}$[0-20/0-30%]",
    "H_{2-15Hz}[40-100%]": "Diastolic Filling Load\nH$_{{2-15Hz}}$[40-100%]",
    "H_{15-80Hz}[37-70%]": "Myocardial Stiffness Tone\nH$_{{15-80Hz}}$[37-70%]",
}


def select_files():
    """
    Automatically locate Excel data files in the script directory.

    Returns:
        tuple: (ctrl_h_file, hf_file) paths or None if files not found
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ctrl_h_file = os.path.join(script_dir, "Fig. 3c,d_Ctrl-H.xlsx")
    hf_file = os.path.join(script_dir, "Fig. 3c,d_HF.xlsx")

    if not os.path.exists(ctrl_h_file):
        ctrl_h_file = None
    else:
        pass

    if not os.path.exists(hf_file):
        hf_file = None
    else:
        pass

    return ctrl_h_file, hf_file


def load_data(file_path):
    """
    Load data from Excel file.

    Args:
        file_path (str): Path to Excel file

    Returns:
        pd.DataFrame: Loaded data
    """
    df = pd.read_excel(file_path)
    return df


def get_features(df, features):
    """
    Extract feature columns from dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list): Requested feature names

    Returns:
        list: Available feature names (fallback to numeric columns if needed)
    """
    available = [f for f in features if f in df.columns]
    if len(available) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    return available


def plot_shap_summary(ctrl_h_df, hf_df, features, output_dir):
    """
    Generate SHAP swarm plot for feature importance visualization.

    Trains Random Forest classifier and computes SHAP values using TreeExplainer.
    Creates publication-quality swarm plot showing feature contributions.

    Args:
        ctrl_h_df (pd.DataFrame): Healthy control group data
        hf_df (pd.DataFrame): Heart failure group data
        features (list): Feature names to analyze
        output_dir (str): Directory for saving output PNG
    """
    try:
        import shap

        use_shap_library = True
    except ImportError:
        use_shap_library = False

    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    f1, f2, f3 = features[0], features[1], features[2]

    print("[1/5] Loading and preparing data...")
    X_ctrl = ctrl_h_df[[f1, f2, f3]].dropna().values
    X_hf = hf_df[[f1, f2, f3]].dropna().values
    print(f"  Healthy samples: {len(X_ctrl)}, HF samples: {len(X_hf)}")

    X = np.vstack([X_ctrl, X_hf])
    y = np.array([0] * len(X_ctrl) + [1] * len(X_hf))

    print("[2/5] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[3/5] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=1000, random_state=42, max_depth=8, class_weight="balanced"
    )
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)

    if use_shap_library:
        print("[4/5] Computing SHAP values with TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.sca(ax)

        display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=display_names,
            show=False,
            plot_size=None,
            cmap="viridis",
            sort=True,
        )

        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)

        ax.set_xlim(-0.5, 0.6)

        ax.set_title("SHAP Summary Plot", fontsize=14, pad=20)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        label_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05

        arrow_y = label_y - 0.1
        text_y = label_y

        ax.text(
            xlim[0] + (xlim[1] - xlim[0]) * 0.15,
            text_y,
            "Pro-Healthy",
            fontsize=12,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax.annotate(
            "",
            xy=(xlim[0] + (xlim[1] - xlim[0]) * 0.05, arrow_y),
            xytext=(xlim[0] + (xlim[1] - xlim[0]) * 0.25, arrow_y),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

        ax.text(
            xlim[1] - (xlim[1] - xlim[0]) * 0.15,
            text_y,
            "Pro-HF",
            fontsize=12,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax.annotate(
            "",
            xy=(xlim[1] - (xlim[1] - xlim[0]) * 0.05, arrow_y),
            xytext=(xlim[1] - (xlim[1] - xlim[0]) * 0.25, arrow_y),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

    print("[5/5] Saving visualization...")
    plt.tight_layout()

    png_path = os.path.join(output_dir, "Fig. 3c_SHAP.png")
    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {png_path}")

    if use_shap_library:
        feature_importance = np.abs(shap_values).mean(axis=0)
        print("\nFeature importance ranking:")
        for feat, imp in sorted(
            zip(features, feature_importance), key=lambda x: x[1], reverse=True
        ):
            print(f"  {feat}: {imp:.4f}")

    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main execution function.

    Orchestrates the complete SHAP analysis pipeline:
    1. Locate and load data files
    2. Extract features
    3. Generate SHAP swarm plot
    """
    print("=" * 60)
    print("SHAP Swarm Plot Analysis - Figure 3-c")
    print("=" * 60)

    ctrl_h_file, hf_file = select_files()

    if not ctrl_h_file or not hf_file:
        print("Error: Required data files not found")
        return

    print("\nLoading data files...")
    ctrl_h_df = load_data(ctrl_h_file)
    hf_df = load_data(hf_file)
    print(f"  Ctrl-H: {ctrl_h_df.shape[0]} samples, {ctrl_h_df.shape[1]} features")
    print(f"  HF: {hf_df.shape[0]} samples, {hf_df.shape[1]} features")

    features = get_features(ctrl_h_df, [FEATURE_1, FEATURE_2, FEATURE_3])

    if len(features) < 3:
        print("Error: Insufficient features (need at least 3)")
        return

    print(f"\nSelected features: {features}")

    output_dir = os.path.dirname(ctrl_h_file)

    print("\nGenerating SHAP analysis...")
    plot_shap_summary(ctrl_h_df, hf_df, features, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
