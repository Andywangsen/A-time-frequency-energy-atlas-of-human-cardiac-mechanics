# -*- coding: utf-8 -*-
"""
==============================================================================
Figure 4-c: Model Training and Internal Validation

Generates publication-quality ROC curves demonstrating Random Forest model performance in internal validation.
Supports multiple classification tasks, 5-fold cross-validation, and Bootstrap confidence interval calculation.

Key Features:
- Load and preprocess Excel data
- Perform 5-fold stratified cross-validation
- Calculate Bootstrap confidence intervals
- Generate random chance zone (1000 permutations)
- Calculate confusion matrix and clinical metrics
- Plot threshold tuning curves
- Generate publication-quality PNG and SVG images
==============================================================================
"""

import pandas as pd
import numpy as np
import random
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

import joblib

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42
img_path = "Fig. 4c_Combined_ROC_Analysis"
OUTPUT_DIR = "."
FAKE_ROUNDS = 1000
N_FOLDS = 5
FIG_WIDTH = 10
FIG_HEIGHT = 10
OUTPUT_DPI = 300
SAVE_CONFUSION_MATRIX = False
SAVE_SVG = False

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


def get_script_dir():
    """Get the absolute path of the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def get_model(model_name):
    """Factory function for model creation with consistent hyperparameters."""
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=1000,
            max_depth=8,
            random_state=GLOBAL_SEED,
            class_weight="balanced",
        )
    elif model_name == "SVM":
        return SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=GLOBAL_SEED,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def bootstrap_roc_ci(y_true, y_proba, n_bootstraps=1000, confidence=0.95):
    """
    Bootstrap 95% CI for ROC Curve.
    Returns: mean_fpr, mean_tpr, tpr_lower, tpr_upper, mean_auc, ci_lower, ci_upper
    """
    np.random.seed(GLOBAL_SEED)
    n_samples = len(y_true)

    # Unified grid for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for _ in range(n_bootstraps):
        # Bootstrap sampling
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true_boot, y_proba_boot)

        # Interp to grid
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        # AUC
        aucs.append(roc_auc_score(y_true_boot, y_proba_boot))

    tprs = np.array(tprs)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    alpha = (1 - confidence) / 2
    tpr_lower = np.percentile(tprs, alpha * 100, axis=0)
    tpr_upper = np.percentile(tprs, (1 - alpha) * 100, axis=0)

    mean_auc = np.mean(aucs)
    ci_lower = np.percentile(aucs, alpha * 100)
    ci_upper = np.percentile(aucs, (1 - alpha) * 100)

    return mean_fpr, mean_tpr, tpr_lower, tpr_upper, mean_auc, ci_lower, ci_upper


def run_experiment(file1, file2, model_name, features):
    """
    Loads data, preprocesses, runs 5-Fold CV, returns consolidated results.
    """
    print(f"  Processing: {file1} vs {file2} | Model: {model_name}")

    base_dir = get_script_dir()
    path1 = os.path.join(base_dir, file1)
    path2 = os.path.join(base_dir, file2)

    # Load Data
    try:
        df1 = pd.read_excel(path1)
        df2 = pd.read_excel(path2)
        df1["class"] = (
            0  # Assuming first file is Class 0 (or label encoding handles it)
        )
        df2["class"] = 1  # Assuming second file is Class 1

        # Actually use the filename (without ext) as category for LabelEncoder to be consistent with original script
        cat1 = os.path.splitext(os.path.basename(file1))[0].split("_")[1]
        cat2 = os.path.splitext(os.path.basename(file2))[0].split("_")[1]
        df1["class"] = cat1
        df2["class"] = cat2

        df_all = pd.concat([df1, df2], ignore_index=True)
    except Exception as e:
        print(f"Error loading files: {e}")
        return None

    # Check features
    missing = [f for f in features if f not in df_all.columns]
    if missing:
        print(f"    Missing features: {missing}")
        return None

    X = df_all[features].apply(pd.to_numeric, errors="coerce").values
    le = LabelEncoder()
    y = le.fit_transform(df_all["class"])

    # Drop NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"    Samples: {len(y)}, Class distribution: {np.bincount(y)}")

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=GLOBAL_SEED)
    y_proba_oof = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, _ = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = get_model(model_name)
        clf.fit(X_train_scaled, y_train)
        y_proba_oof[test_idx] = clf.predict_proba(X_test_scaled)[:, 1]

    # Train Final Model on Full Data
    print("    Training final model on full dataset...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    final_model = get_model(model_name)
    final_model.fit(X_scaled, y)

    # Bootstrap CI for the OOF predictions
    stats = bootstrap_roc_ci(y, y_proba_oof)

    # Raw ROC for non-smoothed plotting
    fpr_raw, tpr_raw, _ = roc_curve(y, y_proba_oof)

    return stats, (fpr_raw, tpr_raw), y, y_proba_oof, final_model, le.classes_


def calculate_fake_zone(y, y_proba_real, rounds=1000):
    """
    Calculates the random chance zone by permuting labels 1000 times.
    """
    print(f"  Calculating Random Chance Zone ({rounds} rounds)...")
    np.random.seed(GLOBAL_SEED)

    mean_fpr = np.linspace(0, 1, 100)
    fake_tprs = []
    fake_aucs = []

    for i in range(rounds):
        # Permute labels
        y_fake = np.random.permutation(y)

        # Calculate AUC for stats
        fake_aucs.append(roc_auc_score(y_fake, y_proba_real))

        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_fake, y_proba_real)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        fake_tprs.append(tpr_interp)

    fake_tprs = np.array(fake_tprs)
    fake_tpr_lower = np.percentile(fake_tprs, 2.5, axis=0)
    fake_tpr_upper = np.percentile(fake_tprs, 97.5, axis=0)

    # Calculate P-value for the real AUC against this null distribution
    real_auc = roc_auc_score(y, y_proba_real)
    n_better = sum(1 for auc in fake_aucs if auc >= real_auc)
    p_value = (n_better + 1) / (rounds + 1)

    return mean_fpr, fake_tpr_lower, fake_tpr_upper, p_value


def calculate_metrics(y_true, y_pred):
    """Calculate detailed clinical metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    ppv = precision_score(y_true, y_pred)  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


def plot_threshold_tuning_curves(y_true, y_probs, output_dir, title):
    """Draw Sensitivity and Specificity vs Threshold curves."""
    thresholds = np.linspace(0, 1, 101)
    sensitivities = []
    specificities = []

    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)

    # Find optimal threshold (Youden's J statistic)
    youden_index = np.array(sensitivities) + np.array(specificities) - 1
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    best_sens = sensitivities[best_idx]
    best_spec = specificities[best_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(
        thresholds, sensitivities, label="Sensitivity (TPR)", color="blue", linewidth=2
    )
    plt.plot(
        thresholds, specificities, label="Specificity (TNR)", color="green", linewidth=2
    )

    # Mark optimal point
    plt.axvline(
        best_threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold(Youden Index): {best_threshold:.2f}",
    )
    plt.scatter([best_threshold] * 2, [best_sens, best_spec], color="black", zorder=5)

    plt.title(f"{title} - Threshold Tuning", fontsize=14)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc="lower center", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Threshold_Tuning_{title}.png"), dpi=300)
    if SAVE_SVG:
        plt.savefig(
            os.path.join(output_dir, f"Threshold_Tuning_{title}.svg"), format="svg"
        )
    plt.close()

    return best_threshold


def plot_confusion_matrix(y_true, y_pred, output_dir, title, class_names):
    """Draw Confusion Matrix and Save Metrics to Excel."""
    if not SAVE_CONFUSION_MATRIX:
        return

    # 1. Save Metrics to Excel
    metrics = calculate_metrics(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Add raw counts to metrics
    metrics["TP"] = tp
    metrics["TN"] = tn
    metrics["FP"] = fp
    metrics["FN"] = fn

    df_metrics = pd.DataFrame([metrics])

    # Reorder columns for better readability if desired, but default is fine
    cols = ["TP", "TN", "FP", "FN", "Sensitivity", "Specificity", "PPV", "NPV", "F1"]
    # Ensure all columns exist (they should from calculate_metrics + local add)
    df_metrics = df_metrics[cols]  # Optional, might differ in specific names

    excel_path = os.path.join(output_dir, f"{title}_Metrics.xlsx")
    df_metrics.to_excel(excel_path, index=False)
    # print(f"Saved metrics to {excel_path}")

    # 2. Plot Simple Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14},
    )
    plt.title(title, fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}.png"), dpi=300, bbox_inches="tight")
    if SAVE_SVG:
        plt.savefig(
            os.path.join(output_dir, f"{title}.svg"), format="svg", bbox_inches="tight"
        )
    plt.close()


def main():
    # Update Output Directory to current directory
    global OUTPUT_DIR
    OUTPUT_DIR = "."

    # Configuration List (Preserving user's recent manual edits)
    experiments = [
        {
            "name": "CVD-LRRCV vs HF-LRRCV",
            "files": ("Fig. 4c_CVD-LRRCV.xlsx", "Fig. 4c_HF-LRRCV.xlsx"),
            "model": "RF",
            "features": [
                "E_{2-100Hz}[0-10%]",
                "ER_{2-100Hz}[0-16/0-34%]",
                "H_{2-13Hz}[36-100%]",
            ],
            "color": "royalblue",
        },
        {
            "name": "CVD-HRRCV vs HF-HRRCV",
            "files": ("Fig. 4c_CVD-HRRCV.xlsx", "Fig. 4c_HF-HRRCV.xlsx"),
            "model": "RF",
            "features": [
                "E_{2-100Hz}[0-30%]",
                "E_{12-80Hz}[39-70%]",
                "ER_{2-100Hz}[0-21/0-28%]",
            ],
            "color": "salmon",
        },
        {
            "name": "CVD vs HF",
            "files": ("Fig. 4c_CVD.xlsx", "Fig. 4c_HF.xlsx"),
            "model": "RF",
            "features": [
                "E_{2-100Hz}[0-10%]",
                "H_{15-80Hz}[39-70%]",
                "ER_{2-100Hz}[0-18/0-35%]",
            ],
            "color": "darkorchid",
        },
    ]

    results = {}

    # 1. Run Experiments & Calculate Fake Zones
    print("Step 1: Running Experiments & Calculating Fake Zones...")

    # Store P-values to display the most significant one or all
    p_values = []

    for exp in experiments:
        # A. Run Real Experiment
        res = run_experiment(
            exp["files"][0], exp["files"][1], exp["model"], exp["features"]
        )

        if res:
            # res structure: (stats_tuple, (fpr_raw, tpr_raw), y_true, y_probs, final_model, class_names)
            # Unpack
            stats, raw_roc, y_true, y_probs, final_model, class_names = res

            # Save Model
            model_filename = (
                f"{'Fig. 4d_' + exp['name'].replace(' ', '_')}_{exp['model']}.pkl"
            )
            full_model_path = os.path.join(get_script_dir(), OUTPUT_DIR, model_filename)
            joblib.dump(final_model, full_model_path)
            joblib.dump(final_model, full_model_path)
            print(f"    Saved model to: {model_filename}")

            if SAVE_CONFUSION_MATRIX:
                # Need to convert probabilities to predictions (using 0.5 threshold)
                y_pred_oof = (y_probs >= 0.5).astype(int)
                plot_confusion_matrix(
                    y_true,
                    y_pred_oof,
                    os.path.join(get_script_dir(), OUTPUT_DIR),
                    f"Confusion_Matrix_{exp['name']}",
                    class_names,
                )

                # New Analysis Plots

                # plot_threshold_tuning_curves(
                #     y_true,
                #     y_probs,
                #     os.path.join(get_script_dir(), OUTPUT_DIR),
                #     f"{exp['name']}",
                # )

            # B. Calculate Fake Zone for THIS experiment
            print(
                f"    Calculating Fake Zone for {exp['name']} ({FAKE_ROUNDS} rounds)..."
            )
            zone_fpr, zone_lower, zone_upper, p_val = calculate_fake_zone(
                y_true, y_probs, FAKE_ROUNDS
            )

            p_values.append(p_val)

            # Store everything
            results[exp["name"]] = {
                "real_stats": stats,
                "real_raw": raw_roc,
                "fake_stats": (zone_fpr, zone_lower, zone_upper),
                "p_val": p_val,
            }
        else:
            print(f"Skipping {exp['name']} due to errors.")
            return

    # 3. Plotting
    print("Step 3: Generating Plot...")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # A. Draw Mean Random Line (Base)
    ax.plot(
        [0, 1],
        [0, 1],
        color="#7F8C8D",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Mean Random AUC = 0.50 (1000 Shuffled Labels)",
    )

    # Loop through experiments to plot
    for exp in experiments:
        name = exp["name"]
        if name not in results:
            continue

        data = results[name]

        # Real Data
        (
            mean_fpr,
            mean_tpr,
            real_tpr_lower,
            real_tpr_upper,
            mean_auc,
            ci_lower,
            ci_upper,
        ) = data["real_stats"]
        (fpr_raw, tpr_raw) = data["real_raw"]

        # Fake Data
        (zone_fpr, zone_lower, zone_upper) = data["fake_stats"]

        color = exp["color"]

        # 1. Plot Fake Zone Filling
        ax.fill_between(
            zone_fpr,
            zone_lower,
            zone_upper,
            color="#95A5A6",
            alpha=0.15,  # Light filling
        )

        # 2. Plot Fake Zone Boundaries (Colored Dashed Lines)
        ax.plot(
            zone_fpr, zone_lower, color=color, linestyle="--", linewidth=1.0, alpha=0.6
        )
        ax.plot(
            zone_fpr, zone_upper, color=color, linestyle="--", linewidth=1.0, alpha=0.6
        )

        # 3. Plot Real ROC Curve (Raw/Stepped)
        label = (
            f"{name} Set AUC = {mean_auc:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})"
        )
        ax.plot(fpr_raw, tpr_raw, color=color, linewidth=2.5, label=label)

    # Create manual legend to avoid duplicate "Random Chance Zone" entries
    zone_patch = mpatches.Patch(
        color="#95A5A6", alpha=0.15, label="Random Chance Zone (95% CI)"
    )
    handles, labels = ax.get_legend_handles_labels()
    # Insert patch after "Mean Random AUC" (index 1)
    handles.insert(1, zone_patch)

    # Legend
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    # D. Styling and Annotations
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    # Axis Labels
    ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)

    # Title
    ax.set_title(
        "Internal Validation and Statistical Significance (RF)",
        fontsize=15,
        fontweight="bold",
    )

    # P-Value Annotation (Taking the max P-value or generally < 0.001 if all are good)
    max_p = max(p_values) if p_values else 1.0

    if max_p < 0.001:
        p_text = "P < 0.001 (All Models vs. Null Models, n = 1,000)"
    else:
        p_text = f"P <= {max_p:.3f} (vs. Null Models)"

    ax.text(
        0.98,
        0.05,
        p_text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    # Grid
    ax.grid(True, alpha=0.3)

    # Save (PNG Only, Current Directory)
    full_output_dir = os.path.join(get_script_dir(), OUTPUT_DIR)
    os.makedirs(full_output_dir, exist_ok=True)

    png_path = os.path.join(full_output_dir, f"{img_path}.png")
    # svg_path = os.path.join(full_output_dir, f"{img_path}.svg") # Disabled

    plt.tight_layout()
    plt.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    if SAVE_SVG:
        svg_path = os.path.join(full_output_dir, f"{img_path}.svg")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")

    print(f"\nSuccess! Plot saved to:\n  {png_path}")


if __name__ == "__main__":
    main()
