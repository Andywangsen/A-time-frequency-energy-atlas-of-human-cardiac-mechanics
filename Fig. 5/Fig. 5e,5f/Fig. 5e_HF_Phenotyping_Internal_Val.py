# -*- coding: utf-8 -*-
"""
==============================================================================
Figure 5-e: Combined ROC Analysis - Permutation Test

Generates publication-quality ROC curves demonstrating machine learning model classification performance.
Supports automatic loading of multiple Excel files, 5-fold cross-validation, and Bootstrap confidence intervals.

Key Features:
- Load HFpEF and HFrEF clinical data from Excel files
- Perform 5-fold cross-validation using Random Forest model
- Calculate ROC curves and 95% confidence intervals
- Generate permutation test validation (1000 permutations)
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

import joblib

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42
img_path = "Fig. 5e_Internal_ROC_Analysis"
OUTPUT_DIR = "."
FAKE_ROUNDS = 1000
N_FOLDS = 5
FIG_WIDTH = 10
FIG_HEIGHT = 10
OUTPUT_DPI = 300
OUTPUT_DPI = 300
SAVE_CONFUSION_MATRIX = False
SAVE_SVG = False

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


def get_script_dir():
    """Get absolute path of the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def get_model(model_name):
    """
    Model factory function, creates classifiers with consistent hyperparameters.

    Args:
        model_name: Model name ("RF" or "SVM")

    Returns:
        Configured classifier object
    """
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
    Calculate ROC curve confidence intervals using Bootstrap method.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bootstraps: Number of bootstrap resamples (default 1000)
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple (mean_fpr, mean_tpr, tpr_lower, tpr_upper, mean_auc, ci_lower, ci_upper)
    """
    np.random.seed(GLOBAL_SEED)
    n_samples = len(y_true)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_boot, y_proba_boot)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

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
    Load data, preprocess, run 5-fold cross-validation, and return combined results.

    Args:
        file1: First Excel filename
        file2: Second Excel filename
        model_name: Model name
        features: Feature list

    Returns:
        Tuple (stats, raw_roc, y_true, y_proba, final_model, class_names)
    """
    print(f"  Processing: {file1} vs {file2} | Model: {model_name}")

    base_dir = get_script_dir()
    path1 = os.path.join(base_dir, file1)
    path2 = os.path.join(base_dir, file2)

    try:
        df1 = pd.read_excel(path1)
        df2 = pd.read_excel(path2)
        df1["class"] = 0
        df2["class"] = 1

        cat1 = os.path.splitext(os.path.basename(file1))[0].split("_")[1]
        cat2 = os.path.splitext(os.path.basename(file2))[0].split("_")[1]
        df1["class"] = cat1
        df2["class"] = cat2

        df_all = pd.concat([df1, df2], ignore_index=True)
    except Exception as e:
        print(f"Error loading files: {e}")
        return None

    missing = [f for f in features if f not in df_all.columns]
    if missing:
        print(f"    Missing features: {missing}")
        return None

    X = df_all[features].apply(pd.to_numeric, errors="coerce").values
    le = LabelEncoder()
    y = le.fit_transform(df_all["class"])

    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"    Samples: {len(y)}, Class distribution: {np.bincount(y)}")

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

    print("    Training final model on full dataset...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    final_model = get_model(model_name)
    final_model.fit(X_scaled, y)

    stats = bootstrap_roc_ci(y, y_proba_oof)

    fpr_raw, tpr_raw, _ = roc_curve(y, y_proba_oof)

    return stats, (fpr_raw, tpr_raw), y, y_proba_oof, final_model, le.classes_


def calculate_fake_zone(y, y_proba_real, rounds=1000):
    """
    Calculate random chance zone by permuting labels 1000 times.

    Args:
        y: True labels
        y_proba_real: Real predicted probabilities
        rounds: Permutation rounds (default 1000)

    Returns:
        Tuple (mean_fpr, fake_tpr_lower, fake_tpr_upper, p_value, mean_fake_auc)
    """
    print(f"  Calculating Random Chance Zone ({rounds} rounds)...")
    np.random.seed(GLOBAL_SEED)

    mean_fpr = np.linspace(0, 1, 100)
    fake_tprs = []
    fake_aucs = []

    for i in range(rounds):
        y_fake = np.random.permutation(y)

        fake_aucs.append(roc_auc_score(y_fake, y_proba_real))

        fpr, tpr, _ = roc_curve(y_fake, y_proba_real)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        fake_tprs.append(tpr_interp)

    fake_tprs = np.array(fake_tprs)
    fake_tpr_lower = np.percentile(fake_tprs, 2.5, axis=0)
    fake_tpr_upper = np.percentile(fake_tprs, 97.5, axis=0)

    real_auc = roc_auc_score(y, y_proba_real)
    n_better = sum(1 for auc in fake_aucs if auc >= real_auc)
    p_value = (n_better + 1) / (rounds + 1)

    mean_fake_auc = np.mean(fake_aucs)

    return mean_fpr, fake_tpr_lower, fake_tpr_upper, p_value, mean_fake_auc


def calculate_metrics(y_true, y_pred):
    """
    Calculate detailed clinical metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing PPV, NPV, F1, Sensitivity, Specificity
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


def plot_threshold_tuning_curves(y_true, y_probs, output_dir, title):
    """
    Plot Sensitivity and Specificity vs Threshold curves.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        output_dir: Output directory
        title: Chart title

    Returns:
        Best threshold
    """
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
    """
    Plot confusion matrix and save metrics to Excel.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Output directory
        title: Chart title
        class_names: Class names
    """
    if not SAVE_CONFUSION_MATRIX:
        return

    metrics = calculate_metrics(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics["TP"] = tp
    metrics["TN"] = tn
    metrics["FP"] = fp
    metrics["FN"] = fn

    df_metrics = pd.DataFrame([metrics])

    cols = ["TP", "TN", "FP", "FN", "Sensitivity", "Specificity", "PPV", "NPV", "F1"]
    df_metrics = df_metrics[cols]

    excel_path = os.path.join(output_dir, f"{title}_Metrics.xlsx")
    df_metrics.to_excel(excel_path, index=False)

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
    """
    Main program entry.

    Execute complete ROC analysis pipeline:
    1. Run experiments and calculate permutation test
    2. Generate confusion matrix and threshold tuning curves
    3. Plot combined ROC curves
    4. Save result images
    """
    print("=" * 70)
    print("Figure 5-e Combined ROC Analysis - Permutation Test")
    print("=" * 70)

    global OUTPUT_DIR
    OUTPUT_DIR = "."

    print("\nStep 1 Running Experiments and Permutation Test...")
    experiments = [
        {
            "name": "HFpEF vs HFrEF",
            "files": ("Fig. 5e_HFpEF.xlsx", "Fig. 5e_HFrEF.xlsx"),
            "model": "RF",
            "features": [
                "E_{2-12Hz}[40-100%]",
                "H_{10-80Hz}[39-70%]",
                "E_{2-100Hz}[0-10%]",
            ],
            "color": "red",
        }
    ]

    results = {}

    p_values = []
    mean_fake_aucs = []

    for exp in experiments:
        res = run_experiment(
            exp["files"][0], exp["files"][1], exp["model"], exp["features"]
        )

        if res:
            stats, raw_roc, y_true, y_probs, final_model, class_names = res

            model_filename = (
                f"{'Fig. 5f_' + exp['name'].replace(' ', '_')}_{exp['model']}.pkl"
            )
            full_model_path = os.path.join(get_script_dir(), OUTPUT_DIR, model_filename)
            joblib.dump(final_model, full_model_path)
            joblib.dump(final_model, full_model_path)
            print(f"    [OK] Saved model to: {model_filename}")

            if SAVE_CONFUSION_MATRIX:
                y_pred_oof = (y_probs >= 0.5).astype(int)
                plot_confusion_matrix(
                    y_true,
                    y_pred_oof,
                    os.path.join(get_script_dir(), OUTPUT_DIR),
                    f"Confusion_Matrix_{exp['name']}",
                    class_names,
                )

                # plot_threshold_tuning_curves(
                #     y_true,
                #     y_probs,
                #     os.path.join(get_script_dir(), OUTPUT_DIR),
                #     f"{exp['name']}",
                # )

            zone_fpr, zone_lower, zone_upper, p_val, mean_fake_auc = (
                calculate_fake_zone(y_true, y_probs, FAKE_ROUNDS)
            )

            p_values.append(p_val)
            mean_fake_aucs.append(mean_fake_auc)

            results[exp["name"]] = {
                "real_stats": stats,
                "real_raw": raw_roc,
                "fake_stats": (zone_fpr, zone_lower, zone_upper),
                "p_val": p_val,
            }
        else:
            print(f"  [X] Skipping {exp['name']} due to errors.")
            return

    print("\nStep 2 Generating ROC Curves...")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    avg_mean_fake_auc = np.mean(mean_fake_aucs) if mean_fake_aucs else 0.50

    ax.plot(
        [0, 1],
        [0, 1],
        color="#7F8C8D",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label=f"Mean Random AUC = {avg_mean_fake_auc:.2f} ({FAKE_ROUNDS} Shuffled Labels)",
    )

    for exp in experiments:
        name = exp["name"]
        if name not in results:
            continue

        data = results[name]

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

        (zone_fpr, zone_lower, zone_upper) = data["fake_stats"]

        color = exp["color"]

        ax.fill_between(
            zone_fpr,
            zone_lower,
            zone_upper,
            color="#95A5A6",
            alpha=0.15,
            label="Random Chance Zone (95% CI)",
        )

        ax.plot(
            zone_fpr, zone_lower, color=color, linestyle="--", linewidth=1.0, alpha=0.6
        )
        ax.plot(
            zone_fpr, zone_upper, color=color, linestyle="--", linewidth=1.0, alpha=0.6
        )

        label = (
            f"{name} Set AUC = {mean_auc:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})"
        )
        ax.plot(fpr_raw, tpr_raw, color=color, linewidth=2.5, label=label)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)

    ax.set_title(
        "Internal Validation and Statistical Significance (RF)",
        fontsize=15,
        fontweight="bold",
    )

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

    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax.grid(True, alpha=0.3)

    full_output_dir = os.path.join(get_script_dir(), OUTPUT_DIR)
    os.makedirs(full_output_dir, exist_ok=True)

    png_path = os.path.join(full_output_dir, f"{img_path}.png")

    plt.tight_layout()
    plt.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    if SAVE_SVG:
        svg_path = os.path.join(full_output_dir, f"{img_path}.svg")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")

    print("\nDone  ROC Curves Saved:")
    print(f"  PNG: {png_path}")
    if SAVE_SVG:
        print(f"  SVG: {svg_path}")


if __name__ == "__main__":
    main()
