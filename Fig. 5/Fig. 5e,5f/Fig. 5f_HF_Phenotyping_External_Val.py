# -*- coding: utf-8 -*-
"""
Figure 5-f: External Validation Analysis

Generates publication-quality ROC curves demonstrating pre-trained model performance on external validation data.
Supports loading pre-trained models, refitting scalers, and loading external validation datasets.

Key Features:
- Load pre-trained Random Forest model
- Load training data to refit the scaler
- Load external validation datasets
- Generate ROC curves for external validation
- Calculate confusion matrices and clinical metrics
- Plot threshold tuning curves
- Generate publication-quality PNG and SVG images
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import seaborn as sns

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_FILE = "Fig. 5f_External_ROC_Analysis.png"
OUTPUT_DPI = 300
OUTPUT_DPI = 300
SAVE_CONFUSION_MATRIX = True
SAVE_SVG = False
FIG_WIDTH = 10
FIG_HEIGHT = 10


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def load_and_scale(trn_files, val_file, features):
    base_dir = get_script_dir()

    dfs_train = []
    class_names = []
    for f in trn_files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            dfs_train.append(pd.read_excel(path))
            class_name = os.path.splitext(os.path.basename(f))[0].split("_")[1]
            class_names.append(class_name)
        else:
            print(f"Warning: Training file not found: {f}")
            return None, None, None, None

    class_names = sorted(class_names)

    df_train = pd.concat(dfs_train, ignore_index=True)

    if not all(feat in df_train.columns for feat in features):
        print(f"Missing features in training data: {features}")
        return None, None, None, None

    X_train = df_train[features].apply(pd.to_numeric, errors="coerce").fillna(0).values
    scaler = StandardScaler()
    scaler.fit(X_train)

    val_path = os.path.join(base_dir, val_file)
    if not os.path.exists(val_path):
        print(f"Validation file not found: {val_file}")
        return None, None, None, None

    df_val = pd.read_excel(val_path)

    if "label" not in df_val.columns:
        print(f"Column 'label' not found in {val_file}")
        return None, None, None, None

    label_map = {"HFpEF": 0, "HFrEF": 1}
    y_val = df_val["label"].map(lambda x: label_map.get(str(x).strip(), -1)).values

    valid = y_val != -1
    if not valid.all():
        print(
            f"Warning: Dropping {np.sum(~valid)} samples with unknown labels. Unique: {df_val['label'].unique()}"
        )
        y_val = y_val[valid]
        df_val = df_val.iloc[valid]

    y_val = y_val.astype(int)

    if not all(feat in df_val.columns for feat in features):
        print(f"Missing features in validation data {val_file}: {features}")
        return None, None, None, None

    X_val = df_val[features].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_val_scaled = scaler.transform(X_val)

    unique, counts = np.unique(y_val, return_counts=True)
    val_counts = dict(zip(unique, counts))

    return X_val_scaled, y_val, val_counts, class_names


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
    plt.savefig(
        os.path.join(output_dir, f"Extended Data Fig. 4e_Threshold_Tuning_{title}.png"),
        dpi=300,
    )
    if SAVE_SVG:
        plt.savefig(
            os.path.join(
                output_dir, f"Extended Data Fig. 4e_Threshold_Tuning_{title}.svg"
            ),
            format="svg",
        )
    plt.close()


def get_best_threshold(y_true, y_probs):
    """
    Find the optimal threshold using Youden's Index.
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
    return thresholds[best_idx]


def plot_confusion_matrix(y_true, y_pred_probs, output_dir, title, class_names):
    """
    Plot confusion matrix and save metrics to Excel.

    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities
        output_dir: Output directory
        title: Chart title
        class_names: Class names
    """
    if not SAVE_CONFUSION_MATRIX:
        return

    best_threshold = get_best_threshold(y_true, y_pred_probs)
    y_pred = (y_pred_probs >= best_threshold).astype(int)

    metrics = calculate_metrics(y_true, y_pred)

    metrics["Threshold"] = best_threshold

    df_metrics = pd.DataFrame([metrics])

    cols = ["Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "F1"]
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

    Execute complete external validation analysis pipeline:
    1. Load pre-trained models and validation data
    2. Calculate ROC and AUC
    3. Generate confusion matrix and metrics
    4. Plot combined ROC curves
    5. Save results
    """
    experiments = [
        {
            "name": "HFpEF vs HFrEF",
            "model_file": "Fig. 5f_HFpEF_vs_HFrEF_RF.pkl",
            "val_file": "Fig. 5f_V_HFpEF_HFrEF.xlsx",
            "trn_files": ("Fig. 5e_HFpEF.xlsx", "Fig. 5e_HFrEF.xlsx"),
            "features": [
                "E_{2-12Hz}[40-100%]",
                "H_{10-80Hz}[39-70%]",
                "E_{2-100Hz}[0-10%]",
            ],
            "color": "red",
        },
    ]

    results = {}

    print("=" * 70)
    print("Figure 5-f External Validation Analysis")
    print("=" * 70)

    for exp in experiments:
        print(f"\nProcessing {exp['name']}...")

        model_path = os.path.join(get_script_dir(), exp["model_file"])
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"  [X] Error loading model {exp['model_file']}: {e}")
            continue

        X_val, y_val, counts, class_names = load_and_scale(
            exp["trn_files"], exp["val_file"], exp["features"]
        )
        if X_val is None:
            continue

        y_probs = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_probs)
        fpr, tpr, _ = roc_curve(y_val, y_probs)

        results[exp["name"]] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "counts": counts,
            "color": exp["color"],
        }

        print(f"  [OK] AUC = {auc:.3f}")

        if SAVE_CONFUSION_MATRIX:
            plot_confusion_matrix(
                y_val,
                y_probs,
                get_script_dir(),
                f"Extended Data Fig. 4e_Confusion_Matrix_External_Validation_{exp['name']}",
                class_names,
            )

            plot_threshold_tuning_curves(
                y_val, y_probs, get_script_dir(), f"External_{exp['name']}"
            )

    print("\n Plotting Generating ROC Curves...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot Diagonal
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random Chance")

    # Plot Curves
    count_text_lines = []

    for exp in experiments:
        name = exp["name"]
        if name not in results:
            continue

        res = results[name]

        # Plot ROC
        label = f"{name} Set AUC = {res['auc']:.2f}"
        ax.plot(res["fpr"], res["tpr"], color=res["color"], linewidth=2, label=label)

        # Prepare Count Text
        # Assuming label 1 is CVD, 0 is HF based on filename context from training

        n_cvd = res["counts"].get(0, 0)  # Assuming 0 is CVD
        n_hf = res["counts"].get(1, 0)  # Assuming 1 is HF

        # Clean naming for text
        # "CVD-LRRCV vs HF-LRRCV" -> "CVD-LRRCV", "HF-LRRCV"
        parts = name.split(" vs ")
        # Just use parts[0] and parts[1]

        count_text_lines.append(f"{parts[0]}: n = {n_cvd}")
        count_text_lines.append(f"{parts[1]}: n = {n_hf}")
        count_text_lines.append("")  # Spacer

    ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
    ax.set_title("External Validation", fontsize=16, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="upper left", frameon=False, fontsize=12)

    # Add Count Text
    ax.text(
        0.65, 0.05, "\n".join(count_text_lines), transform=ax.transAxes, fontsize=12
    )

    out_path = os.path.join(get_script_dir(), OUTPUT_FILE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=OUTPUT_DPI)
    if SAVE_SVG:
        svg_path = os.path.join(
            get_script_dir(), "Figure 5-f_External_Validation_Analysis.svg"
        )
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
    print("\nDone ROC Curves Saved:")
    print(f"  PNG: {out_path}")
    if SAVE_SVG:
        print(f"  SVG: {svg_path}")


if __name__ == "__main__":
    main()
