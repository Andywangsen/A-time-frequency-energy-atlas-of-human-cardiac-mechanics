# -*- coding: utf-8 -*-
"""
==============================================================================
Figure 3-b: ROC Curve Model Evaluation with Permutation Testing

Comprehensive machine learning model evaluation pipeline including:
- 5-fold cross-validation with out-of-fold predictions
- Bootstrap confidence intervals for AUC and ROC curves
- Permutation testing (1000 shuffled labels) for statistical significance
- Independent test set validation to prevent data leakage
- Single-feature model comparison
- Publication-quality ROC visualization
==============================================================================
"""

import pandas as pd
import numpy as np
import random
import os
import glob
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
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

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

SELECTED_FEATURES = [
    "H_{15-80Hz}[37-70%]",
    "H_{2-15Hz}[40-100%]",
    "ER_{2-100Hz}[0-20/0-30%]",
]

SINGLE_FEATURE = "ER_{2-100Hz}[0-20/0-30%]"
SELECTED_MODEL = "RF"
FAKE_ROUNDS = 1000
N_FOLDS = 5
HOLDOUT_RATIO = 0.2

FIG_WIDTH = 10
FIG_HEIGHT = 10
OUTPUT_DPI = 300
SAVE_CONFUSION_MATRIX = True
SAVE_SVG = False


def get_script_dir():
    """Get the absolute path of the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


OUTPUT_DIR = os.path.join(get_script_dir(), "")


def get_model(model_name):
    """Return model instance based on name"""
    models = {
        "RF": RandomForestClassifier(
            n_estimators=1000,
            max_depth=8,
            random_state=GLOBAL_SEED,
            class_weight="balanced",
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=GLOBAL_SEED
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=GLOBAL_SEED,
            class_weight="balanced",
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=200, random_state=GLOBAL_SEED
        ),
    }
    return models.get(model_name)


def bootstrap_auc_ci(
    y_true, y_proba, n_bootstraps=1000, confidence=0.95, random_state=42
):
    """
    Calculate AUC 95% confidence interval using Bootstrap method.
    Resamples predictions 1000 times to estimate CI.
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    bootstrapped_aucs = []

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        auc = roc_auc_score(y_true_boot, y_proba_boot)
        bootstrapped_aucs.append(auc)

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(bootstrapped_aucs, alpha * 100)
    ci_upper = np.percentile(bootstrapped_aucs, (1 - alpha) * 100)
    mean_auc = np.mean(bootstrapped_aucs)

    return mean_auc, ci_lower, ci_upper


def bootstrap_roc_ci(
    y_true, y_proba, n_bootstraps=1000, confidence=0.95, random_state=42
):
    """
    Calculate ROC curve confidence interval band using Bootstrap method.
    Returns: unified FPR grid, mean TPR, TPR lower bound, TPR upper bound
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_proba_boot)
        tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    alpha = (1 - confidence) / 2
    tpr_lower = np.percentile(tprs, alpha * 100, axis=0)
    tpr_upper = np.percentile(tprs, (1 - alpha) * 100, axis=0)

    return mean_fpr, mean_tpr, tpr_lower, tpr_upper


def cross_validate_and_predict(X, y, model_name, n_folds=5, verbose=True):
    """
    Perform stratified k-fold cross-validation and predict on full data.
    Returns: fold AUCs, out-of-fold probabilities, out-of-fold predictions
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    fold_aucs = []

    y_proba_oof = np.zeros(len(y))
    y_pred_oof = np.zeros(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = get_model(model_name)
        model.fit(X_train_scaled, y_train)

        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)

        y_proba_oof[test_idx] = y_proba
        y_pred_oof[test_idx] = y_pred

        auc = roc_auc_score(y_test, y_proba)
        fold_aucs.append(auc)

        if verbose:
            print(f"    Fold {fold + 1}: AUC = {auc:.4f}")

    return fold_aucs, y_proba_oof, y_pred_oof


def plot_roc_curve_with_ci(
    y_true,
    y_proba,
    mean_auc,
    ci_lower,
    ci_upper,
    title,
    output_name,
    output_dir,
    color="#3498DB",
    fake_rocs=None,
):
    """Plot ROC curve with confidence interval band"""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    if fake_rocs:
        gray_colors = ["#7F8C8D", "#95A5A6", "#BDC3C7", "#D5DBDB", "#E5E8E8"]
        for i, (fpr, tpr, auc, ci_l, ci_u) in enumerate(fake_rocs):
            color_fake = gray_colors[i % len(gray_colors)]
            ax.plot(
                fpr,
                tpr,
                color=color_fake,
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                label=f"Fake Round {i + 1} (AUC = {auc:.3f})",
            )

    mean_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc_ci(
        y_true, y_proba, n_bootstraps=1000
    )

    fpr_raw, tpr_raw, _ = roc_curve(y_true, y_proba)

    ax.plot(
        fpr_raw,
        tpr_raw,
        color=color,
        linewidth=2.5,
        label=f"Real Data AUC = {mean_auc:.3f}, 95% CI: {ci_lower:.3f}-{ci_upper:.3f}",
    )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random AUC = 0.500")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate 1 - Specificity", fontsize=12)
    ax.set_ylabel("True Positive Rate Sensitivity", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{output_name}.png"),
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return mean_fpr, mean_tpr, tpr_lower, tpr_upper


def calculate_empirical_pvalue(auc_real, fake_aucs):
    """Calculate empirical p-value"""
    n_permutations = len(fake_aucs)
    n_better = sum(1 for auc_fake in fake_aucs if auc_fake >= auc_real)
    p_value = (n_better + 1) / (n_permutations + 1)
    return p_value, n_better


def run_permutation_test(y_true, y_proba, n_rounds=1000, seed_start=42, verbose=False):
    """Run permutation test to get null distribution statistics"""
    mean_fpr_grid = np.linspace(0, 1, 100)
    fake_aucs = []
    fake_tprs = []

    for r in range(n_rounds):
        if verbose and (r + 1) % 100 == 0:
            print(f"  Running Round {r + 1}/{n_rounds}...")

        np.random.seed(seed_start + r)
        y_fake = np.random.permutation(y_true)

        auc_fake = roc_auc_score(y_fake, y_proba)
        fake_aucs.append(auc_fake)

        fpr_f, tpr_f, _ = roc_curve(y_fake, y_proba)
        tpr_interp = np.interp(mean_fpr_grid, fpr_f, tpr_f)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        fake_tprs.append(tpr_interp)

    fake_tprs = np.array(fake_tprs)
    fake_mean_tpr = np.mean(fake_tprs, axis=0)
    fake_tpr_lower = np.percentile(fake_tprs, 2.5, axis=0)
    fake_tpr_upper = np.percentile(fake_tprs, 97.5, axis=0)

    return fake_aucs, mean_fpr_grid, fake_mean_tpr, fake_tpr_lower, fake_tpr_upper


def calculate_metrics(y_true, y_pred):
    """Calculate detailed clinical metrics"""
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
    """Draw Sensitivity and Specificity vs Threshold curves"""
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
        os.path.join(output_dir, f"Extended Data Fig. 4a_Threshold_Tuning_{title}.png"),
        dpi=300,
    )
    if SAVE_SVG:
        plt.savefig(
            os.path.join(
                output_dir, f"Extended Data Fig. 4a_Threshold_Tuning_{title}.svg"
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


def plot_confusion_matrix(y_true, y_pred_probs, classes, output_dir, title):
    """Draw Confusion Matrix and Save Metrics to Excel"""
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
        xticklabels=classes,
        yticklabels=classes,
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


def plot_all_roc_combined(
    real_roc,
    fake_stats,
    output_dir,
    p_value=None,
    output_filename="Fig. 3b_ROC",
    single_feature_roc=None,
    fake_stats_single=None,
):
    """
    Plot Independent Test Set ROC + Random Chance Zone
    fake_stats: (mean_fpr, fake_mean_tpr, fake_tpr_lower, fake_tpr_upper)
    single_feature_roc: Optional, single feature model ROC data
    fake_stats_single: Optional, single feature model fake stats
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Unpack fake table statistics
    mean_fpr, fake_mean_tpr, fake_tpr_lower, fake_tpr_upper = fake_stats

    # 1. Draw "Random Chance Zone" (Gray shadow band)
    ax.fill_between(
        mean_fpr,
        fake_tpr_lower,
        fake_tpr_upper,
        color="#CCCCCC",  # Light Gray
        alpha=0.5,
        label="Random Chance Zone (95% CI)",
    )

    # Draw dashed boundary lines for Three-Feature Model (Red)
    ax.plot(
        mean_fpr, fake_tpr_lower, color="red", linestyle="--", linewidth=1, alpha=0.8
    )
    ax.plot(
        mean_fpr, fake_tpr_upper, color="red", linestyle="--", linewidth=1, alpha=0.8
    )

    # Draw dashed boundary lines for Single-Feature Model (Blue)
    if fake_stats_single:
        _, _, fake_tpr_lower_s, fake_tpr_upper_s = fake_stats_single
        ax.plot(
            mean_fpr,
            fake_tpr_lower_s,
            color="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )
        ax.plot(
            mean_fpr,
            fake_tpr_upper_s,
            color="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )
    # Add dashed boundary lines
    ax.plot(
        mean_fpr,
        fake_tpr_lower,
        color="#999999",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
    )
    ax.plot(
        mean_fpr,
        fake_tpr_upper,
        color="#999999",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
    )

    # Draw fake table mean line (dashed) - Add legend
    ax.plot(
        mean_fpr,
        fake_mean_tpr,
        color="#999999",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Mean Random AUC = 0.50 (1000 Shuffled Labels)",
    )

    # 2. Independent Test Set ROC (with CI band, no smoothing) - Draw on top layer
    fpr_real, tpr_real, tpr_lower_real, tpr_upper_real, mean_auc, ci_lower, ci_upper = (
        real_roc
    )

    # 1.5 Draw Single Feature ROC (if provided) - Green
    if single_feature_roc is not None:
        fpr_single, tpr_single, auc_single, ci_lower_single, ci_upper_single = (
            single_feature_roc
        )
        ax.plot(
            fpr_single,
            tpr_single,
            color="blue",
            linewidth=2.5,
            linestyle="-",
            label=f"(Constant) Independent Test Set AUC = {auc_single:.2f} (95% CI: {ci_lower_single:.2f} - {ci_upper_single:.2f})",
            alpha=1.0,
        )

    # Draw true table curve - Red
    ax.plot(
        fpr_real,
        tpr_real,
        color="red",
        linewidth=2.5,
        label=f"(3-Feature) Independent Test Set AUC = {mean_auc:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})",
    )

    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5)

    # Set axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)

    # Set title
    title = f"Permutation Test and Independent Set Validation (Ctrl-H vs HF, {SELECTED_MODEL})"
    ax.set_title(title, fontsize=14, fontweight="normal", pad=20)

    # Legend at top left
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    # Remove grid
    ax.grid(False)

    # Add annotation text at bottom center
    if p_value is not None:
        # Construct annotation text (two lines)
        annotation_lines = [
            f"P < {p_value:.3f} (vs. Null Models, n = {FAKE_ROUNDS:,} Permutations)",
        ]

        # Alignment: First line left aligned, second line centered
        alignments = ["center", "center"]
        x_positions = [0.5, 0.5]  # First line left side, second line centered

        # Add text at bottom position
        y_positions = [0.10, 0.05]  # Y positions for two lines of text
        for i, line in enumerate(annotation_lines):
            ax.text(
                x_positions[i],
                y_positions[i],
                line,
                transform=ax.transAxes,
                fontsize=11,
                horizontalalignment=alignments[i],
                verticalalignment="bottom",
                color="black",
            )

    plt.tight_layout()
    base_name = output_filename
    fig.savefig(
        os.path.join(output_dir, f"{base_name}.png"),
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    print("=" * 80)
    print(
        "Heart Energy Atlas - Real ROC + Fake ROC Validation + Confusion Matrix (5-Fold CV)"
    )
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {SELECTED_MODEL}")
    print(f"  Features: {SELECTED_FEATURES}")
    print(f"  Cross Validation: {N_FOLDS}-fold")
    print(f"  Fake Rounds: {FAKE_ROUNDS}")

    # ==================== 1. Load Data ====================
    print("\n[Step 1] Loading Data...")

    script_dir = get_script_dir()
    print(f"  Searching directory: {script_dir}")
    file_paths = glob.glob(os.path.join(script_dir, "*.xlsx"))
    # Filter out temp files or non-data files if necessary, for now assume all xlsx are data
    file_paths = [
        f
        for f in file_paths
        if not os.path.basename(f).startswith("~$")
        and (
            os.path.basename(f).startswith("Fig. 3b_Ctrl")
            or os.path.basename(f).startswith("Fig. 3b_HF")
        )
    ]

    if not file_paths:
        print("  [X] Error: Excel files (.xlsx) not found in current directory")

        return

    file_paths = sorted(list(file_paths))

    data_list = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        category = os.path.splitext(filename)[0].split("_")[1]
        try:
            df = pd.read_excel(file_path)
            df["class"] = category
            data_list.append(df)
            print(
                f"  [OK] Loaded: {filename} (Category: {category}, Samples: {len(df)})"
            )
        except Exception as e:
            print(f"  [X] Failed: {filename} {e}")

    if not data_list or len(data_list) < 2:
        print(
            "\n  [X] Error: Need at least two Excel files for binary classification (found: {})".format(
                len(data_list)
            )
        )
        # input("Press Enter to exit...")
        return

    # ==================== 2. Data Processing ====================
    print("\n[Step 2] Data Processing...")
    df_all = pd.concat(data_list, ignore_index=True)

    missing_features = [f for f in SELECTED_FEATURES if f not in df_all.columns]
    if missing_features:
        print(f"  [X] Following features not found: {missing_features}")
        return

    X = df_all[SELECTED_FEATURES].apply(pd.to_numeric, errors="coerce").values

    le = LabelEncoder()
    y = le.fit_transform(df_all["class"])
    class_names = le.classes_

    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"  Valid samples: {len(y)}")
    print(f"  Classes: {class_names}")

    # ==================== 2.5 Force Split Independent Test Set (or Full Data Mode) ====================
    if HOLDOUT_RATIO > 0:
        print(
            f"\n[Step 2.5] Forcing split {int(HOLDOUT_RATIO * 100)}% Independent Test Set (Prevent Data Leakage)..."
        )
        X_dev, X_final_test, y_dev, y_final_test = train_test_split(
            X,
            y,
            test_size=HOLDOUT_RATIO,
            stratify=y,  # Stratified sampling, keep class ratio
            random_state=GLOBAL_SEED,
        )
        print(f"  [OK] Development Set (for CV Training): {len(y_dev)} samples")
        print(f"  [OK] Independent Test Set (Locked): {len(y_final_test)} samples")

    else:
        print(f"\n[Step 2.5] Full Data Mode (HOLDOUT_RATIO = 0)...")
        X_dev, y_dev = X, y
        X_final_test, y_final_test = X, y
        print(f"  [OK] Development Set (Full): {len(y_dev)} samples")
        print(f"  [OK] Independent Test Set (Full Reuse): {len(y_final_test)} samples")
        print(f"  !  Note: ROC curve reflects training set fit (Training AUC)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==================== 3. Real Table 5-Fold CV (Dev Set Only) ====================

    fold_aucs, y_proba_oof, y_pred_oof = cross_validate_and_predict(
        X_dev, y_dev, SELECTED_MODEL, N_FOLDS
    )

    # Calculate Bootstrap AUC CI (1000 resamples)
    print(f"\n  Calculating Bootstrap AUC CI (1000 resamples)...")
    mean_auc, ci_lower, ci_upper = bootstrap_auc_ci(
        y_dev, y_proba_oof, n_bootstraps=1000
    )

    # Calculate Overall AUC using Full OOF Predictions
    overall_auc = roc_auc_score(y_dev, y_proba_oof)
    acc = accuracy_score(y_dev, y_pred_oof)

    print(f"\n  Real Table Results:")
    print(f"    Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"    Mean AUC: {mean_auc:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
    print(f"    Overall AUC (OOF): {overall_auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")

    # if SAVE_CONFUSION_MATRIX:
    #     plot_confusion_matrix(
    #         y_dev,
    #         y_proba_oof,
    #         class_names,
    #         OUTPUT_DIR,
    #         "Confusion_Matrix_CV_OOF",
    #     )

    #     # New Analysis Plots for CV OOF
    #     plot_threshold_tuning_curves(y_dev, y_proba_oof, OUTPUT_DIR, "CV_OOF")

    # Draw Real Table ROC
    # Real Table ROC drawing moved to after fake table generation

    # ==================== 4. Fake Table Validation (Always Use Full Data) ====================
    print(
        f"\n[Step 4] Fake Table Validation (Total Rounds: {FAKE_ROUNDS}, Monitoring Mode: Full Data Permutation)..."
    )
    fake_aucs = []
    # Store TPR interpolation results for all fake rounds (Unified FPR grid)
    mean_fpr_grid = np.linspace(0, 1, 100)
    fake_tprs = []

    if HOLDOUT_RATIO > 0:
        print(
            f"  [Prepare] Independent Test Set detected, integrating full data (Dev Set OOF + Independent Test Set Predict)..."
        )
        # 1. To get independent test set prediction probabilities (same status as Dev Set OOF), we need to use model trained on Dev Set to predict it
        # Logic here is similar to Step 6.5, calculate in advance for fake table validation consistency
        scaler_temp = StandardScaler()
        X_dev_scaled_temp = scaler_temp.fit_transform(X_dev)
        X_final_test_scaled_temp = scaler_temp.transform(X_final_test)

        model_temp = get_model(SELECTED_MODEL)
        model_temp.fit(X_dev_scaled_temp, y_dev)
        y_proba_test_temp = model_temp.predict_proba(X_final_test_scaled_temp)[:, 1]

        # 2. Merge Data
        # y_dev: Dev Set True Labels
        # y_final_test: Independent Test Set True Labels
        y_full_real = np.concatenate([y_dev, y_final_test])

        # y_proba_oof: OOF Prediction Probabilities on Dev Set
        # y_proba_test_temp: Prediction Probabilities on Independent Test Set
        y_proba_full_fixed = np.concatenate([y_proba_oof, y_proba_test_temp])

        print(
            f"  [OK] Full Data Integration Complete: Total Samples n={len(y_full_real)}"
        )
        print(
            f"  !  Note: Fake table validation will use this full data prediction vs shuffled full labels"
        )

    else:
        print(
            f"  [Prepare] Full Data Mode (HOLDOUT_RATIO=0), using full OOF results directly..."
        )
        y_full_real = y_dev  # In Step 2.5, if ratio=0, y_dev is full y
        y_proba_full_fixed = y_proba_oof
        print(
            f"  [OK] Full Data Preparation Complete: Total Samples n={len(y_full_real)}"
        )

    fake_aucs, mean_fpr_grid, fake_mean_tpr, fake_tpr_lower, fake_tpr_upper = (
        run_permutation_test(
            y_full_real,
            y_proba_full_fixed,
            n_rounds=FAKE_ROUNDS,
            seed_start=GLOBAL_SEED + 1000,
            verbose=True,
        )
    )

    print(f"\n  Fake Table Validation Complete. Mean AUC: {np.mean(fake_aucs):.4f}")

    # ==================== 5. Calculate Statistical Significance ====================
    print("\n[Step 5] Calculating Statistical Significance (Permutation Test)...")

    # No need to draw real table plot first, draw combined plot at the end

    p_value, n_better = calculate_empirical_pvalue(mean_auc, fake_aucs)

    print(f"  Permutation Test Results:")
    print(f"    Real Table AUC: {mean_auc:.4f}")
    print(f"    Empirical P-value: {p_value:.10f}")

    # ==================== 6. Draw Combined Plot ====================
    print("\n[Step 6] Drawing Combined ROC Curve...")
    # Evaluate Independent Test Set first, then draw combined plot

    # ==================== 6.5 Independent Test Set Evaluation (Key Evidence Against Data Leakage) ====================
    print(
        "\n[Step 6.5] Independent Test Set Evaluation (Data never participated in training)..."
    )

    # Train final model using all development set
    scaler_final = StandardScaler()
    X_dev_scaled = scaler_final.fit_transform(X_dev)
    X_final_test_scaled = scaler_final.transform(X_final_test)

    final_model = get_model(SELECTED_MODEL)
    final_model.fit(X_dev_scaled, y_dev)

    # Predict on independent test set
    y_proba_final_test = final_model.predict_proba(X_final_test_scaled)[:, 1]
    y_pred_final_test = final_model.predict(X_final_test_scaled)

    # Calculate independent test set metrics
    auc_final_test = roc_auc_score(y_final_test, y_proba_final_test)
    acc_final_test = accuracy_score(y_final_test, y_pred_final_test)
    auc_final_test_mean, auc_final_test_ci_lower, auc_final_test_ci_upper = (
        bootstrap_auc_ci(
            y_final_test, y_proba_final_test, n_bootstraps=1000, random_state=123
        )
    )

    # Independent Test Set ROC Curve (No smoothing) + CI band
    fpr_final_test, tpr_final_test, _ = roc_curve(y_final_test, y_proba_final_test)
    _, _, tpr_lower_final, tpr_upper_final = bootstrap_roc_ci(
        y_final_test, y_proba_final_test, n_bootstraps=1000, random_state=123
    )
    # Interpolate CI to original FPR grid
    mean_fpr_grid = np.linspace(0, 1, 100)
    tpr_lower_interp = np.interp(fpr_final_test, mean_fpr_grid, tpr_lower_final)
    tpr_upper_interp = np.interp(fpr_final_test, mean_fpr_grid, tpr_upper_final)

    print(
        f"  [OK] Independent Test Set AUC: {auc_final_test:.4f} (95% CI: {auc_final_test_ci_lower:.4f}-{auc_final_test_ci_upper:.4f})"
    )
    print(f"  [OK] Independent Test Set Accuracy: {acc_final_test:.4f}")
    print(f"  !  This result is the strongest evidence against data leakage!")

    if SAVE_CONFUSION_MATRIX:
        plot_confusion_matrix(
            y_final_test,
            y_proba_final_test,
            class_names,
            OUTPUT_DIR,
            "Extended Data Fig. 4a_Confusion_Matrix_Independent_Test",
        )

        # New Analysis Plots for Independent Test
        plot_threshold_tuning_curves(
            y_final_test, y_proba_final_test, OUTPUT_DIR, "Independent_Test"
        )

    # ==================== 6.7 Single Feature Model Training (For Comparison) ====================
    print("\n[Step 6.7] Single Feature Model Training (For Comparison)...")

    # Determine single feature
    single_feature_name = (
        SINGLE_FEATURE if SINGLE_FEATURE is not None else SELECTED_FEATURES[0]
    )
    print(f"  Selected Single Feature: {single_feature_name}")

    # Extract single feature data
    if single_feature_name not in df_all.columns:
        print(
            f"  [X] Warning: Feature {single_feature_name} not found, skipping single feature training"
        )
        single_feature_roc_pack_test = None
    else:
        X_single = (
            df_all[[single_feature_name]].apply(pd.to_numeric, errors="coerce").values
        )
        X_single = X_single[valid_mask]  # Use same valid sample mask

        # Split data according to HOLDOUT_RATIO - Use same splitting strategy as multi-feature
        if HOLDOUT_RATIO > 0:
            # Important: Use same random seed and stratification strategy to ensure exact same sample split
            X_single_dev, X_single_test, _, _ = train_test_split(
                X_single,
                y,  # Use full labels for stratification
                test_size=HOLDOUT_RATIO,
                stratify=y,
                random_state=GLOBAL_SEED,  # Use same random seed
            )
        else:
            X_single_dev = X_single
            X_single_test = X_single

        # Train Single Feature Model (Use same CV strategy)
        print(f"  Training Single Feature Model ({N_FOLDS}-fold CV)...")
        print(f"\n[Step 6.7] Single Feature Model Training  n={len(y_dev)})..")
        fold_aucs_single, y_proba_oof_single, y_pred_oof_single = (
            cross_validate_and_predict(
                X_single_dev, y_dev, SELECTED_MODEL, N_FOLDS, verbose=True
            )
        )

        # Calculate Single Feature Model AUC and CI (Training Set CV) - Use same random seed
        mean_auc_single, ci_lower_single, ci_upper_single = bootstrap_auc_ci(
            y_dev, y_proba_oof_single, n_bootstraps=1000, random_state=42
        )

        # Calculate Single Feature ROC Curve (Training Set CV)
        fpr_single_train, tpr_single_train, _ = roc_curve(y_dev, y_proba_oof_single)

        print(
            f"  [OK] Single Feature Model AUC (CV): {mean_auc_single:.4f} (95% CI: {ci_lower_single:.4f}-{ci_upper_single:.4f})"
        )

        # ==================== Single Feature Model Independent Test Set Evaluation ====================
        print(f"  Single Feature Model Independent Test Set Evaluation...")

        # Train final single feature model using all development set (Logic consistent with multi-feature step 6.5)
        scaler_single = StandardScaler()
        X_single_dev_scaled = scaler_single.fit_transform(X_single_dev)
        X_single_test_scaled = scaler_single.transform(X_single_test)

        final_model_single = get_model(SELECTED_MODEL)
        final_model_single.fit(X_single_dev_scaled, y_dev)

        # Predict on Independent Test Set
        y_proba_single_test = final_model_single.predict_proba(X_single_test_scaled)[
            :, 1
        ]

        # Calculate Independent Test Set Metrics - Use same random seed
        auc_single_test = roc_auc_score(y_final_test, y_proba_single_test)
        auc_single_test_mean, auc_single_test_ci_lower, auc_single_test_ci_upper = (
            bootstrap_auc_ci(
                y_final_test,
                y_proba_single_test,
                n_bootstraps=1000,
                random_state=123,
            )
        )

        # Calculate Single Feature Independent Test Set ROC Curve
        fpr_single_test, tpr_single_test, _ = roc_curve(
            y_final_test, y_proba_single_test
        )

        print(
            f"  [OK] Single Feature Model AUC (Independent Test): {auc_single_test:.4f} (95% CI: {auc_single_test_ci_lower:.4f}-{auc_single_test_ci_upper:.4f})"
        )

        # Prepare Single Feature Independent Test Set ROC Data Pack
        single_feature_roc_pack_test = (
            fpr_single_test,
            tpr_single_test,
            auc_single_test_mean,
            auc_single_test_ci_lower,
            auc_single_test_ci_upper,
        )

        # Single Feature Permutation Test
        print(f"  Running Single Feature Permutation Test...")
        # Prepare full probability
        if HOLDOUT_RATIO > 0:
            y_proba_single_full = np.concatenate(
                [y_proba_oof_single, y_proba_single_test]
            )
        else:
            y_proba_single_full = y_proba_oof_single

        (fake_aucs_single, _, _, fake_tpr_lower_single, fake_tpr_upper_single) = (
            run_permutation_test(
                y_full_real,
                y_proba_single_full,
                n_rounds=FAKE_ROUNDS,
                seed_start=GLOBAL_SEED + 1000,
                verbose=False,
            )
        )
        fake_stats_single = (None, None, fake_tpr_lower_single, fake_tpr_upper_single)

    # ==================== 6.7 Draw Independent Test Set ROC Plot ====================
    print("\n[Step 6.7] Drawing Independent Test Set ROC Plot...")

    # Draw combined plot (Using Independent Test Set ROC)
    real_roc = (
        fpr_final_test,
        tpr_final_test,
        tpr_lower_interp,
        tpr_upper_interp,
        auc_final_test_mean,
        auc_final_test_ci_lower,
        auc_final_test_ci_upper,
    )

    # Fake Table Statistics Pack
    fake_stats = (mean_fpr_grid, fake_mean_tpr, fake_tpr_lower, fake_tpr_upper)

    plot_all_roc_combined(
        real_roc,
        fake_stats,
        OUTPUT_DIR,
        p_value=p_value,
        output_filename="Fig. 3b_ROC_Independent_Test",
        single_feature_roc=single_feature_roc_pack_test,  # Use Independent Test Set Single Feature ROC
        fake_stats_single=fake_stats_single,
    )
    print(
        f"  [OK] Combined ROC Saved (Top Journal Style - Independent Test, With Single Feature Comparison)"
    )

    # ==================== 7. Calculate Statistical Summary ====================
    avg_fake_auc = np.mean(fake_aucs)

    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    print(f"  Model: {SELECTED_MODEL} ({N_FOLDS}-fold CV)")
    print(
        f"  Dev Set Samples: {len(y_dev)}, Independent Test Set Samples: {len(y_final_test)}"
    )

    print(
        f"  * AUC (Independent Test): {auc_final_test:.4f} (95% CI: {auc_final_test_ci_lower:.4f}-{auc_final_test_ci_upper:.4f})"
    )
    print(f"  Mean Fake Table AUC: {avg_fake_auc:.4f}")
    print(f"  Empirical P-value: {p_value:.4f}")
    print(f"\n  [Anti-Data Leakage Statement]")
    print(
        f"  Independent Test Set ({int(HOLDOUT_RATIO * 100)}%) never participated in: RFE feature selection, Cross Validation, Threshold Search"
    )
    print(
        f"  Independent Test Set AUC = {auc_final_test:.4f} is an unbiased estimate of model generalization ability"
    )

    print(f"\nRun Complete!")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
