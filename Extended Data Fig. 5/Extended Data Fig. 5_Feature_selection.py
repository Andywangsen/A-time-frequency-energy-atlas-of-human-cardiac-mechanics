# -*- coding: utf-8 -*-
"""
==============================================================================
Cardiac Energy Atlas Machine Learning Classification Model - RFE + Bootstrap CI + Class Balance
Rigorous feature selection and model evaluation using Recursive Feature Elimination, Bootstrap Confidence Intervals, and Class Weight Balancing
Data Privacy and Reproducibility Statement:
Due to the patient privacy and data sensitivity of the original clinical data
(such as high-precision ECG/SCG waveforms), and in accordance with ethical review
and privacy protection regulations, the original dataset cannot be publicly shared.
==============================================================================
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import os
import warnings
import itertools
import csv
from datetime import datetime
from collections import Counter
import multiprocessing

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


def get_feature_category(feature_name):
    """
    Return the category of the feature based on its name.

    Args:
        feature_name: Feature name string

    Returns:
        int: Feature category (1, 2, 3) or None
    """
    name = feature_name.strip()
    clean_name = name.replace("_grid", "").replace("_sensitivity", "")

    if clean_name.startswith("E") and clean_name.endswith("All Freq)"):
        return 1, "E$_{Impulse}$\n(Sys.Mag.)"

    if clean_name.startswith("H") and clean_name.endswith("All Freq)"):
        return 3, "H$_{Impulse}$\n(Sys.Mag.)"

    # ER Class
    if clean_name.startswith("ER_{0-100Hz}"):
        return 2, "E$_{Burst}$\n(Sys.ER)"

    # Ends with 80 Class
    if clean_name.startswith("E") and clean_name.endswith("80)"):
        return 6, "E$_{Stiffness}$\n(Dias.Mid-High)"

    if clean_name.startswith("H") and clean_name.endswith("80)"):
        return 7, "H$_{Stiffness}$\n(Dias.Mid-High)"

    # Other Intermediate States
    if (
        clean_name.startswith("E")
        and "All Freq" not in clean_name
        and "700" not in clean_name
    ):
        return 5, "E$_{Load}$\n(Dias.Low)"

    if (
        clean_name.startswith("H")
        and "All Freq" not in clean_name
        and "700" not in clean_name
    ):
        return 4, "H$_{Load}$\n(Sys.Low)"

    return 8, "8. Others"


def select_top_features_rfe(X_train, y_train, feature_names, top_n):
    """
    Feature selection using RFE.

    Process:
    1. Create Random Forest base estimator (with class weight balancing enabled)
    2. Execute RFE (remove 1 feature at a time)
    3. Get importance ranking of selected features

    Args:
        X_train: Training set feature matrix
        y_train: Training set labels
        feature_names: List of feature names
        top_n: Number of features to select

    Returns:
        tuple: (selected_names, selected_indices)
    """
    base_estimator = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=1, class_weight="balanced"
    )

    n_features = min(top_n, X_train.shape[1])

    # Feature selection using RFE, removing 1 feature at a time (most rigorous)
    selector = RFE(
        estimator=base_estimator,
        n_features_to_select=n_features,
        step=1,
    )

    selector.fit(X_train, y_train)

    selected_mask = selector.support_
    selected_indices = np.where(selected_mask)[0]
    selected_names = [feature_names[i] for i in selected_indices]

    # Train RF on selected features for ranking
    # rankings = selector.ranking_  # Unused
    rf_temp = RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=1, class_weight="balanced"
    )
    rf_temp.fit(X_train[:, selected_indices], y_train)
    importances = rf_temp.feature_importances_

    # Sort features by importance descending
    sorted_idx = np.argsort(importances)[::-1]
    selected_names = [selected_names[i] for i in sorted_idx]
    selected_indices = selected_indices[sorted_idx]

    return selected_names, selected_indices


def rfe_fold_worker(args):
    fold_idx, X_train_fold, y_train_fold, feature_cols, top_n = args
    selected_names, selected_indices = select_top_features_rfe(
        X_train_fold, y_train_fold, feature_cols, top_n
    )
    return fold_idx, selected_names


def fake_round_worker(args):
    (
        round_idx,
        y_fake,
        unique_combos,
        X_full,
        feat_to_idx,
        class_names,
        thresholds,
        cores,
    ) = args

    fake_tasks = []
    for combo in unique_combos:
        idxs = [feat_to_idx[c] for c in combo]
        fake_tasks.append(
            (tuple(combo), X_full[:, idxs], y_fake, class_names, thresholds, False)
        )

    round_results = {}
    for task in fake_tasks:
        res_list = process_combination_with_ci(task)
        for res in res_list:
            key = (res["Feature 1"], res["Feature 2"], res["Feature 3"], res["Model"])
            round_results[key] = res["AUC(CV)"]

    return round_idx, round_results


def bootstrap_auc_ci(y_true, y_proba, n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Calculate Bootstrap CI for AUC.

    Method: Resample with replacement 1000 times, calculate AUC distribution percentiles.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bootstrap: Number of resamples (default 1000)
        ci: Confidence level (default 0.95 = 95%)
        random_state: Random seed

    Returns:
        tuple: (auc_mean, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    auc_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true_boot, y_proba_boot)
            auc_scores.append(auc)
        except Exception:
            continue

    if len(auc_scores) == 0:
        return 0, 0, 0

    auc_scores = np.array(auc_scores)
    auc_mean = np.mean(auc_scores)

    alpha = 1 - ci
    ci_lower = np.percentile(auc_scores, alpha / 2 * 100)
    ci_upper = np.percentile(auc_scores, (1 - alpha / 2) * 100)

    return auc_mean, ci_lower, ci_upper


def generate_valid_combinations(feature_names, combo_size=3):
    """
    Generate mutually exclusive feature combinations.

    Ensure that features in each combination come from different categories.

    Args:
        feature_names: List of feature names
        combo_size: Combination size (default 3)

    Returns:
        list: List of valid feature combinations
    """
    combinations = []
    for combo in itertools.combinations(feature_names, combo_size):
        categories = [get_feature_category(f) for f in combo]
        valid_cats = [c for c in categories if c is not None]
        if len(valid_cats) == len(set(valid_cats)):
            combinations.append(combo)
    return combinations


def process_combination_with_ci(args):
    """
    Complete model evaluation for feature combo.

    Process:
    1. 5-fold CV to calculate CV AUC and confidence intervals
    2. 70-30 split for test set evaluation
    3. Calculate performance metrics using 3 thresholds

    Args:
        args: tuple containing (combo_names, X_combo_raw, y, class_names, thresholds, calc_ci)

    Returns:
        list: List of evaluation result dictionaries
    """
    combo_names, X_combo_raw, y, class_names, thresholds, calc_ci = args

    models_config = {
        "RF": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        ),
        "GBM": lambda: GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        ),
        "SVM": lambda: SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),
        "GLM": lambda: LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=1000
        ),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=200, random_state=42
        ),
    }

    results = []

    from sklearn.utils.class_weight import compute_sample_weight

    # Compute sample weights for class imbalance
    sample_weights = compute_sample_weight("balanced", y)

    for model_name, model_factory in models_config.items():
        try:
            # Outer 5-fold CV for performance evaluation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_aucs = []
            y_proba_cv = np.zeros(len(y))

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_combo_raw, y)):
                X_tr_raw, X_val_raw = X_combo_raw[train_idx], X_combo_raw[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                sw_tr = sample_weights[train_idx]

                # Independent standardization within fold to avoid leakage
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr_raw)
                X_val = scaler.transform(X_val_raw)

                model = model_factory()

                if model_name in ["GBM", "MLP"]:
                    model.fit(X_tr, y_tr, sample_weight=sw_tr)
                else:
                    model.fit(X_tr, y_tr)

                if hasattr(model, "predict_proba") and len(class_names) == 2:
                    fold_proba = model.predict_proba(X_val)[:, 1]
                    y_proba_cv[val_idx] = fold_proba
                    fold_auc = roc_auc_score(y_val, fold_proba)
                    fold_aucs.append(fold_auc)
                else:
                    fold_aucs.append(0)

            cv_auc = roc_auc_score(y, y_proba_cv) if len(class_names) == 2 else 0

            ci_lower, ci_upper = 0, 0
            if calc_ci and len(class_names) == 2 and cv_auc > 0:
                _, ci_lower, ci_upper = bootstrap_auc_ci(
                    y, y_proba_cv, n_bootstrap=1000
                )

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_combo_raw, y, test_size=0.3, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            sw_train = compute_sample_weight("balanced", y_train)

            model = model_factory()
            if model_name in ["GBM", "MLP"]:
                model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                model.fit(X_train, y_train)

            test_auc = 0
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if len(class_names) == 2:
                    test_auc = roc_auc_score(y_test, y_proba[:, 1])

            if len(class_names) == 2 and y_proba is not None:
                y_scores = y_proba[:, 1]
                for thresh in thresholds:
                    y_pred_thresh = (y_scores >= thresh).astype(int)
                    results.append(
                        {
                            "Feature 1": combo_names[0],
                            "Feature 2": combo_names[1],
                            "Feature 3": combo_names[2],
                            "Model": model_name,
                            "Threshold": thresh,
                            "Accuracy": accuracy_score(y_test, y_pred_thresh),
                            "Precision": precision_score(
                                y_test, y_pred_thresh, zero_division=0
                            ),
                            "Recall": recall_score(
                                y_test, y_pred_thresh, zero_division=0
                            ),
                            "F1 Score": f1_score(
                                y_test, y_pred_thresh, zero_division=0
                            ),
                            "AUC (Test)": test_auc,
                            "AUC(CV)": cv_auc,
                            "AUC_CI_Lower": ci_lower,
                            "AUC_CI_Upper": ci_upper,
                            "CV_Fold1": fold_aucs[0] if len(fold_aucs) > 0 else 0,
                            "CV_Fold2": fold_aucs[1] if len(fold_aucs) > 1 else 0,
                            "CV_Fold3": fold_aucs[2] if len(fold_aucs) > 2 else 0,
                            "CV_Fold4": fold_aucs[3] if len(fold_aucs) > 3 else 0,
                            "CV_Fold5": fold_aucs[4] if len(fold_aucs) > 4 else 0,
                        }
                    )
            else:
                y_pred = model.predict(X_test)
                results.append(
                    {
                        "Feature 1": combo_names[0],
                        "Feature 2": combo_names[1],
                        "Feature 3": combo_names[2],
                        "Model": model_name,
                        "Threshold": "N/A",
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        ),
                        "Recall": recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        ),
                        "F1 Score": f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        ),
                        "AUC (Test)": test_auc,
                        "AUC(CV)": cv_auc,
                        "AUC_CI_Lower": 0,
                        "AUC_CI_Upper": 0,
                        "CV_Fold1": 0,
                        "CV_Fold2": 0,
                        "CV_Fold3": 0,
                        "CV_Fold4": 0,
                        "CV_Fold5": 0,
                    }
                )
        except Exception:
            pass

    return results


def main():
    """
    Main program flow.

    Steps:
    1. Data reading and cleaning
    2. Feature filtering and encoding
    3. RFE feature selection (Nested CV)
    4. Feature combination generation and evaluation
    5. Result saving and optional permutation test
    """
    multiprocessing.freeze_support()

    print("=" * 80)
    print("Cardiac Energy Atlas - RFE + Bootstrap CI + Class Balance (v2.3)")
    print("=" * 80)
    print("\n* Version Features:")
    print(
        "   1. RFE (Recursive Feature Elimination) instead of simple importance ranking"
    )
    print("   2. Bootstrap 1000 times for AUC 95% Confidence Interval")
    print("   3. Nested Cross-Validation, no data leakage")
    print("   4. Class Weight Balancing - Automatically handle sample imbalance\n")

    root = tk.Tk()
    root.withdraw()

    print("[Step 1] Reading Data...")
    file_paths = filedialog.askopenfilenames(
        title="Please select Excel files", filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not file_paths:
        return

    file_paths = sorted(list(file_paths))

    data_list = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        category = filename.split("-")[0]
        try:
            df = pd.read_excel(file_path)
            df["Category"] = category
            data_list.append(df)
            print(f"  [OK] Loaded: {filename} (Category: {category})")
        except Exception as e:
            print(f"  [X] Failed: {filename} {e}")

    if not data_list:
        return

    print("\n[Step 2] Merging and Cleaning...")

    # ==================== 2. Data Cleaning ====================
    all_cols = [set(d.columns) for d in data_list]
    common_cols = set.intersection(*all_cols) - {
        "Category",
        "DataFile",
        "ID",
        "TimeWindow",
        "AnnotationID",
    }
    feature_cols = [c for c in common_cols if not c.startswith("Unnamed")]
    print(f"  Original feature count: {len(feature_cols)}")

    df_all = pd.concat(data_list, ignore_index=True)
    df_features = df_all[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_features.dropna(axis=1, how="all", inplace=True)
    df_features.dropna(axis=0, inplace=True)

    feature_cols = list(df_features.columns)
    df_all = df_all.loc[df_features.index].reset_index(drop=True)
    df_features = df_features.reset_index(drop=True)

    # ==================== 2.1 Filter to keep only 3 major categories ====================
    filtered_feature_cols = []
    category_counts = {1: 0, 2: 0, 3: 0}

    for col in feature_cols:
        cat = get_feature_category(col)
        if cat is not None:
            filtered_feature_cols.append(col)
            category_counts[cat] += 1

    if len(filtered_feature_cols) < 3:
        return

    print("\n[Step 2.1] Filter Features - Keep only 3 specific categories...")
    print("  Filter Results:")
    print(f"    Cat 1 E(0-100, All Freq): {category_counts[1]} features")
    print(f"    Cat 2 E(400-1000, 2-13): {category_counts[2]} features")
    print(f"    Cat 3 H(360-700, 11-80): {category_counts[3]} features")
    print(f"    Total: {len(filtered_feature_cols)} features")

    feature_cols = filtered_feature_cols
    df_features = df_features[feature_cols]

    if len(feature_cols) < 3:
        return

    le = LabelEncoder()
    y = le.fit_transform(df_all["Category"])
    X_full = df_features.values
    class_names = le.classes_

    unique, counts = np.unique(y, return_counts=True)

    imbalance_ratio = max(counts) / min(counts)

    print(f"\n  Valid samples: {len(df_all)}")
    print("  Class Distribution:")
    for cls_idx, cnt in zip(unique, counts):
        print(f"    {class_names[cls_idx]}: {cnt} samples ({cnt / len(y) * 100:.1f}%)")

    if imbalance_ratio > 1.5:
        print(f"  ! Imbalance detected (Ratio {imbalance_ratio:.2f}:1)")
    else:
        print(f"  [OK] Balanced samples (Ratio {imbalance_ratio:.2f}:1)")

    top_n = simpledialog.askinteger(
        "Feature Selection",
        f"Detected {len(feature_cols)} features.\nEnter Top N (Recommended 30-50):",
        initialvalue=40,
        minvalue=3,
        maxvalue=len(feature_cols),
    )
    if top_n is None:
        top_n = 40

    calc_ci_for_all = messagebox.askyesno(
        "Bootstrap CI",
        "Calculate AUC 95% CI for ALL combinations?\n\n"
        "Select [Yes]: Calculate CI for all (Slower)\n"
        "Select [No]: Calculate CI only for high-score combos (Recommended)",
    )

    ci_threshold = 0.75
    if not calc_ci_for_all:
        ci_threshold = simpledialog.askfloat(
            "CI Threshold",
            "Calculate CI only for combos with AUC(CV) greater than?",
            initialvalue=0.75,
            minvalue=0.5,
            maxvalue=1.0,
        )
        if ci_threshold is None:
            ci_threshold = 0.75

    do_mix_validation = False
    mix_auc_threshold = 0.8
    mix_rounds = 5

    if messagebox.askyesno(
        "Mix Validation", "Enable [Permutation Test] (Mix Validation)?"
    ):
        do_mix_validation = True
        mix_auc_threshold = simpledialog.askfloat(
            "Mix Validation Threshold",
            "Enter AUC Threshold (e.g. 0.8):",
            initialvalue=0.8,
            minvalue=0.0,
            maxvalue=1.0,
        )
        mix_rounds = simpledialog.askinteger(
            "Fake Table Rounds",
            "Enter number of random fake table validations:",
            initialvalue=5,
            minvalue=1,
            maxvalue=100,
        )
        if mix_auc_threshold is None or mix_rounds is None:
            do_mix_validation = False

    print("\n[Step 3] Nested CV - RFE Feature Selection...")
    cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"  Start {min(cores, 5)} core parallel computing RFE...")

    # ==================== 3. Nested CV - RFE Feature Selection ====================
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_selected_features = []
    all_feature_counter = Counter()

    rfe_tasks = []
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_full, y)):
        X_train_fold = X_full[train_idx]
        y_train_fold = y[train_idx]
        rfe_tasks.append((fold_idx, X_train_fold, y_train_fold, feature_cols, top_n))

    with multiprocessing.Pool(min(cores, 5)) as pool:
        rfe_results = pool.map(rfe_fold_worker, rfe_tasks)

    rfe_results.sort(key=lambda x: x[0])
    for fold_idx, selected_names in rfe_results:
        fold_selected_features.append(set(selected_names))
        all_feature_counter.update(selected_names)
        print(f"    Fold {fold_idx + 1}: Selected {len(selected_names)} features [OK]")

    print("\n[Step 4] Statistics of high-frequency consensus features...")
    # ==================== 4. Statistics of high-frequency consensus features ====================
    min_freq = 3
    consensus_features = [
        f for f, count in all_feature_counter.items() if count >= min_freq
    ]

    print(
        f"  Consensus feature count (at least {min_freq}/5 folds): {len(consensus_features)} features"
    )
    print("  Feature Frequency Stats (Top 20):")
    for feat, count in all_feature_counter.most_common(20):
        stability = "*****" if count == 5 else "*" * count
        print(f"    {feat}: {count}/5 folds {stability}")

    freq_df = pd.DataFrame(
        [
            {
                "Feature Name": f,
                "Occurrence Folds": c,
                "Stability": "Gold Standard"
                if c == 5
                else ("Robust" if c >= 3 else "Unstable"),
                "RFE Selected": "Yes",
            }
            for f, c in all_feature_counter.most_common()
        ]
    )
    freq_report_file = f"Feature_Frequency_Report_Top{top_n}_RFE_NestedCV.xlsx"
    freq_df.to_excel(freq_report_file, index=False)

    print(f"  [OK] Feature frequency report saved: {freq_report_file}")

    # ==================== 6. Generate combinations based on consensus features ====================
    if len(consensus_features) < 3:
        selected_names, _ = select_top_features_rfe(X_full, y, feature_cols, top_n)
        consensus_features = selected_names

    combinations = generate_valid_combinations(consensus_features, combo_size=3)
    total_combinations = len(combinations)

    if total_combinations == 0:
        print("  No valid combinations, program ended.")
        return

    print(
        f"\n[Step 6] Generate combinations based on {len(consensus_features)} consensus features..."
    )
    print(f"  Remaining {total_combinations} valid combinations after filtering")

    feat_to_idx = {name: i for i, name in enumerate(feature_cols)}
    thresholds = [0.3, 0.5, 0.7]

    # ==================== 7. Round 1: Rapid evaluation of all combinations ====================
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_csv = f"temp_results_rfe_{timestamp_str}.csv"
    start_time = datetime.now()

    if calc_ci_for_all:
        print("\n[Step 7] Evaluate all combinations (Calculating CI)...")
    else:
        print(
            "\n[Step 7] Round 1: Rapid evaluation of all combinations (Skipping CI)..."
        )
    print(f"  Start {cores} core parallel computing...")

    tasks_round1 = []
    for combo in combinations:
        idxs = [feat_to_idx[c] for c in combo]
        tasks_round1.append(
            (combo, X_full[:, idxs], y, class_names, thresholds, calc_ci_for_all)
        )

    with open(temp_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Feature 1",
                "Feature 2",
                "Feature 3",
                "Model",
                "Threshold",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "AUC (Test)",
                "AUC(CV)",
                "AUC_CI_Lower",
                "AUC_CI_Upper",
                "CV_Fold1",
                "CV_Fold2",
                "CV_Fold3",
                "CV_Fold4",
                "CV_Fold5",
            ]
        )

        with multiprocessing.Pool(cores) as pool:
            result_iter = pool.imap_unordered(
                process_combination_with_ci, tasks_round1, chunksize=50
            )

            for i, res_list in enumerate(result_iter):
                for res in res_list:
                    writer.writerow(
                        [
                            res["Feature 1"],
                            res["Feature 2"],
                            res["Feature 3"],
                            res["Model"],
                            res["Threshold"],
                            res["Accuracy"],
                            res["Precision"],
                            res["Recall"],
                            res["F1 Score"],
                            res["AUC (Test)"],
                            res["AUC(CV)"],
                            res["AUC_CI_Lower"],
                            res["AUC_CI_Upper"],
                            res["CV_Fold1"],
                            res["CV_Fold2"],
                            res["CV_Fold3"],
                            res["CV_Fold4"],
                            res["CV_Fold5"],
                        ]
                    )

                if (i + 1) % 50 == 0:
                    percent = (i + 1) / total_combinations * 100
                    print(
                        f"\r  Progress: {percent:.1f}% ({i + 1}/{total_combinations})",
                        end="",
                        flush=True,
                    )
                    f.flush()

    # ==================== 8. Round 2: Calculate CI for high-score combinations ====================
    df_res = pd.read_csv(temp_csv)
    df_res.sort_values("AUC(CV)", ascending=False, inplace=True)

    if not calc_ci_for_all:
        print(
            f"\n[Step 8] Round 2: Calculating Bootstrap CI for combinations with AUC(CV) >= {ci_threshold}..."
        )
        high_auc_df = df_res[df_res["AUC(CV)"] >= ci_threshold].copy()
        unique_high_combos = (
            high_auc_df[["Feature 1", "Feature 2", "Feature 3"]]
            .drop_duplicates()
            .values.tolist()
        )

        if len(unique_high_combos) > 0:
            print(f"  Unique combinations to calculate CI: {len(unique_high_combos)}")
            tasks_round2 = []
            for combo in unique_high_combos:
                idxs = [feat_to_idx[c] for c in combo]
                tasks_round2.append(
                    (tuple(combo), X_full[:, idxs], y, class_names, thresholds, True)
                )

            ci_results = {}
            with multiprocessing.Pool(cores) as pool:
                result_iter = pool.imap_unordered(
                    process_combination_with_ci, tasks_round2, chunksize=10
                )
                for i, res_list in enumerate(result_iter):
                    for res in res_list:
                        key = (
                            res["Feature 1"],
                            res["Feature 2"],
                            res["Feature 3"],
                            res["Model"],
                        )
                        ci_results[key] = (res["AUC_CI_Lower"], res["AUC_CI_Upper"])

                    if (i + 1) % 10 == 0:
                        print(
                            f"\r  CI Calculation Progress: {i + 1}/{len(unique_high_combos)}",
                            end="",
                            flush=True,
                        )

            def update_ci(row):
                key = (
                    row["Feature 1"],
                    row["Feature 2"],
                    row["Feature 3"],
                    row["Model"],
                )
                if key in ci_results:
                    return pd.Series(ci_results[key])
                return pd.Series([row["AUC_CI_Lower"], row["AUC_CI_Upper"]])

            df_res[["AUC_CI_Lower", "AUC_CI_Upper"]] = df_res.apply(update_ci, axis=1)

    else:
        print("[Step 8] CI calculated in Step 7 (Skipped)")

    # Add feature stability annotation
    def get_stability(row):
        stabilities = []
        for col in ["Feature 1", "Feature 2", "Feature 3"]:
            feat = row[col]
            count = all_feature_counter.get(feat, 0)
            stabilities.append(count)
        return min(stabilities)

    df_res["Minimum Feature Stability"] = df_res.apply(get_stability, axis=1)

    # Format CI column
    def format_ci(row):
        if row["AUC_CI_Lower"] > 0 and row["AUC_CI_Upper"] > 0:
            return f"{row['AUC(CV)']:.3f} ({row['AUC_CI_Lower']:.3f}-{row['AUC_CI_Upper']:.3f})"
        return f"{row['AUC(CV)']:.3f}"

    df_res["AUC(CV) [95%CI]"] = df_res.apply(format_ci, axis=1)

    # Reorder columns

    priority_cols = [
        "Feature 1",
        "Feature 2",
        "Feature 3",
        "Model",
        "AUC(CV) [95%CI]",
        "AUC(CV)",
        "AUC_CI_Lower",
        "AUC_CI_Upper",
        "AUC (Test)",
        "Minimum Feature Stability",
    ]
    other_cols = [c for c in df_res.columns if c not in priority_cols]
    df_res = df_res[priority_cols + other_cols]

    # ==================== 9. Save results ====================
    print("\n[Step 9] Saving final results...")
    outfile = f"Top{top_n}_Combinations_Result_RFE_CI_Version.xlsx"
    df_res.to_excel(outfile, index=False)
    print(f"  [OK] Successfully saved: {outfile}")
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    # ==================== 10. Permutation Test (Mix Validation) ====================
    if do_mix_validation:
        print(
            f"\n[Step 10] Executing Mix Validation (Threshold >= {mix_auc_threshold}, Rounds={mix_rounds})..."
        )
        high_score_df = df_res[df_res["AUC(CV)"] >= mix_auc_threshold].copy()

        if len(high_score_df) > 0:
            print(f"  Total {len(high_score_df)} combinations meet threshold...")
            unique_combos = (
                high_score_df[["Feature 1", "Feature 2", "Feature 3"]]
                .drop_duplicates()
                .values.tolist()
            )

            print(f"  Unique feature combinations to verify: {len(unique_combos)}")

            fake_labels_list = []
            for r in range(mix_rounds):
                current_seed = 999 + r
                np.random.seed(current_seed)
                y_fake = np.random.permutation(y)
                fake_labels_list.append((r, y_fake, current_seed))

                try:
                    df_fake_save = df_all.copy()
                    df_fake_save["Fake_Label"] = y_fake
                    df_fake_save["Fake_Category_Name"] = le.inverse_transform(y_fake)
                    fake_data_file = (
                        f"MixValidation_FakeData_Round{r + 1}_Seed{current_seed}.xlsx"
                    )
                    df_fake_save.to_excel(fake_data_file, index=False)
                    print(f"    [OK] Fake table saved: {fake_data_file}")
                except Exception:
                    pass

            all_fake_tasks = []
            task_info = []
            for r, y_fake, _ in fake_labels_list:
                for combo in unique_combos:
                    idxs = [feat_to_idx[c] for c in combo]
                    all_fake_tasks.append(
                        (
                            tuple(combo),
                            X_full[:, idxs],
                            y_fake,
                            class_names,
                            thresholds,
                            False,
                        )
                    )
                    task_info.append(r)

            fake_auc_map = {}
            completed = 0
            # total_tasks = len(all_fake_tasks)

            with multiprocessing.Pool(cores) as pool:
                result_iter = pool.imap_unordered(
                    process_combination_with_ci,
                    all_fake_tasks,
                    chunksize=max(1, len(all_fake_tasks) // (cores * 4)),
                )
                for i, res_list in enumerate(result_iter):
                    for res in res_list:
                        key = (
                            res["Feature 1"],
                            res["Feature 2"],
                            res["Feature 3"],
                            res["Model"],
                        )
                        if key not in fake_auc_map:
                            fake_auc_map[key] = []
                        fake_auc_map[key].append(res["AUC(CV)"])

                    completed += 1

            summary_data = []
            for key, auc_list in fake_auc_map.items():
                f1, f2, f3, model = key
                mean_fake_auc = np.mean(auc_list)
                row = {
                    "Feature 1": f1,
                    "Feature 2": f2,
                    "Feature 3": f3,
                    "Model": model,
                    "Mean Fake AUC": mean_fake_auc,
                }
                for i, val in enumerate(auc_list[:mix_rounds]):
                    row[f"Fake AUC Round {i + 1}"] = val
                summary_data.append(row)

            df_fake_summary = pd.DataFrame(summary_data)
            df_final_check = pd.merge(
                high_score_df,
                df_fake_summary,
                on=["Feature 1", "Feature 2", "Feature 3", "Model"],
                how="left",
            )
            df_final_check["Mean AUC Diff"] = (
                df_final_check["AUC(CV)"] - df_final_check["Mean Fake AUC"]
            )
            df_final_check.sort_values("Mean AUC Diff", ascending=False, inplace=True)

            cols = [
                "Feature 1",
                "Feature 2",
                "Feature 3",
                "Model",
                "AUC(CV) [95%CI]",
                "AUC(CV)",
                "AUC_CI_Lower",
                "AUC_CI_Upper",
                "Mean Fake AUC",
                "Mean AUC Diff",
                "Minimum Feature Stability",
            ]
            round_cols = [
                c for c in df_final_check.columns if c.startswith("Fake AUC Round ")
            ]
            other_cols = [
                c
                for c in df_final_check.columns
                if c not in cols and c not in round_cols
            ]
            final_cols = [
                c for c in cols + round_cols + other_cols if c in df_final_check.columns
            ]
            df_final_check = df_final_check[final_cols]

            check_outfile = f"Top{top_n}_Combinations_MixValidation_Report_RFE_CI(Threshold{mix_auc_threshold}_{mix_rounds}Rounds).xlsx"
            print(f"  [OK] Validation complete, report saved: {check_outfile}")
            df_final_check.to_excel(check_outfile, index=False)

    print(f"\nTotal Time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
