"""Run Report 2 fraud detection experiments."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from config import (
    BALANCING_STRATEGIES,
    COST_FN,
    COST_FP,
    CV_FOLDS,
    DATASETS,
    MODEL_NAMES,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)
from data_loader import load_dataset
from evaluation import compute_metrics, save_confusion_matrix, save_curves, tune_threshold
from models import XGB_AVAILABLE, build_pipeline, get_base_models
from preprocessing import build_preprocessor
from utils import ensure_dir, experiment_name

warnings.filterwarnings("ignore", category=UserWarning)


def _build_cv_scores():
    return {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }


def run_all_experiments(include_is_flagged_fraud: bool = False):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOTS_DIR)

    base_models = get_base_models()
    if "xgboost" not in base_models:
        print("[INFO] xgboost is not installed. XGBoost experiments will be skipped.")

    model_names = [m for m in MODEL_NAMES if m in base_models]

    results_rows = []

    for dataset_name in DATASETS:
        print(f"\n[INFO] Running dataset: {dataset_name}")
        X, y = load_dataset(dataset_name, include_is_flagged_fraud=include_is_flagged_fraud)

        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp,
            y_train_temp,
            test_size=VAL_SIZE,
            stratify=y_train_temp,
            random_state=RANDOM_STATE,
        )

        for model_name in model_names:
            model = base_models[model_name]
            for strategy in BALANCING_STRATEGIES:
                exp_name = experiment_name(dataset_name, model_name, strategy)
                print(f"  -> {exp_name}")

                preprocessor = build_preprocessor(dataset_name, X_train)
                pipeline = build_pipeline(
                    dataset_name=dataset_name,
                    preprocessor=preprocessor,
                    model=model,
                    model_name=model_name,
                    strategy=strategy,
                    y_train=y_train,
                )

                pipeline.fit(X_train, y_train)
                val_prob = pipeline.predict_proba(X_val)[:, 1]
                best_thr, threshold_df = tune_threshold(y_val, val_prob, cost_fp=COST_FP, cost_fn=COST_FN)
                threshold_df.to_csv(OUTPUT_DIR / f"{exp_name}__thresholds.csv", index=False)

                test_prob = pipeline.predict_proba(X_test)[:, 1]
                test_metrics_default = compute_metrics(y_test, test_prob, threshold=0.5, cost_fp=COST_FP, cost_fn=COST_FN)
                test_metrics_tuned = compute_metrics(y_test, test_prob, threshold=best_thr, cost_fp=COST_FP, cost_fn=COST_FN)

                y_pred_tuned = (test_prob >= best_thr).astype(int)
                save_curves(
                    y_true=y_test,
                    y_prob=test_prob,
                    output_path_prefix=Path(PLOTS_DIR / exp_name),
                    title_prefix=exp_name,
                )
                save_confusion_matrix(
                    y_true=y_test,
                    y_pred=y_pred_tuned,
                    output_path=Path(PLOTS_DIR / f"{exp_name}__cm_tuned.png"),
                    title=f"{exp_name} | threshold={best_thr:.2f}",
                )

                preprocessor_cv = build_preprocessor(dataset_name, X_train_temp)
                cv_pipeline = build_pipeline(
                    dataset_name=dataset_name,
                    preprocessor=preprocessor_cv,
                    model=clone(model),
                    model_name=model_name,
                    strategy=strategy,
                    y_train=y_train_temp,
                )
                cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                cv_res = cross_validate(
                    cv_pipeline,
                    X_train_temp,
                    y_train_temp,
                    cv=cv,
                    scoring=_build_cv_scores(),
                    n_jobs=-1,
                    return_train_score=False,
                )

                result_row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "strategy": strategy,
                    "best_threshold_from_val": best_thr,
                    "test_roc_auc": test_metrics_tuned["roc_auc"],
                    "test_pr_auc": test_metrics_tuned["pr_auc"],
                    "test_recall": test_metrics_tuned["recall"],
                    "test_precision": test_metrics_tuned["precision"],
                    "test_f1": test_metrics_tuned["f1"],
                    "test_accuracy": test_metrics_tuned["accuracy"],
                    "test_expected_cost": test_metrics_tuned["expected_cost"],
                    "test_tn": test_metrics_tuned["tn"],
                    "test_fp": test_metrics_tuned["fp"],
                    "test_fn": test_metrics_tuned["fn"],
                    "test_tp": test_metrics_tuned["tp"],
                    "test_expected_cost_at_0_5": test_metrics_default["expected_cost"],
                    "cv_roc_auc_mean": np.mean(cv_res["test_roc_auc"]),
                    "cv_roc_auc_std": np.std(cv_res["test_roc_auc"]),
                    "cv_pr_auc_mean": np.mean(cv_res["test_pr_auc"]),
                    "cv_pr_auc_std": np.std(cv_res["test_pr_auc"]),
                    "cv_recall_mean": np.mean(cv_res["test_recall"]),
                    "cv_precision_mean": np.mean(cv_res["test_precision"]),
                    "cv_f1_mean": np.mean(cv_res["test_f1"]),
                }
                results_rows.append(result_row)

    results_df = pd.DataFrame(results_rows).sort_values(
        by=["dataset", "test_expected_cost", "test_pr_auc"], ascending=[True, True, False]
    )
    results_path = OUTPUT_DIR / "comparison_table.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Saved comparison table: {results_path}")


if __name__ == "__main__":
    # keep default False for methodological safety in PaySim
    run_all_experiments(include_is_flagged_fraud=False)
