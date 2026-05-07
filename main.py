"""Run Report 2 fraud detection experiments."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from config import (
    BALANCING_STRATEGIES,
    COST_FN,
    COST_FP,
    CV_FOLDS,
    DATASETS,
    HYPERPARAMETER_MODELS,
    HYPERPARAMETER_N_ITER,
    HYPERPARAMETER_STRATEGIES,
    MODEL_NAMES,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    RUN_HYPERPARAMETER_ANALYSIS,
    TEST_SIZE,
    VAL_SIZE,
)
from data_loader import load_dataset
from evaluation import compute_metrics, save_confusion_matrix, save_curves, tune_threshold
from models import XGB_AVAILABLE, build_pipeline, get_base_models, get_hyperparameter_grid
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


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Report 2 fraud detection experiments.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        nargs="+",
        default=None,
        help="Dataset(s) to run. Omit to run all configured datasets.",
    )
    parser.add_argument(
        "--include-is-flagged-fraud",
        action="store_true",
        help="Include PaySim isFlaggedFraud feature. Default excludes it to reduce leakage risk.",
    )
    return parser.parse_args()


def _run_hyperparameter_analysis(dataset_name, X_train_temp, y_train_temp, base_models, model_names):
    """Run compact CV-based hyperparameter analysis and return one row per candidate."""
    rows = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for model_name in HYPERPARAMETER_MODELS:
        if model_name not in model_names:
            continue

        param_grid = get_hyperparameter_grid(model_name)
        if not param_grid:
            continue

        n_iter = min(HYPERPARAMETER_N_ITER, len(list(ParameterGrid(param_grid))))

        for strategy in HYPERPARAMETER_STRATEGIES:
            exp_name = experiment_name(dataset_name, model_name, strategy)
            print(f"  -> hyperparameters: {exp_name}")

            preprocessor = build_preprocessor(dataset_name, X_train_temp)
            pipeline = build_pipeline(
                dataset_name=dataset_name,
                preprocessor=preprocessor,
                model=clone(base_models[model_name]),
                model_name=model_name,
                strategy=strategy,
                y_train=y_train_temp,
            )
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=_build_cv_scores(),
                refit="pr_auc",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                return_train_score=False,
                error_score=np.nan,
            )
            search.fit(X_train_temp, y_train_temp)

            cv_results = pd.DataFrame(search.cv_results_)
            for _, result in cv_results.iterrows():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "strategy": strategy,
                        "rank_pr_auc": int(result["rank_test_pr_auc"]),
                        "mean_cv_pr_auc": result["mean_test_pr_auc"],
                        "std_cv_pr_auc": result["std_test_pr_auc"],
                        "mean_cv_roc_auc": result["mean_test_roc_auc"],
                        "std_cv_roc_auc": result["std_test_roc_auc"],
                        "mean_cv_recall": result["mean_test_recall"],
                        "std_cv_recall": result["std_test_recall"],
                        "mean_fit_time": result["mean_fit_time"],
                        "params": result["params"],
                    }
                )

    return rows


def run_all_experiments(include_is_flagged_fraud: bool = False, dataset_names: list[str] | None = None):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOTS_DIR)

    base_models = get_base_models()
    if "xgboost" not in base_models:
        print("[INFO] xgboost is not installed. XGBoost experiments will be skipped.")

    model_names = [m for m in MODEL_NAMES if m in base_models]

    results_rows = []
    hyperparameter_rows = []

    selected_datasets = list(DATASETS) if dataset_names is None else dataset_names

    for dataset_name in selected_datasets:
        print(f"\n[INFO] Running dataset: {dataset_name}")
        try:
            X, y = load_dataset(dataset_name, include_is_flagged_fraud=include_is_flagged_fraud)
        except FileNotFoundError as exc:
            print(f"[WARN] Skipping dataset '{dataset_name}': {exc}")
            continue

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

        if RUN_HYPERPARAMETER_ANALYSIS:
            hyperparameter_rows.extend(
                _run_hyperparameter_analysis(dataset_name, X_train_temp, y_train_temp, base_models, model_names)
            )

    if not results_rows:
        raise RuntimeError("No experiments were run because no configured dataset files were found.")

    results_df = pd.DataFrame(results_rows).sort_values(
        by=["dataset", "test_expected_cost", "test_pr_auc"], ascending=[True, True, False]
    )
    results_path = OUTPUT_DIR / "comparison_table.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Saved comparison table: {results_path}")

    if hyperparameter_rows:
        hyperparameter_df = pd.DataFrame(hyperparameter_rows).sort_values(
            by=["dataset", "model", "strategy", "rank_pr_auc"]
        )
        hyperparameter_path = OUTPUT_DIR / "hyperparameter_analysis.csv"
        hyperparameter_df.to_csv(hyperparameter_path, index=False)
        print(f"[INFO] Saved hyperparameter analysis: {hyperparameter_path}")


if __name__ == "__main__":
    args = _parse_args()
    run_all_experiments(
        include_is_flagged_fraud=args.include_is_flagged_fraud,
        dataset_names=args.dataset,
    )
