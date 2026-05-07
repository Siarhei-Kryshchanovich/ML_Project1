# CODE_EXPLANATION

## 1. Short project overview
This project runs reproducible fraud-detection experiments on two datasets (`creditcard.csv` and PaySim), compares several ML models with multiple class-imbalance strategies, tunes decision thresholds using validation data, and saves metrics/plots for reporting.

## 2. Execution flow (main part)

Below is the real execution order when you run:

```bash
python main.py
```

1. **Python starts `main.py` and imports all modules.**  
   `main.py` imports configuration constants from `config.py`, data loading from `data_loader.py`, preprocessing from `preprocessing.py`, pipeline/model logic from `models.py`, evaluation helpers from `evaluation.py`, and utility helpers from `utils.py`.

2. **`__main__` block triggers experiment run.**  
   At the bottom of `main.py`, `run_all_experiments(include_is_flagged_fraud=False)` is called.

3. **Output directories are created.**  
   Inside `run_all_experiments`, `ensure_dir(OUTPUT_DIR)` and `ensure_dir(PLOTS_DIR)` create `outputs/` and `outputs/plots/` if missing.

4. **Base models are prepared.**  
   `get_base_models()` builds the model dictionary:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost (only if import succeeds in `models.py`)

5. **Optional XGBoost behavior is checked.**  
   If xgboost is not installed, `xgboost` is not present in `base_models`, and `main.py` prints an info message and skips it.

6. **Model list is filtered.**  
   `MODEL_NAMES` from `config.py` is filtered to names that really exist in `base_models`.

7. **Loop over datasets starts.**  
   `for dataset_name in DATASETS:` iterates through the configured datasets (`creditcard`, `paysim`) from `config.py`.

8. **Each dataset is loaded via `load_dataset`.**  
   `load_dataset(dataset_name, include_is_flagged_fraud=False)` does:
   - reads CSV from `DATASETS[dataset_name]["path"]`
   - selects target column from `DATASETS[dataset_name]["target"]`
   - converts target to integer
   - builds drop list from configured `drop_cols` + target column
   - for PaySim, additionally drops `isFlaggedFraud` by default
   - returns `X` (features) and `y` (target)

9. **Target columns are selected.**  
   - Credit card target: `Class`
   - PaySim target: `isFraud`

10. **Unnecessary columns are dropped.**  
    - Credit card: only target removed (no extra drop columns in config)
    - PaySim: `nameOrig`, `nameDest`, target (`isFraud`), and usually `isFlaggedFraud`

11. **Data split: train/validation/test.**  
    Two-step split in `main.py` with stratification:
    - Step A: `train_test_split(..., test_size=TEST_SIZE)` creates `train_temp` and `test`
    - Step B: `train_test_split(..., test_size=VAL_SIZE)` on `train_temp` creates `train` and `val`

    With current config (`TEST_SIZE=0.20`, `VAL_SIZE=0.25` of remaining), this is effectively:
    - ~60% train
    - ~20% validation
    - ~20% test

12. **Loop over models starts.**  
    For each available model in filtered `model_names`.

13. **Loop over balancing strategies starts.**  
    For each strategy in `BALANCING_STRATEGIES`:
    - `baseline`
    - `class_weight`
    - `undersample`
    - `oversample`
    - `smote`
    - `adasyn`

14. **Experiment name is built.**  
    `experiment_name(dataset_name, model_name, strategy)` creates names like `paysim__random_forest__smote`.

15. **Preprocessing is built (`build_preprocessor`).**  
    `build_preprocessor(dataset_name, X_train)`:
    - detects numeric columns via dtype
    - detects categorical columns via dtype
    - creates numeric pipeline: median imputation + standard scaling

16. **Numeric vs categorical handling details.**  
    - **Numeric features:** always go through imputer + scaler.
    - **Categorical features:**
      - For PaySim, if `type` exists, it gets a dedicated path (most-frequent imputation + one-hot encoding).
      - Any other categorical columns go through fallback categorical pipeline (also impute + one-hot).

17. **Training pipeline is built (`build_pipeline`).**  
    `build_pipeline(...)` creates an imbalanced-learn pipeline with ordered steps:
    1) preprocessor  
    2) sampler (only for sampling strategies)  
    3) model

18. **Balancing strategy selection logic.**  
    - `baseline`: no sampler, original model settings
    - `class_weight`: no sampler; class weighting is applied in `_apply_class_weight`:
      - sklearn models -> `class_weight="balanced"`
      - xgboost -> `scale_pos_weight = negatives/positives`
    - `undersample` -> `RandomUnderSampler`
    - `oversample` -> `RandomOverSampler`
    - `smote` -> `SMOTE`
    - `adasyn` -> `ADASYN`

19. **Model training happens.**  
    `pipeline.fit(X_train, y_train)` runs preprocessing + (optional resampling) + model fitting on training data.

20. **Validation probabilities are computed.**  
    `val_prob = pipeline.predict_proba(X_val)[:, 1]` extracts fraud probability for class `1` on validation set.

21. **Threshold tuning on validation set.**  
    `tune_threshold(y_val, val_prob, cost_fp, cost_fn)`:
    - scans thresholds from `0.01` to `0.99`
    - computes metrics and confusion matrix-based cost at each threshold
    - picks threshold with minimum expected cost
    - returns best threshold + full threshold table

22. **Threshold table is saved to CSV.**  
    Saved as:  
    `outputs/<experiment_name>__thresholds.csv`

23. **Final test probabilities are computed.**  
    `test_prob = pipeline.predict_proba(X_test)[:, 1]` on unseen test set.

24. **Final test evaluation is performed at two thresholds.**  
    `compute_metrics(...)` is called twice:
    - default threshold `0.5`
    - tuned threshold from validation

25. **How metrics are calculated.**  
    In `evaluation.py`, metrics include:
    - ROC-AUC (`roc_auc_score`)
    - PR-AUC (`average_precision_score`)
    - recall, precision, F1, accuracy
    - confusion matrix components (`tn, fp, fn, tp`)
    - expected cost = `fp * COST_FP + fn * COST_FN`

26. **Plots are generated and saved.**  
    - `save_curves(...)` saves ROC and PR curve images:
      - `outputs/plots/<experiment_name>_roc.png`
      - `outputs/plots/<experiment_name>_pr.png`
    - `save_confusion_matrix(...)` saves tuned-threshold confusion matrix:
      - `outputs/plots/<experiment_name>__cm_tuned.png`

27. **Cross-validation on train_temp is run.**  
    A fresh preprocessor/pipeline is rebuilt using `X_train_temp, y_train_temp`, then `cross_validate` with `StratifiedKFold(CV_FOLDS)` computes mean/std CV metrics.

28. **One result row is appended.**  
    A dictionary with dataset, model, strategy, tuned threshold, test metrics, default-cost metric, and CV summaries is appended to `results_rows`.

29. **Inner loops continue automatically.**  
    Flow repeats for next strategy, then next model, then next dataset.

30. **Final comparison table is built and saved.**  
    After all loops, `results_rows` -> DataFrame, sorted by dataset + expected cost + PR-AUC, then saved to:
    `outputs/comparison_table.csv`

## 3. Supporting file roles
- **`main.py`**: central orchestration; runs loops, splits data, trains/evaluates, saves outputs.
- **`config.py`**: all experiment constants (paths, datasets, costs, models, strategies, split ratios, CV folds).
- **`data_loader.py`**: dataset-specific loading, target extraction, and drop-column logic.
- **`preprocessing.py`**: feature-type detection and `ColumnTransformer` construction.
- **`models.py`**: model factory, sampler factory, class-weight adjustment, full pipeline builder.
- **`evaluation.py`**: metric computation, threshold tuning, ROC/PR plotting, confusion matrix plotting.
- **`utils.py`**: helper utilities for output directories and experiment naming.
- **`requirements.txt`**: required Python packages.
- **`README.md`**: short usage guide, datasets, methods, and run command.

## 4. Dataset-specific logic

### `creditcard.csv`
- Loaded from `data/creditcard.csv`.
- Target column is `Class`.
- No configured extra drop columns.
- Main flow: numeric preprocessing + chosen balancing/model strategy.

### PaySim (`PS_20174392719_1491204439457_log.csv`)
- Loaded from configured PaySim path in `config.py`.
- Target column is `isFraud`.
- Always drops identifier-like columns `nameOrig`, `nameDest`.
- By default also drops `isFlaggedFraud` (`include_is_flagged_fraud=False`) to reduce leakage risk.
- `type` is processed as categorical with one-hot encoding.

### Specific columns requested
- **`type`**: categorical transaction type, encoded via OHE in preprocessing.
- **`nameOrig`**: dropped in loader (identifier column).
- **`nameDest`**: dropped in loader (identifier column).
- **`isFlaggedFraud`**: excluded by default for methodological safety; can be included only if caller sets `include_is_flagged_fraud=True`.

## 5. Outputs
Generated artifacts are saved under `outputs/`:

1. **Per experiment threshold table (CSV)**  
   `outputs/<dataset>__<model>__<strategy>__thresholds.csv`  
   Contains metrics/cost for thresholds 0.01..0.99 and selected best threshold.

2. **Per experiment plots** in `outputs/plots/`:
   - ROC curve PNG
   - PR curve PNG
   - confusion matrix PNG (using tuned threshold)

3. **Global comparison table**  
   `outputs/comparison_table.csv`  
   Contains one row per dataset/model/strategy with test and CV summaries.

How to use these in Report 2:
- use `comparison_table.csv` to compare models/strategies across datasets,
- use per-experiment threshold CSVs to justify threshold choice,
- use ROC/PR/CM plots as visual evidence of performance and trade-offs.

## 6. Practical notes
- **Leakage prevention:** preprocessing and resampling are inside pipelines and fit only on training folds/splits; PaySim `isFlaggedFraud` is excluded by default.
- **Threshold tuning discipline:** threshold is tuned only on validation data, then applied to test data once.
- **Class imbalance handling:** strategy is explicitly controlled in loops, so comparisons are consistent.
- **XGBoost optional behavior:** if import fails, experiments still run for other models and XGBoost is skipped with info message.
- **Reproducibility:** random seeds come from `RANDOM_STATE` in `config.py` and are passed to splitters/models/samplers.
