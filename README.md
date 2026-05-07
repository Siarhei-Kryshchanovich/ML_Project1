# Financial Fraud Detection – Report 2

## Short project description
This project implements machine learning experiments for fraud detection on highly imbalanced data. It uses two datasets: `creditcard.csv` and `PS_20174392719_1491204439457_log.csv`. The code compares multiple classification models and several imbalance-handling strategies in a reproducible workflow. Results are exported to files so they can be used directly in the report.

## Implemented models
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- XGBoost

The project always has at least two ensemble models from scikit-learn:
Random Forest and Extra Trees. XGBoost is an additional ensemble when the
package is installed.

## Implemented imbalance handling methods
- baseline
- class weighting
- random undersampling
- random oversampling
- SMOTE
- ADASYN

## Main evaluation metrics
- ROC-AUC
- PR-AUC
- Recall
- Precision
- F1-score
- Accuracy
- Confusion matrix
- Expected cost

Expected cost is computed as:

```text
expected_cost = false_positives * COST_FP + false_negatives * COST_FN
```

The default configuration uses `COST_FP = 10` and `COST_FN = 500`, so missed
fraud is penalized more heavily than a false alert.

## Installation
Install dependencies:

```bash
pip install -r requirements.txt
```

`xgboost` is already listed in `requirements.txt`.
If there is any installation issue, install it manually:

```bash
pip install xgboost
```

If XGBoost is unavailable, the code may skip this model (depending on environment and installed packages).

## Expected dataset locations
- `data/creditcard.csv`
- `data/PS_20174392719_1491204439457_log.csv`

For convenience, the loader also accepts the same filenames placed directly in
the project root.

## How to run the project
Run the main experiment script:

```bash
python main.py
```

Run only PaySim:

```bash
python main.py --dataset paysim
```

Run only the credit card dataset:

```bash
python main.py --dataset creditcard
```

Outputs are saved in:
- `outputs/`
- `outputs/plots/`

Important output files:
- `outputs/comparison_table.csv` - all dataset/model/balancing combinations with test metrics and CV summaries.
- `outputs/hyperparameter_analysis.csv` - compact randomized hyperparameter analysis with PR-AUC ranks.
- `outputs/*__thresholds.csv` - threshold scans used to choose the validation threshold.
- `outputs/plots/*_roc.png` and `outputs/plots/*_pr.png` - ROC and precision-recall curves.
- `outputs/plots/*__cm_tuned.png` - confusion matrices at the tuned threshold.

## Methodology notes
- To avoid data leakage, preprocessing and balancing are inside imbalanced-learn pipelines.
- Sampling methods are fit only on the training split or training fold.
- Thresholds are selected on validation data and applied once on test data.
- Cross-validation is stratified and uses fixed random seeds for reproducibility.
- Hyperparameter analysis is enabled by default through `RUN_HYPERPARAMETER_ANALYSIS`.

## Report 2 grading checklist
- 4+ classification models: Logistic Regression, Decision Tree, Random Forest, Extra Trees, optional XGBoost.
- At least 2 ensemble models: Random Forest and Extra Trees, optional XGBoost.
- 3+ imbalance methods: class weighting, undersampling, oversampling, SMOTE, ADASYN.
- Required metrics: ROC-AUC, PR-AUC, Recall.
- Cost-sensitive evaluation: expected-cost metric and validation-based threshold tuning.
- Experiment automation: nested loops over datasets, models, balancing strategies, and output export.
- Hyperparameter analysis: `outputs/hyperparameter_analysis.csv`.
