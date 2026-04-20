# Financial Fraud Detection – Report 2

## Short project description
This project implements machine learning experiments for fraud detection on highly imbalanced data. It uses two datasets: `creditcard.csv` and `PS_20174392719_1491204439457_log.csv`. The code compares multiple classification models and several imbalance-handling strategies in a reproducible workflow. Results are exported to files so they can be used directly in the report.

## Implemented models
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

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

## How to run the project
Run the main experiment script:

```bash
python main.py
```

Outputs are saved in:
- `outputs/`
- `outputs/plots/`

## Methodology note
To avoid data leakage, balancing methods are applied only to training data.
