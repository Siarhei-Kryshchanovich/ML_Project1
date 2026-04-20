"""Project configuration for Report 2 fraud detection experiments."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.25  # 25% of remaining train_temp -> 20% of full data
N_JOBS = -1
CV_FOLDS = 3

# Costs used for cost-sensitive evaluation
COST_FP = 10
COST_FN = 500

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

DATASETS = {
    "creditcard": {
        "path": DATA_DIR / "creditcard.csv",
        "target": "Class",
        "drop_cols": [],
    },
    "paysim": {
        "path": DATA_DIR / "PS_20174392719_1491204439457_log.csv",
        "target": "isFraud",
        "drop_cols": ["nameOrig", "nameDest"],
    },
}

BALANCING_STRATEGIES = [
    "baseline",
    "class_weight",
    "undersample",
    "oversample",
    "smote",
    "adasyn",
]

MODEL_NAMES = ["logreg", "decision_tree", "random_forest", "xgboost"]
