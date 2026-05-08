"""Project configuration for Report 2 fraud detection experiments."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.25  # 25% of remaining train_temp -> 20% of full data
N_JOBS = -1
CV_FOLDS = 3
RUN_HYPERPARAMETER_ANALYSIS = True
HYPERPARAMETER_N_ITER = 6

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

MODEL_NAMES = ["logreg", "decision_tree", "random_forest", "extra_trees", "xgboost"]
# MODEL_NAMES = ["extra_trees"]
HYPERPARAMETER_MODELS = ["logreg", "decision_tree", "random_forest", "extra_trees", "xgboost"]
HYPERPARAMETER_STRATEGIES = ["class_weight", "smote"]

# Dataset-specific controls for hyperparameter analysis.
# PaySim is very large, so we intentionally constrain tuning to avoid
# RAM spikes from nested CV + resampling + parallel tree ensembles.
HYPERPARAMETER_DATASET_SETTINGS = {
    "default": {
        "strategies": HYPERPARAMETER_STRATEGIES,
        "cv_folds": CV_FOLDS,
        "n_iter": HYPERPARAMETER_N_ITER,
        "subsample_frac": 1.0,
        "search_n_jobs": -1,
        "search_pre_dispatch": "2*n_jobs",
        "model_n_jobs_overrides": {},
    },
    "paysim": {
        # Keep methodologically valid CV tuning, but skip memory-heavy synthetic
        # oversamplers with large ensembles on full PaySim.
        "strategies": HYPERPARAMETER_STRATEGIES,
        # Fewer folds lowers concurrent fit memory use.
        "cv_folds": 2,
        # Use a stratified subsample for robust but lighter tuning.
        "subsample_frac": 0.30,
        # Single-process search avoids joblib duplicating large arrays.
        "search_n_jobs": 1,
        "search_pre_dispatch": 1,
        # Tree ensembles and xgboost are forced to single-thread in tuning.
        "model_n_jobs_overrides": {
            "random_forest": 2,
            "extra_trees": 2,
            "xgboost": 2,
        },
        # Smaller randomized search budget for PaySim.
        "n_iter": 4,
    },
}
