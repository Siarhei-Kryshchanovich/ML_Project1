"""Model and balancing strategy builders."""
from __future__ import annotations

from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBClassifier = None


def get_base_models() -> dict:
    """Return dictionary of base models."""
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=None),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def get_hyperparameter_grid(model_name: str) -> dict:
    """Return a compact model-specific grid for Report 2 hyperparameter analysis."""
    grids = {
        "logreg": {
            "model__C": [0.1, 1.0, 3.0, 10.0],
        },
        "decision_tree": {
            "model__max_depth": [4, 8, 12, None],
            "model__min_samples_leaf": [1, 5, 20],
        },
        "random_forest": {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [8, 12, None],
            "model__min_samples_leaf": [1, 2, 5],
        },
        "extra_trees": {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [8, 12, None],
            "model__min_samples_leaf": [1, 2, 5],
        },
        "xgboost": {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [3, 6, 8],
            "model__learning_rate": [0.03, 0.08, 0.15],
            "model__subsample": [0.8, 1.0],
        },
    }
    return grids.get(model_name, {})


def get_sampler(strategy: str):
    """Return sampler for resampling strategy."""
    samplers = {
        "undersample": RandomUnderSampler(random_state=RANDOM_STATE),
        "oversample": RandomOverSampler(random_state=RANDOM_STATE),
        "smote": SMOTE(random_state=RANDOM_STATE),
        "adasyn": ADASYN(random_state=RANDOM_STATE),
    }
    return samplers.get(strategy)


def _apply_class_weight(model, model_name: str, y_train):
    """Return a cloned model with class-weight adjustments."""
    model = clone(model)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    ratio = neg / max(pos, 1)

    if model_name in {"logreg", "decision_tree", "random_forest", "extra_trees"}:
        model.set_params(class_weight="balanced")
    elif model_name == "xgboost" and hasattr(model, "set_params"):
        model.set_params(scale_pos_weight=ratio)
    return model


def build_pipeline(
    dataset_name: str,
    preprocessor,
    model,
    model_name: str,
    strategy: str,
    y_train,
) -> ImbPipeline:
    """Create an imblearn pipeline with optional balancing and class weighting."""
    model_to_use = clone(model)
    if strategy == "class_weight":
        model_to_use = _apply_class_weight(model_to_use, model_name, y_train)

    sampler = get_sampler(strategy)
    steps = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model_to_use))

    return ImbPipeline(steps=steps)
