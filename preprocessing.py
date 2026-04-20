"""Preprocessing builders per dataset."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(dataset_name: str, X):
    """Return a ColumnTransformer for the selected dataset."""
    numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("num", numeric_pipeline, numeric_features)]

    if dataset_name == "paysim" and "type" in categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, ["type"]))
        categorical_features = [c for c in categorical_features if c != "type"]

    if categorical_features:
        # Generic fallback for any extra categorical columns.
        fallback_cat = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat_extra", fallback_cat, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
