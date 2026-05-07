"""Dataset loading utilities."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATASETS


def _resolve_dataset_path(path: Path) -> Path:
    """Return configured dataset path, with a root-level filename fallback."""
    if path.exists():
        return path

    root_fallback = Path(__file__).resolve().parent / path.name
    if root_fallback.exists():
        return root_fallback

    raise FileNotFoundError(
        f"Dataset file not found. Expected either '{path}' or '{root_fallback}'."
    )


def load_dataset(dataset_name: str, include_is_flagged_fraud: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and return X, y.

    For PaySim, raw identifier columns are excluded by design.
    isFlaggedFraud is excluded by default because in many setups it behaves
    like an operational flag that can leak post-transaction decisions.
    """
    cfg = DATASETS[dataset_name]
    df = pd.read_csv(_resolve_dataset_path(cfg["path"]))

    target_col = cfg["target"]
    y = df[target_col].astype(int)

    drop_cols = list(cfg["drop_cols"]) + [target_col]
    if dataset_name == "paysim" and not include_is_flagged_fraud:
        drop_cols.append("isFlaggedFraud")

    X = df.drop(columns=drop_cols, errors="ignore")
    return X, y
