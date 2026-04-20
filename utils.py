"""Utility helpers for IO and naming."""
from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def experiment_name(dataset_name: str, model_name: str, strategy: str) -> str:
    """Build stable experiment name."""
    return f"{dataset_name}__{model_name}__{strategy}"
