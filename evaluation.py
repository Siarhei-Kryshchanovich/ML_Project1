"""Evaluation, threshold tuning, and plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_cost_from_cm(cm: np.ndarray, cost_fp: int, cost_fn: int) -> int:
    """Compute expected cost from confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    return int(fp * cost_fp + fn * cost_fn)


def compute_metrics(y_true, y_prob, threshold: float, cost_fp: int, cost_fn: int) -> dict:
    """Compute classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "expected_cost": expected_cost_from_cm(cm, cost_fp=cost_fp, cost_fn=cost_fn),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def tune_threshold(y_true, y_prob, cost_fp: int, cost_fn: int) -> tuple[float, pd.DataFrame]:
    """Tune threshold on validation data by minimizing expected cost."""
    rows = []
    for thr in np.linspace(0.01, 0.99, 99):
        rows.append(compute_metrics(y_true, y_prob, threshold=float(thr), cost_fp=cost_fp, cost_fn=cost_fn))

    df = pd.DataFrame(rows)
    best_idx = df["expected_cost"].idxmin()
    best_thr = float(df.loc[best_idx, "threshold"])
    return best_thr, df


def save_curves(y_true, y_prob, output_path_prefix: Path, title_prefix: str):
    """Save ROC and PR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_roc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} - Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_pr.png", dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred, output_path: Path, title: str):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
