"""Evaluation metrics shared across the project.

Every metric is computed from actual predictions passed in by the caller;
nothing here fabricates results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

# Upper bound used to flag a failed model whose predictions are nonsense.
ET0_PLAUSIBLE_MAX_MM = 20.0


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """MAE, RMSE, R^2 for a regression prediction vector."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("Empty prediction vector.")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    bias = float(np.mean(y_pred - y_true))
    return {
        f"{prefix}mae": float(mae),
        f"{prefix}rmse": rmse,
        f"{prefix}r2": float(r2),
        f"{prefix}bias": bias,
    }


def classification_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """Accuracy, precision, recall, F1 and confusion-matrix entries."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between y_true and y_pred.")
    if len(y_true) == 0:
        raise ValueError("Empty prediction vector.")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}tn": tn,
        f"{prefix}fp": fp,
        f"{prefix}fn": fn,
        f"{prefix}tp": tp,
    }


def confusion_matrix_frame(y_true, y_pred) -> pd.DataFrame:
    """Human-readable 2x2 confusion matrix."""
    m = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])
    return pd.DataFrame(
        m,
        index=["Actual: no irrigation", "Actual: irrigation"],
        columns=["Predicted: no irrigation", "Predicted: irrigation"],
    )


def metrics_table(rows: dict[str, dict[str, float]], keys: list[str]) -> pd.DataFrame:
    """Build a comparison table (index = model name, columns = metrics)."""
    data = {name: {k: row.get(k, np.nan) for k in keys} for name, row in rows.items()}
    return pd.DataFrame(data).T


def flag_suspicious_predictions(y_pred, max_plausible: float = ET0_PLAUSIBLE_MAX_MM) -> str | None:
    """Warn if predictions fall outside a physically plausible range."""
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_pred < 0):
        return "Model produced negative ET0 predictions (physically impossible)."
    if np.nanmax(y_pred) > max_plausible:
        return f"Model produced ET0 predictions above {max_plausible} mm/day."
    return None
