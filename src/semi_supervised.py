"""Semi-supervised learning experiment (self-training with confidence filtering).

Design (kept deliberately simple and auditable):

1. The *training portion only* is split chronologically into a small
   LABELLED subset and a larger UNLABELLED subset.  The test set is never
   touched by pseudo-label generation.
2. A base model is trained on the labelled subset (supervised baseline).
3. The model predicts the unlabelled subset; pseudo-labels are kept only
   where a *documented* confidence criterion is met.
4. The model is retrained on labelled + accepted pseudo-labelled data.
5. Both the baseline and the retrained model are evaluated on the same
   untouched test set.

Confidence criteria (no invented scores):

- Classification (Logistic Regression): the predicted class probability
  max(P(class|x)) is a calibrated-ish confidence the model itself emits.
  Pseudo-labels are accepted when this probability >= the chosen threshold.

- Regression (KNN): an ensemble-style local uncertainty estimate — the
  standard deviation of the neighbouring training targets.  When the k
  closest historical days disagree strongly, the pseudo-label is unreliable.
  Pseudo-labels are accepted when that standard deviation <= the chosen
  threshold (in target units, mm/day).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import classification_metrics, regression_metrics
from .preprocessing import RANDOM_SEED


@dataclass
class SSLResult:
    mode: str
    labelled_fraction: float
    n_labelled: int
    n_unlabelled: int
    n_pseudo_accepted: int
    acceptance_pct: float
    baseline_metrics: dict[str, float]
    ssl_metrics: dict[str, float]
    rounds: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    @property
    def improvement(self) -> dict[str, float]:
        """Per-metric change (ssl minus baseline). Positive usually helps
        for accuracy/F1; for MAE/RMSE a negative change is an improvement."""
        keys = [k for k in self.baseline_metrics if k in self.ssl_metrics]
        return {k: self.ssl_metrics[k] - self.baseline_metrics[k] for k in keys}


def _split_labelled_unlabelled(
    X: pd.DataFrame, y: pd.Series, labelled_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Randomly (seeded) split the training window into labelled/unlabelled."""
    if not 0.01 <= labelled_fraction <= 0.95:
        raise ValueError("labelled_fraction must be within [0.01, 0.95].")
    rng = np.random.default_rng(seed)
    idx = X.index.to_numpy()
    perm = rng.permutation(len(idx))
    n_lab = max(1, int(len(idx) * labelled_fraction))
    lab_pos = np.sort(perm[:n_lab])
    unlab_pos = np.sort(perm[n_lab:])
    return X.iloc[lab_pos], y.iloc[lab_pos], X.iloc[unlab_pos], y.iloc[unlab_pos]


def _classification_model(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def _regression_model(k: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=k, weights="distance")),
    ])


def self_training_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labelled_fraction: float = 0.2,
    confidence_threshold: float = 0.9,
    max_rounds: int = 3,
    seed: int = RANDOM_SEED,
) -> SSLResult:
    """Self-training for the binary irrigation-required classifier."""
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be within [0.5, 1.0].")

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    X_lab, y_lab, X_unlab, _ = _split_labelled_unlabelled(X_train, y_train, labelled_fraction, seed)

    baseline = _classification_model(seed)
    baseline.fit(X_lab, y_lab)
    base_metrics = classification_metrics(y_test, baseline.predict(X_test), prefix="")

    accepted_X: list[pd.DataFrame] = []
    accepted_y: list[pd.Series] = []
    rounds: list[dict[str, Any]] = []
    total_accepted = 0
    remaining = X_unlab.copy()

    current = baseline
    for rnd in range(1, max_rounds + 1):
        if len(remaining) == 0:
            break
        proba = current.predict_proba(remaining)
        conf = proba.max(axis=1)
        pred = proba.argmax(axis=1)
        mask = conf >= confidence_threshold
        n_new = int(mask.sum())
        if n_new == 0:
            rounds.append({"round": rnd, "new_pseudo": 0, "cumulative_pseudo": total_accepted})
            break
        accepted_X.append(remaining[mask])
        accepted_y.append(pd.Series(pred[mask], index=remaining.index[mask], name="pseudo"))
        remaining = remaining[~mask]
        total_accepted += n_new
        rounds.append({"round": rnd, "new_pseudo": n_new, "cumulative_pseudo": total_accepted})

        X_aug = pd.concat([X_lab] + accepted_X)
        y_aug = pd.concat([y_lab] + accepted_y)
        current = _classification_model(seed)
        current.fit(X_aug, y_aug)

    ssl_metrics = classification_metrics(y_test, current.predict(X_test), prefix="")
    n_lab = len(X_lab)
    n_unlab = len(X_unlab)
    return SSLResult(
        mode="classification",
        labelled_fraction=labelled_fraction,
        n_labelled=n_lab,
        n_unlabelled=n_unlab,
        n_pseudo_accepted=total_accepted,
        acceptance_pct=round(100 * total_accepted / n_unlab, 1) if n_unlab else 0.0,
        baseline_metrics=base_metrics,
        ssl_metrics=ssl_metrics,
        rounds=rounds,
        notes=(
            f"Confidence rule: max class probability >= {confidence_threshold}. "
            "Baseline = same labelled subset, no pseudo-labels."
        ),
    )


def self_training_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labelled_fraction: float = 0.2,
    max_std_mm: float = 0.8,
    knn_k: int = 15,
    max_rounds: int = 3,
    seed: int = RANDOM_SEED,
) -> SSLResult:
    """Self-training for the ET0 regression task using KNN local dispersion.

    Confidence criterion: the standard deviation of the k nearest training
    targets.  Small dispersion => the model's neighbourhood agrees => the
    pseudo-label is considered reliable.
    """
    if max_std_mm <= 0:
        raise ValueError("max_std_mm must be positive.")

    X_lab, y_lab, X_unlab, _ = _split_labelled_unlabelled(X_train, y_train, labelled_fraction, seed)

    baseline = _regression_model(min(knn_k, len(X_lab)))
    baseline.fit(X_lab, y_lab)
    base_metrics = regression_metrics(y_test, baseline.predict(X_test), prefix="")

    accepted_X: list[pd.DataFrame] = []
    accepted_y: list[pd.Series] = []
    rounds: list[dict[str, Any]] = []
    total_accepted = 0
    remaining = X_unlab.copy()
    current = baseline

    for rnd in range(1, max_rounds + 1):
        if len(remaining) == 0:
            break
        scaler: StandardScaler = current.named_steps["scaler"]
        knn: KNeighborsRegressor = current.named_steps["model"]
        Xr = scaler.transform(remaining)
        distances, neighbour_idx = knn.kneighbors(Xr)
        neighbour_targets = knn._y[neighbour_idx]
        disp = neighbour_targets.std(axis=1)
        pred = neighbour_targets.mean(axis=1)
        mask = disp <= max_std_mm
        n_new = int(mask.sum())
        if n_new == 0:
            rounds.append({"round": rnd, "new_pseudo": 0, "cumulative_pseudo": total_accepted})
            break
        accepted_X.append(remaining[mask])
        accepted_y.append(pd.Series(np.round(pred[mask], 3), index=remaining.index[mask], name="pseudo"))
        remaining = remaining[~mask]
        total_accepted += n_new
        rounds.append({"round": rnd, "new_pseudo": n_new, "cumulative_pseudo": total_accepted})

        X_aug = pd.concat([X_lab] + accepted_X)
        y_aug = pd.concat([y_lab] + accepted_y)
        current = _regression_model(min(knn_k, len(X_aug)))
        current.fit(X_aug, y_aug)

    ssl_metrics = regression_metrics(y_test, current.predict(X_test), prefix="")
    n_unlab = len(X_unlab)
    return SSLResult(
        mode="regression",
        labelled_fraction=labelled_fraction,
        n_labelled=len(X_lab),
        n_unlabelled=n_unlab,
        n_pseudo_accepted=total_accepted,
        acceptance_pct=round(100 * total_accepted / n_unlab, 1) if n_unlab else 0.0,
        baseline_metrics=base_metrics,
        ssl_metrics=ssl_metrics,
        rounds=rounds,
        notes=(
            f"Confidence rule: std deviation of {min(knn_k, len(X_lab))} neighbour "
            f"targets <= {max_std_mm} mm/day. Baseline = same labelled subset."
        ),
    )
