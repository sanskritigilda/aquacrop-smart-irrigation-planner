"""Dataset preparation: feature tables, chronological splits, scaling.

Weather is time-ordered, so every train/test split in this project is
chronological (earliest days -> train, most recent days -> test) to prevent
future information leaking into past predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

# Weather inputs available before irrigation decisions are made.
FEATURE_COLUMNS = ["tmax_c", "tmin_c", "rh_pct", "precip_mm", "solar_mj", "wind_ms"]

TARGET_REGRESSION = "et0_mm"          # calculated FAO-56 target
TARGET_CLASSIFICATION = "irrig_required"


@dataclass
class Split:
    """Chronological train/test split of a feature/target frame."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def __post_init__(self) -> None:
        if len(self.X_train) == 0 or len(self.X_test) == 0:
            raise ValueError("Train and test sets must both be non-empty.")
        if self.X_train.index.max() >= self.X_test.index.min():
            raise ValueError("Chronological split violated: train contains future dates.")

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def build_feature_frame(df: pd.DataFrame, target: str = TARGET_REGRESSION) -> pd.DataFrame:
    """Select model features plus the requested target from a weather frame."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found; run water-demand first.")
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates found in the input frame.")
    out = df[FEATURE_COLUMNS + [target]].dropna().copy()
    if len(out) < 50:
        raise ValueError(f"Only {len(out)} usable rows; need at least 50 to model.")
    return out


def chronological_split(
    frame: pd.DataFrame,
    target: str = TARGET_REGRESSION,
    train_fraction: float = 0.8,
) -> Split:
    """Split a (features + target) frame by date: first ``train_fraction``
    of days for training, the remainder for testing."""
    if not 0.1 <= train_fraction <= 0.95:
        raise ValueError("train_fraction must be within [0.10, 0.95].")
    ordered = frame.sort_index()
    n = len(ordered)
    cut = int(n * train_fraction)
    cut = max(cut, 1)
    cut = min(cut, n - 1)
    train = ordered.iloc[:cut]
    test = ordered.iloc[cut:]
    return Split(
        X_train=train.drop(columns=[target]),
        X_test=test.drop(columns=[target]),
        y_train=train[target],
        y_test=test[target],
    )


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardise features using statistics from the TRAIN split only."""
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )
    return X_train_s, X_test_s, scaler


def add_lagged_features(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Optionally add lagged weather columns (e.g. yesterday's ET0 drivers).

    Only past values are used (positive lags), never future values.
    """
    out = df.copy()
    for lag in lags:
        if lag <= 0:
            raise ValueError("Lags must be positive (past) values.")
        for col in FEATURE_COLUMNS[:3]:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out.dropna()


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed numpy (and therefore every sklearn estimator using it)."""
    np.random.seed(seed)
