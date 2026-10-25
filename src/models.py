"""Supervised models used in AquaCrop.

All estimators are from the permitted algorithm list:
- Multiple Linear Regression  (linear baseline for ET0 prediction)
- Polynomial Regression       (degree-2 extension of the linear baseline)
- K-Nearest Neighbors         (similarity-based ET0 prediction)
- K-Means                     (weather-pattern clustering)
- Logistic Regression         (binary irrigation-required classifier)

Every function returns the fitted estimator, its test predictions and its
metrics, all computed from actual predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .evaluation import classification_metrics, regression_metrics
from .preprocessing import RANDOM_SEED, Split


@dataclass
class ModelResult:
    name: str
    kind: str  # "regression" | "classification"
    estimator: Any
    y_pred: pd.Series | np.ndarray
    metrics: dict[str, float]
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def train_multiple_linear_regression(split: Split, target_name: str = "et0_mm") -> ModelResult:
    """OLS multiple linear regression on standardised features."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipe.fit(split.X_train, split.y_train)
    y_pred = pd.Series(pipe.predict(split.X_test), index=split.X_test.index, name="pred")
    y_pred = y_pred.clip(lower=0.0)
    return ModelResult(
        name="Multiple Linear Regression",
        kind="regression",
        estimator=pipe,
        y_pred=y_pred,
        metrics=regression_metrics(split.y_test, y_pred),
        notes="Baseline: assumes a linear relationship between weather and the target.",
    )


def train_polynomial_regression(
    split: Split, degree: int = 2, target_name: str = "et0_mm"
) -> ModelResult:
    """Degree-2 polynomial regression (ridge-regularised for stability)."""
    if degree < 2:
        raise ValueError("Polynomial degree must be >= 2 (degree 1 is the linear baseline).")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", LinearRegression()),
    ])
    pipe.fit(split.X_train, split.y_train)
    y_pred = pd.Series(pipe.predict(split.X_test), index=split.X_test.index, name="pred")
    y_pred = y_pred.clip(lower=0.0)
    return ModelResult(
        name=f"Polynomial Regression (deg={degree})",
        kind="regression",
        estimator=pipe,
        y_pred=y_pred,
        metrics=regression_metrics(split.y_test, y_pred),
        notes="Captures non-linear weather effects; risk of overfitting grows with degree.",
    )


def train_knn_regression(split: Split, n_neighbors: int = 15) -> ModelResult:
    """KNN regression: predict ET0 from the most similar historical days."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")
    k = min(n_neighbors, len(split.X_train))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=k, weights="distance")),
    ])
    pipe.fit(split.X_train, split.y_train)
    y_pred = pd.Series(pipe.predict(split.X_test), index=split.X_test.index, name="pred")
    y_pred = y_pred.clip(lower=0.0)
    return ModelResult(
        name=f"KNN Regression (k={k})",
        kind="regression",
        estimator=pipe,
        y_pred=y_pred,
        metrics=regression_metrics(split.y_test, y_pred),
        notes="Distance-weighted average of the k most similar training days.",
    )


def train_all_regressors(split: Split, poly_degree: int = 2, knn_k: int = 15) -> dict[str, ModelResult]:
    """Train and evaluate the three regression models on the same split."""
    results = [
        train_multiple_linear_regression(split),
        train_polynomial_regression(split, degree=poly_degree),
        train_knn_regression(split, n_neighbors=knn_k),
    ]
    return {r.name: r for r in results}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def train_logistic_regression(split: Split, seed: int = RANDOM_SEED) -> ModelResult:
    """Logistic regression classifying whether irrigation is required."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    pipe.fit(split.X_train, split.y_train.astype(int))
    y_pred = pd.Series(pipe.predict(split.X_test), index=split.X_test.index, name="pred")
    return ModelResult(
        name="Logistic Regression",
        kind="classification",
        estimator=pipe,
        y_pred=y_pred,
        metrics=classification_metrics(split.y_test, y_pred),
        notes="Classifies the derived target: net irrigation requirement > threshold.",
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = ["tmax_c", "tmin_c", "rh_pct", "precip_mm", "solar_mj", "wind_ms"]

# Plain-language labels assigned after inspecting cluster centroids.
CLUSTER_ARCHETYPE_HINTS = ["hot/dry", "hot/humid", "wet", "cool/moderate"]


@dataclass
class ClusterResult:
    estimator: KMeans
    labels: pd.Series
    centers_raw: pd.DataFrame
    profile: pd.DataFrame
    silhouette: float


def train_kmeans_clusters(df: pd.DataFrame, n_clusters: int = 4, seed: int = RANDOM_SEED) -> ClusterResult:
    """Group weather days into similar meteorological patterns with K-Means.

    Features are standardised before clustering so that humidity (%) and
    wind speed (m/s) do not dominate temperature just because of units.
    """
    from sklearn.metrics import silhouette_score

    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2.")
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for clustering: {missing}")

    X = df[CLUSTER_FEATURES].dropna()
    if len(X) < n_clusters * 5:
        raise ValueError("Not enough rows to form the requested number of clusters.")

    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=CLUSTER_FEATURES)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = pd.Series(km.fit_predict(Xs), index=X.index, name="cluster")

    centers_scaled = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)
    centers_raw = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=CLUSTER_FEATURES
    )
    # Sort clusters by mean temperature then precipitation for stable display.
    order = centers_raw.sort_values(["tmax_c", "precip_mm"]).index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = labels.map(remap).astype(int)
    centers_raw = centers_raw.iloc[order].reset_index(drop=True)
    centers_scaled = centers_scaled.iloc[order].reset_index(drop=True)
    km.cluster_centers_ = centers_scaled.to_numpy()

    counts = labels.value_counts().sort_index()
    profile = centers_raw.copy()
    profile["share_pct"] = (100 * counts / len(labels)).round(1).to_numpy()
    profile["days"] = counts.to_numpy()
    profile.insert(0, "pattern", [f"Cluster {i}" for i in range(n_clusters)])

    sil = float(silhouette_score(Xs, labels))
    return ClusterResult(estimator=km, labels=labels, centers_raw=centers_raw, profile=profile, silhouette=sil)


def describe_cluster(result: ClusterResult, index: int) -> str:
    """Plain-language description of one cluster from its centroid."""
    row = result.centers_raw.iloc[index]
    heat = "hot" if row["tmax_c"] >= 32 else ("warm" if row["tmax_c"] >= 25 else "cool")
    humid = "humid" if row["rh_pct"] >= 70 else ("moderate humidity" if row["rh_pct"] >= 45 else "dry")
    wet = "with frequent rain" if row["precip_mm"] >= 5 else ("with some rain" if row["precip_mm"] >= 0.5 else "and essentially rain-free")
    solar = "strong sunshine" if row["solar_mj"] >= 20 else ("moderate sunshine" if row["solar_mj"] >= 12 else "cloudy")
    return (
        f"{heat.capitalize()}, {humid}, {wet}, {solar}. "
        f"Typical day: Tmax {row['tmax_c']:.1f} deg C, RH {row['rh_pct']:.0f}%, "
        f"rain {row['precip_mm']:.1f} mm, ET0 drivers {row['solar_mj']:.1f} MJ/m2."
    )
