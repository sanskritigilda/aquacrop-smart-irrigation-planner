"""End-to-end preparation helpers used by the Streamlit dashboard.

Keeps app code thin and makes the data flow explicit:

    weather (measured)  ->  FAO-56 ET0/ETc (calculated)
                        ->  irrigation-required label (derived, not observed)
                        ->  model splits / clustering / simulation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .data_loader import Location, PUNE, load_weather, missing_value_report
from .models import ClusterResult, ModelResult, train_all_regressors, train_kmeans_clusters
from .preprocessing import Split, build_feature_frame, chronological_split, set_global_seed
from .water_demand import crop_water_demand, irrigation_required


@dataclass
class PreparedDataset:
    raw: pd.DataFrame            # measured weather only
    enriched: pd.DataFrame       # weather + calculated targets
    regression_split: Split      # target = et0_mm
    classification_split: Split  # target = irrig_required
    regressors: dict[str, ModelResult]
    clusters: ClusterResult
    source_note: str


def prepare_dataset(
    location: Location = PUNE,
    start: str | date = "2022-01-01",
    end: str | date = "2024-12-31",
    offline: bool = False,
    kc: float = 1.0,
    application_efficiency: float = 0.70,
    irrigation_threshold_mm: float = 3.0,
    n_clusters: int = 4,
    poly_degree: int = 2,
    knn_k: int = 15,
    seed: int = 42,
) -> PreparedDataset:
    """Load, enrich, split and train everything the dashboard needs."""
    set_global_seed(seed)

    raw = load_weather(location=location, start=start, end=end, offline=offline)
    enriched = crop_water_demand(
        raw,
        latitude_deg=location.latitude,
        elevation_m=location.elevation_m or 0.0,
        kc=kc,
        application_efficiency=application_efficiency,
    )
    enriched["irrig_required"] = irrigation_required(enriched, threshold_mm=irrigation_threshold_mm)

    reg_frame = build_feature_frame(enriched, "et0_mm")
    cls_frame = build_feature_frame(enriched, "irrig_required")
    regression_split = chronological_split(reg_frame, "et0_mm")
    classification_split = chronological_split(cls_frame, "irrig_required")

    regressors = train_all_regressors(regression_split, poly_degree=poly_degree, knn_k=knn_k)
    clusters = train_kmeans_clusters(enriched, n_clusters=n_clusters, seed=seed)

    return PreparedDataset(
        raw=raw,
        enriched=enriched,
        regression_split=regression_split,
        classification_split=classification_split,
        regressors=regressors,
        clusters=clusters,
        source_note=raw.attrs.get("source", "unknown"),
    )


def weather_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact day-level summary of the measured weather variables."""
    stats = df.describe().T[["min", "25%", "50%", "75%", "max", "mean", "std"]].round(2)
    labels = {
        "tmax_c": "Max temperature (deg C)",
        "tmin_c": "Min temperature (deg C)",
        "rh_pct": "Relative humidity (%)",
        "precip_mm": "Precipitation (mm/day)",
        "solar_mj": "Solar radiation (MJ/m2/day)",
        "wind_ms": "Wind speed (m/s)",
        "et0_mm": "ET0 (mm/day)",
        "etc_mm": "ETc (mm/day)",
        "peff_mm": "Effective rain (mm/day)",
        "net_irrig_mm": "Net irrigation need (mm/day)",
    }
    stats.index = [labels.get(i, i) for i in stats.index]
    return stats


def missing_report(prepared: PreparedDataset) -> pd.DataFrame:
    return missing_value_report(prepared.raw)


__all__ = [
    "PreparedDataset",
    "prepare_dataset",
    "weather_summary",
    "missing_report",
]
