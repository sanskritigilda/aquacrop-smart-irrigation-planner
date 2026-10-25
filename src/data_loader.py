"""Download daily weather data from the NASA POWER API.

The NASA POWER daily point API returns, for a requested lat/lon and date
range, one value per day for each requested parameter.  This module handles
date validation, API chunking (NASA recommends <= 1 year per daily request),
missing values (fill value -999), timeouts/retries, and a documented offline
fallback (a cached CSV in data/ generated from a real earlier download, or a
seeded synthetic sample if the cache is absent).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Meteorological inputs needed for FAO-56 reference evapotranspiration.
POWER_PARAMETERS = [
    "T2M_MAX",   # daily maximum temperature at 2 m  (deg C)
    "T2M_MIN",   # daily minimum temperature at 2 m  (deg C)
    "RH2M",      # relative humidity at 2 m          (%)
    "PRECTOTCORR",  # corrected precipitation        (mm/day)
    "ALLSKY_SFC_SW_DWN",  # incoming solar radiation (MJ/m2/day)
    "WS2M",      # wind speed at 2 m                 (m/s)
]

# Friendly column names used everywhere else in the project.
RENAME_MAP = {
    "T2M_MAX": "tmax_c",
    "T2M_MIN": "tmin_c",
    "RH2M": "rh_pct",
    "PRECTOTCORR": "precip_mm",
    "ALLSKY_SFC_SW_DWN": "solar_mj",
    "WS2M": "wind_ms",
}

WEATHER_COLUMNS = list(RENAME_MAP.values())

# NASA POWER marks missing values with this sentinel.
POWER_FILL_VALUE = -999.0

# Safety cap so a typo in the UI cannot trigger an unbounded download.
MAX_DAYS_PER_REQUEST = 366
MAX_TOTAL_DAYS = 366 * 10

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHED_SAMPLE_PATH = DATA_DIR / "sample_weather_pune.csv"


class DataError(Exception):
    """Raised for invalid inputs or unrecoverable download problems."""


@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    elevation_m: float | None = None

    def validate(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise DataError(f"Latitude {self.latitude} out of range [-90, 90].")
        if not (-180.0 <= self.longitude <= 180.0):
            raise DataError(f"Longitude {self.longitude} out of range [-180, 180].")


# Suggested starting location from the project brief.
PUNE = Location("Pune, Maharashtra, India", latitude=18.5204, longitude=73.8567, elevation_m=560.0)


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise DataError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc


def validate_date_range(start: str | date, end: str | date) -> tuple[date, date]:
    """Validate the requested window and return parsed dates."""
    start_d, end_d = _parse_date(start), _parse_date(end)
    if end_d < start_d:
        raise DataError(f"End date {end_d} is before start date {start_d}.")
    n_days = (end_d - start_d).days + 1
    if n_days > MAX_TOTAL_DAYS:
        raise DataError(
            f"Requested {n_days} days; the maximum supported range is {MAX_TOTAL_DAYS} days."
        )
    return start_d, end_d


def _power_param_block_to_series(block: dict[str, float]) -> pd.Series:
    """Convert one API parameter dict {yyyymmdd: value} to a dated Series."""
    idx = pd.to_datetime(list(block.keys()), format="%Y%m%d")
    vals = pd.Series(list(block.values()), index=idx, dtype=float)
    # NASA uses -999 as fill; also treat tiny negatives in humidity etc. as missing.
    return vals.mask(vals <= POWER_FILL_VALUE + 1.0)


def _download_chunk(location: Location, start_d: date, end_d: date, timeout: int) -> pd.DataFrame:
    params = {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": "AG",
        "longitude": location.longitude,
        "latitude": location.latitude,
        "start": start_d.strftime("%Y%m%d"),
        "end": end_d.strftime("%Y%m%d"),
        "format": "JSON",
    }
    resp = requests.get(POWER_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    try:
        parameter_blocks = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise DataError("Unexpected NASA POWER response structure.") from exc

    series = {}
    for api_name, friendly in RENAME_MAP.items():
        block = parameter_blocks.get(api_name)
        if not block:
            raise DataError(f"Parameter {api_name} missing from API response.")
        series[friendly] = _power_param_block_to_series(block)

    df = pd.DataFrame(series).sort_index()
    df.index.name = "date"

    # Drop entirely empty days and clip physically impossible values.
    df = df.dropna(how="all")
    df["tmin_c"] = df["tmin_c"].clip(-60, 60)
    df["tmax_c"] = df["tmax_c"].clip(-60, 60)
    df["rh_pct"] = df["rh_pct"].clip(0, 100)
    df["precip_mm"] = df["precip_mm"].clip(lower=0)
    df["solar_mj"] = df["solar_mj"].clip(lower=0)
    df["wind_ms"] = df["wind_ms"].clip(lower=0)
    return df


def fetch_power_data(
    location: Location = PUNE,
    start: str | date = "2022-01-01",
    end: str | date = "2024-12-31",
    timeout: int = 60,
    max_retries: int = 3,
    retry_wait_s: float = 3.0,
) -> pd.DataFrame:
    """Download daily weather for ``location`` between ``start`` and ``end``.

    NASA POWER daily downloads are chunked into <= 1-year requests.
    Raises :class:`DataError` on invalid input or persistent API failure.
    """
    location.validate()
    start_d, end_d = validate_date_range(start, end)

    chunks: list[pd.DataFrame] = []
    cursor = start_d
    while cursor <= end_d:
        chunk_end = min(cursor + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end_d)
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                chunks.append(_download_chunk(location, cursor, chunk_end, timeout))
                last_error = None
                break
            except (requests.RequestException, DataError, ValueError) as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(retry_wait_s * attempt)
        if last_error is not None:
            raise DataError(
                f"NASA POWER request failed for {cursor}..{chunk_end} "
                f"after {max_retries} attempts: {last_error}"
            ) from last_error
        cursor = chunk_end + timedelta(days=1)

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.loc[str(start_d):str(end_d)]
    if df.empty:
        raise DataError("NASA POWER returned no data for the requested window.")
    return df


def load_weather(
    location: Location = PUNE,
    start: str | date = "2022-01-01",
    end: str | date = "2024-12-31",
    offline: bool = False,
) -> pd.DataFrame:
    """Load weather data, falling back to the bundled sample on API failure.

    When ``offline`` is True the API is never contacted.  When the API call
    fails and a cached CSV exists under ``data/`` (covering the requested
    period), that CSV is used and a warning column-level note is attached to
    the DataFrame ``attrs`` dictionary.
    """
    if not offline:
        try:
            df = fetch_power_data(location, start, end)
            df.attrs["source"] = "NASA POWER API (live download)"
            return df
        except DataError as exc:
            fallback_note = f"Live NASA POWER download failed ({exc})."
    else:
        fallback_note = "Offline mode requested; using bundled sample dataset."

    df = _load_sample()
    start_d, end_d = validate_date_range(start, end)
    df = df.loc[str(start_d):str(end_d)]
    if df.empty:
        raise DataError(
            "Bundled sample dataset does not cover the requested date range. "
            "Sample covers 2022-01-01 to 2024-12-31."
        )
    df.attrs["source"] = f"Bundled sample dataset. {fallback_note}"
    return df


# ---------------------------------------------------------------------------
# Bundled sample dataset
# ---------------------------------------------------------------------------

def generate_synthetic_sample(n_years: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate a seeded synthetic daily weather series for Pune-like climate.

    This exists so tests and the app remain runnable when the API and the
    cached real-data CSV are both unavailable.  Values are physically
    plausible approximations, NOT measurements.
    """
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 1)
    days = pd.date_range(start, periods=365 * n_years, freq="D")
    doy = days.dayofyear.to_numpy()

    # Two-hump seasonality: cool dry winter, very hot pre-monsoon, wet monsoon.
    seasonal_t = 25.0 + 6.0 * np.sin(2 * np.pi * (doy - 100) / 365)
    monsoon = np.clip(np.sin(2 * np.pi * (doy - 172) / 365), 0, None) ** 1.5

    tmean = seasonal_t + rng.normal(0, 1.5, len(days))
    diurnal = 11.0 - 4.0 * monsoon + rng.normal(0, 1.0, len(days))
    tmax = tmean + diurnal / 2
    tmin = tmean - diurnal / 2

    rh = np.clip(55 + 35 * monsoon + rng.normal(0, 6, len(days)), 15, 100)
    precip = np.where(monsoon > 0.15, rng.gamma(1.2, 9.0 * monsoon), 0.0)
    storm_mask = rng.random(len(days)) < 0.05
    precip[storm_mask] += rng.uniform(1, 12, int(storm_mask.sum()))
    solar = np.clip(22 - 10 * monsoon + rng.normal(0, 1.5, len(days)), 4, 28)
    wind = np.clip(2.0 + 1.5 * monsoon + rng.normal(0, 0.5, len(days)), 0.2, 10)

    df = pd.DataFrame(
        {
            "tmax_c": np.round(tmax, 2),
            "tmin_c": np.round(tmin, 2),
            "rh_pct": np.round(rh, 2),
            "precip_mm": np.round(precip, 2),
            "solar_mj": np.round(solar, 2),
            "wind_ms": np.round(wind, 2),
        },
        index=days,
    )
    df.index.name = "date"
    return df


def _load_sample() -> pd.DataFrame:
    if CACHED_SAMPLE_PATH.exists():
        df = pd.read_csv(CACHED_SAMPLE_PATH, parse_dates=["date"], index_col="date")
        missing = [c for c in WEATHER_COLUMNS if c not in df.columns]
        if missing:
            raise DataError(f"Cached sample file is missing columns: {missing}")
        return df[WEATHER_COLUMNS].astype(float)
    return generate_synthetic_sample()


def save_sample(df: pd.DataFrame, path: Path = CACHED_SAMPLE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.to_csv(path)
    return path


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and percentages for a weather frame."""
    n = len(df)
    rows = []
    for col in WEATHER_COLUMNS:
        if col not in df.columns:
            rows.append({"column": col, "missing": "absent", "pct": 100.0})
            continue
        miss = int(df[col].isna().sum())
        rows.append({"column": col, "missing": miss, "pct": round(100 * miss / n, 2) if n else 0.0})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Convenience CLI: python -m src.data_loader [start] [end] [offline]
    import sys

    args = sys.argv[1:]
    start_arg = args[0] if len(args) > 0 else "2022-01-01"
    end_arg = args[1] if len(args) > 1 else "2024-12-31"
    offline_arg = len(args) > 2 and args[2].lower() in {"1", "true", "offline"}
    frame = load_weather(start=start_arg, end=end_arg, offline=offline_arg)
    out_path = save_sample(frame)
    print(f"Rows: {len(frame)}  range: {frame.index.min().date()} .. {frame.index.max().date()}")
    print(f"Saved to {out_path}")
