"""Crop water demand estimated with the FAO-56 Penman-Monteith method.

NASA POWER supplies weather measurements only.  It does NOT supply reference
evapotranspiration (ET0), crop water demand, soil moisture, or irrigation
labels.  Everything computed in this module is therefore a *calculated
target* derived from measurements via the FAO-56 dual crop coefficient
manual (Allen et al., 1998):

1. Daily reference ET0 from the FAO-56 Penman-Monteith equation (eq. 53).
   Required inputs (Tmax, Tmin, RH, solar radiation, wind speed) are all
   available from NASA POWER.  Approximations used and documented:
   - Soil heat flux G = 0 for daily time steps (FAO-56 section 1.2).
   - Clear-sky radiation Rso = (0.75 + 2e-5 * z) * Ra, z = site elevation.
   - Wind speed measured at 2 m (POWER WS2M) is used directly; FAO-56
     standardises to 2 m, so no height correction is applied.
2. Crop evapotranspiration ETc = Kc * ET0 with a single, user-configurable
   crop coefficient Kc (default 1.0 = reference grass, i.e. ETc = ET0).
3. Effective rainfall Pe with the USDA-SCS method (mm/day).
4. Net irrigation requirement INR = max(0, ETc - Pe), and gross irrigation
   requirement GIR = INR / application efficiency.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Physical / FAO-56 constants
LAMBDA = 2.45          # latent heat of vaporisation of water, MJ/kg
ALBEDO = 0.23          # albedo of reference grass surface
SIGMA = 4.903e-9       # Stefan-Boltzmann constant, MJ/K^4 m^2 day
G_AS_PER_DAY = 900.0   # wind-function constant (FAO-56 eq. 53)
DEFAULT_ELEVATION_M = 560.0  # Pune, used when site elevation unknown


def saturation_vapor_pressure(temp_c: float | np.ndarray) -> float | np.ndarray:
    """FAO-56 eq. 11: saturation vapor pressure in kPa at temperature temp_c."""
    return 0.6108 * np.exp(17.27 * np.asarray(temp_c, dtype=float) / (np.asarray(temp_c, dtype=float) + 237.3))


def slope_vapor_pressure_deficit(temp_c: float | np.ndarray) -> float | np.ndarray:
    """FAO-56 eq. 13: slope of the saturation vapor pressure curve, kPa/degC."""
    t = np.asarray(temp_c, dtype=float)
    es = saturation_vapor_pressure(t)
    return 4098.0 * es / (t + 237.3) ** 2


def atmospheric_pressure(elevation_m: float = DEFAULT_ELEVATION_M) -> float:
    """FAO-56 eq. 3: atmospheric pressure at elevation z, kPa."""
    return 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26


def psychrometric_constant(elevation_m: float = DEFAULT_ELEVATION_M) -> float:
    """FAO-56 eq. 8: psychrometric constant, kPa/degC."""
    return 0.00163 * atmospheric_pressure(elevation_m) / LAMBDA


def extraterrestrial_radiation(latitude_deg: float, day_of_year: int | np.ndarray) -> float | np.ndarray:
    """FAO-56 Annex 2 / eq. 21: extraterrestrial radiation Ra, MJ/m2/day.

    Ra = (24*60/pi) * Gsc * dr * (ws*sin(phi)*sin(delta) + cos(phi)*cos(delta)*sin(ws))
    with Gsc = 0.0820 MJ/m2/min, so (24*60/pi)*Gsc = 37.6.
    """
    doy = np.asarray(day_of_year, dtype=float)
    phi = math.radians(latitude_deg)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-math.tan(phi) * np.tan(delta), -1.0, 1.0))
    return 37.6 * dr * (ws * math.sin(phi) * np.sin(delta)
                        + math.cos(phi) * np.cos(delta) * np.sin(ws))


def clearsky_radiation(ra: float | np.ndarray, elevation_m: float = DEFAULT_ELEVATION_M) -> float | np.ndarray:
    """FAO-56 eq. 37: clear-sky radiation Rso, MJ/m2/day."""
    return (0.75 + 2e-5 * elevation_m) * np.asarray(ra, dtype=float)


def net_radiation(
    solar_mj: float | np.ndarray,
    tmax_c: float | np.ndarray,
    tmin_c: float | np.ndarray,
    ea_kpa: float | np.ndarray,
    ra: float | np.ndarray,
    elevation_m: float = DEFAULT_ELEVATION_M,
    albedo: float = ALBEDO,
) -> float | np.ndarray:
    """FAO-56 eqs. 38 and 39: net radiation Rn, MJ/m2/day.

    Rn = Rns - Rnl with Rns = (1 - albedo) * Rs and the FAO-56 longwave
    approximation that requires only Rs/Rso, temperature extremes and
    actual vapor pressure.
    """
    solar = np.asarray(solar_mj, dtype=float)
    rso = clearsky_radiation(ra, elevation_m)
    ratio = np.clip(solar / np.maximum(rso, 1e-6), 0.0, 1.3)
    rns = (1.0 - albedo) * solar

    tmax_k = np.asarray(tmax_c, dtype=float) + 273.16
    tmin_k = np.asarray(tmin_c, dtype=float) + 273.16
    rnl = (SIGMA * ((tmax_k ** 4 + tmin_k ** 4) / 2.0)
           * (0.34 - 0.14 * np.sqrt(np.clip(ea_kpa, 0.0, None)))
           * (1.35 * ratio ** 0.5 - 0.35))
    return rns - rnl


def reference_evapotranspiration(
    tmax_c: float | np.ndarray,
    tmin_c: float | np.ndarray,
    rh_pct: float | np.ndarray,
    solar_mj: float | np.ndarray,
    wind_ms: float | np.ndarray,
    latitude_deg: float,
    day_of_year: int | np.ndarray,
    elevation_m: float = DEFAULT_ELEVATION_M,
) -> float | np.ndarray:
    """Daily FAO-56 reference evapotranspiration ET0, mm/day.

    Implements FAO-56 eq. 53 with G = 0 (daily step).  Scalar or array
    inputs are accepted; array inputs must broadcast together.
    """
    tmax = np.asarray(tmax_c, dtype=float)
    tmin = np.asarray(tmin_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    solar = np.asarray(solar_mj, dtype=float)
    wind = np.asarray(wind_ms, dtype=float)

    tmean = (tmax + tmin) / 2.0
    es = (saturation_vapor_pressure(tmax) + saturation_vapor_pressure(tmin)) / 2.0
    ea = es * rh / 100.0
    delta = slope_vapor_pressure_deficit(tmean)
    gamma = psychrometric_constant(elevation_m)
    ra = extraterrestrial_radiation(latitude_deg, day_of_year)
    rn = net_radiation(solar, tmax, tmin, ea, ra, elevation_m)

    numerator = 0.408 * delta * rn + gamma * (G_AS_PER_DAY / (tmean + 273.0)) * wind * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * wind)
    et0 = numerator / np.maximum(denominator, 1e-9)
    return np.clip(et0, 0.0, None)


def effective_precipitation_usda_scs(precip_mm: float | np.ndarray) -> float | np.ndarray:
    """USDA-SCS effective rainfall, mm/day.

    The classic SCS curve-fit is defined for monthly rainfall in mm:
        P <= 250 mm:  Pe = P^2 / 250
        P  > 250 mm:  Pe = 208 + 0.105 * P
    Applied here per day (a standard small-scale adaptation for daily water
    balance models); results are capped at P because effective rainfall can
    never exceed total rainfall.
    """
    p = np.clip(np.asarray(precip_mm, dtype=float), 0.0, None)
    pe = np.where(p <= 250.0, p ** 2 / 250.0, 208.0 + 0.105 * p)
    return np.minimum(pe, p)


def crop_water_demand(
    df: pd.DataFrame,
    latitude_deg: float = 18.5204,
    elevation_m: float = DEFAULT_ELEVATION_M,
    kc: float = 1.0,
    application_efficiency: float = 0.70,
) -> pd.DataFrame:
    """Add ET0, ETc, effective rainfall and irrigation-requirement columns.

    Parameters
    ----------
    df : weather frame with columns tmax_c, tmin_c, rh_pct, precip_mm,
         solar_mj, wind_ms and a datetime index named 'date'.
    kc : single crop coefficient; default 1.0 means ETc equals ET0
         (reference grass).  Time-varying Kc tables are out of scope.
    application_efficiency : fraction of applied water that reaches the
         root zone (0 < Ea <= 1), used to convert net to gross irrigation.
    """
    if not 0 < application_efficiency <= 1:
        raise ValueError("application_efficiency must be in (0, 1].")
    if kc <= 0:
        raise ValueError("kc must be positive.")

    out = df.copy()
    doy = out.index.dayofyear.to_numpy()
    et0 = reference_evapotranspiration(
        out["tmax_c"].to_numpy(),
        out["tmin_c"].to_numpy(),
        out["rh_pct"].to_numpy(),
        out["solar_mj"].to_numpy(),
        out["wind_ms"].to_numpy(),
        latitude_deg=latitude_deg,
        day_of_year=doy,
        elevation_m=elevation_m,
    )
    out["et0_mm"] = np.round(np.asarray(et0, dtype=float), 3)
    out["etc_mm"] = np.round(out["et0_mm"] * kc, 3)
    out["peff_mm"] = np.round(effective_precipitation_usda_scs(out["precip_mm"].to_numpy()), 3)
    out["net_irrig_mm"] = np.round(np.maximum(out["etc_mm"] - out["peff_mm"], 0.0), 3)
    out["gross_irrig_mm"] = np.round(out["net_irrig_mm"] / application_efficiency, 3)
    return out


def irrigation_required(df: pd.DataFrame, threshold_mm: float = 3.0) -> pd.Series:
    """Binary derived label: 1 when the net irrigation requirement exceeds
    ``threshold_mm`` on that day.

    This is a *calculated* label from FAO-56 outputs, NOT a verified field
    irrigation decision.  It is used only as a learning target so that
    logistic regression can be demonstrated on a defensible binary problem.
    """
    if threshold_mm <= 0:
        raise ValueError("threshold_mm must be positive.")
    return (df["net_irrig_mm"] > threshold_mm).astype(int).rename("irrig_required")
