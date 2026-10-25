"""Unit tests for the FAO-56 water-demand calculations."""

import numpy as np
import pandas as pd
import pytest

from src.water_demand import (
    crop_water_demand,
    effective_precipitation_usda_scs,
    extraterrestrial_radiation,
    irrigation_required,
    psychrometric_constant,
    reference_evapotranspiration,
    saturation_vapor_pressure,
)


class TestVaporPressure:
    def test_known_saturation_values(self):
        # Published values: es(20 C) ~ 2.34 kPa, es(30 C) ~ 4.24 kPa.
        assert saturation_vapor_pressure(20.0) == pytest.approx(2.34, abs=0.02)
        assert saturation_vapor_pressure(30.0) == pytest.approx(4.24, abs=0.02)

    def test_monotonic(self):
        temps = np.array([5.0, 15.0, 25.0, 35.0])
        es = saturation_vapor_pressure(temps)
        assert np.all(np.diff(es) > 0)

    def test_psychrometric_constant_sea_level(self):
        assert psychrometric_constant(0.0) == pytest.approx(0.0673, abs=0.001)


class TestRadiation:
    def test_equator_equinox_ra(self):
        # Solar-constant derivation: daily TOA insolation at the equator on
        # the equinox is Gsc * 24h / pi ~ 37.4-37.8 MJ/m2/day.
        ra = float(extraterrestrial_radiation(0.0, 80))
        assert 37.0 < ra < 38.5

    def test_winter_smaller_than_summer_at_mid_latitude(self):
        ra_summer = float(extraterrestrial_radiation(40.0, 172))
        ra_winter = float(extraterrestrial_radiation(40.0, 355))
        assert ra_summer > ra_winter
        assert ra_winter > 5.0

    def test_vectorised(self):
        doys = np.arange(1, 366)
        ra = extraterrestrial_radiation(18.52, doys)
        assert ra.shape == (365,)
        assert np.all(ra > 0)


class TestEffectivePrecipitation:
    def test_never_exceeds_total(self):
        p = np.array([0.0, 1.0, 10.0, 25.0, 80.0, 150.0, 300.0])
        pe = effective_precipitation_usda_scs(p)
        assert np.all(pe <= p + 1e-9)
        assert np.all(pe >= 0)

    def test_zero_rain_zero_effective(self):
        assert float(effective_precipitation_usda_scs(0.0)) == 0.0

    def test_small_rain_mostly_lost(self):
        # 5 mm storm: SCS daily adaptation keeps only a small fraction.
        pe = float(effective_precipitation_usda_scs(5.0))
        assert 0 < pe < 5.0


class TestReferenceET0:
    def test_plausible_daily_range(self):
        et0 = reference_evapotranspiration(
            tmax_c=35.0, tmin_c=20.0, rh_pct=40.0, solar_mj=25.0,
            wind_ms=2.0, latitude_deg=18.5, day_of_year=120,
        )
        et0 = float(et0)
        assert 4.0 < et0 < 10.0  # hot dry season day: high demand

    def test_humid_cloudy_lower_than_hot_clear(self):
        hot_clear = float(reference_evapotranspiration(38, 22, 30, 26, 2.0, 18.5, 120))
        wet_cloudy = float(reference_evapotranspiration(26, 22, 90, 10, 3.0, 18.5, 200))
        assert hot_clear > wet_cloudy

    def test_non_negative_array(self):
        tmax = np.full(5, 30.0)
        et0 = reference_evapotranspiration(tmax, tmax - 10, np.full(5, 60.0),
                                           np.full(5, 20.0), np.full(5, 1.5),
                                           18.5, np.arange(1, 6))
        assert np.all(np.asarray(et0) >= 0)


def _sample_weather_frame(n: int = 90) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "tmax_c": 30 + rng.normal(0, 2, n),
            "tmin_c": 18 + rng.normal(0, 2, n),
            "rh_pct": np.clip(60 + rng.normal(0, 10, n), 10, 100),
            "precip_mm": np.abs(rng.normal(3, 5, n)),
            "solar_mj": np.clip(20 + rng.normal(0, 3, n), 2, 30),
            "wind_ms": np.abs(rng.normal(2, 0.5, n)),
        },
        index=idx,
    )


class TestCropWaterDemand:
    def test_columns_and_kc_scaling(self):
        w = _sample_weather_frame()
        out = crop_water_demand(w, kc=1.0)
        for col in ["et0_mm", "etc_mm", "peff_mm", "net_irrig_mm", "gross_irrig_mm"]:
            assert col in out.columns
        out_kc2 = crop_water_demand(w, kc=1.2)
        assert np.allclose(out_kc2["etc_mm"], (out["et0_mm"] * 1.2).round(3))

    def test_gross_ge_net(self):
        out = crop_water_demand(_sample_weather_frame(), application_efficiency=0.7)
        assert np.all(out["gross_irrig_mm"] + 1e-9 >= out["net_irrig_mm"])

    def test_invalid_inputs_raise(self):
        w = _sample_weather_frame()
        with pytest.raises(ValueError):
            crop_water_demand(w, application_efficiency=1.5)
        with pytest.raises(ValueError):
            crop_water_demand(w, kc=0)

    def test_irrigation_required_threshold(self):
        out = crop_water_demand(_sample_weather_frame())
        label = irrigation_required(out, threshold_mm=3.0)
        assert set(label.unique()) <= {0, 1}
        assert (label == (out["net_irrig_mm"] > 3.0).astype(int)).all()
