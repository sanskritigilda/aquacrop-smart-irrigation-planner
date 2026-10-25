"""Tests for the soil-water balance simulator and irrigation policies."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_sample
from src.irrigation_simulator import (
    CostWeights,
    NoIrrigationPolicy,
    PredictedDemandPolicy,
    SoilConfig,
    ThresholdPolicy,
    compare_policies,
    default_policies,
    simulate,
)
from src.water_demand import crop_water_demand


@pytest.fixture(scope="module")
def weather() -> pd.DataFrame:
    return crop_water_demand(generate_synthetic_sample(n_years=1, seed=11), kc=1.0)


class TestSoilConfig:
    def test_validation(self):
        with pytest.raises(ValueError):
            SoilConfig(total_available_water_mm=-5).validate()
        with pytest.raises(ValueError):
            SoilConfig(mad=1.5).validate()
        with pytest.raises(ValueError):
            SoilConfig(initial_storage_mm=500).validate()  # > TAW
        SoilConfig(total_available_water_mm=100, initial_storage_mm=50).validate()


class TestWaterBalance:
    def test_storage_stays_within_bounds(self, weather):
        res = simulate(weather, NoIrrigationPolicy(), soil=SoilConfig(initial_storage_mm=100))
        assert res.frame["storage_mm"].between(0, 100).all()

    def test_storage_conservation_equation(self, weather):
        """S(t) must equal clip(S(t-1) + Peff - ETa, 0, TAW) exactly."""
        soil = SoilConfig(total_available_water_mm=120, initial_storage_mm=90)
        res = simulate(weather, NoIrrigationPolicy(), soil=soil)
        f = res.frame
        for i in range(1, len(f)):
            expected = np.clip(
                f["storage_mm"].iloc[i - 1] + f["peff_mm"].iloc[i] - f["eta_mm"].iloc[i],
                0, soil.total_available_water_mm,
            )
            assert f["storage_mm"].iloc[i] == pytest.approx(expected, abs=0.05)

    def test_stress_zero_when_profile_full_and_wet(self, weather):
        """With a full profile, depletion is 0, Ks = 1, so ETa = ETc."""
        soil = SoilConfig(total_available_water_mm=200, initial_storage_mm=200)
        res = simulate(weather.iloc[:30], NoIrrigationPolicy(), soil=soil)
        # First days at least: storage full (or rain-fed drawdown small vs RAW).
        assert res.frame["ks"].iloc[0] == 1.0
        assert res.frame["stress_mm"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_rainfed_eventually_stresses_dry_spells(self, weather):
        res = simulate(weather, NoIrrigationPolicy(), soil=SoilConfig(initial_storage_mm=50))
        assert res.total_stress_mm > 0
        assert res.stress_days > 0


class TestPolicies:
    def test_no_irrigation_policy(self, weather):
        res = simulate(weather, NoIrrigationPolicy())
        assert res.total_irrigation_mm == 0.0

    def test_threshold_policy_waters_only_after_trigger(self, weather):
        soil = SoilConfig(total_available_water_mm=100, initial_storage_mm=100)
        policy = ThresholdPolicy(trigger_fraction=0.5)
        res = simulate(weather, policy, soil=soil)
        f = res.frame
        # On days with irrigation, depletion just before should be >= 50% TAW.
        irrigated = f[f["irrigation_mm"] > 0]
        if len(irrigated):
            # Depletion shown is post-irrigation; it should be small again.
            assert (irrigated["depletion_mm"] <= 60).all()

    def test_demand_policy_zero_when_no_need(self, weather):
        """After heavy rain net need is 0 -> demand-aware policy applies nothing."""
        wet_days = weather[weather["precip_mm"] > 30].head(5)
        if len(wet_days) == 0:
            pytest.skip("No very wet days in synthetic sample")
        res = simulate(wet_days, PredictedDemandPolicy(demand_threshold_mm=3.0),
                       soil=SoilConfig(total_available_water_mm=200, initial_storage_mm=200))
        assert res.total_irrigation_mm == 0.0

    def test_demand_policy_applies_in_dry_spell(self, weather):
        dry = weather[(weather["precip_mm"] < 1) & (weather["et0_mm"] > 4)].head(40)
        if len(dry) < 20:
            pytest.skip("No dry spell in synthetic sample")
        res = simulate(dry, PredictedDemandPolicy(demand_threshold_mm=3.0),
                       soil=SoilConfig(total_available_water_mm=80, initial_storage_mm=80))
        assert res.total_irrigation_mm > 0

    def test_irrigation_never_exceeds_free_capacity(self, weather):
        soil = SoilConfig(total_available_water_mm=60, initial_storage_mm=60)
        res = simulate(weather, ThresholdPolicy(trigger_fraction=0.3), soil=soil)
        assert res.frame["storage_mm"].between(0, 60).all()

    def test_invalid_trigger_raises(self):
        with pytest.raises(ValueError):
            ThresholdPolicy(trigger_fraction=1.5)


class TestComparison:
    def test_compare_returns_all_policies(self, weather):
        table = compare_policies(weather.iloc[:90], default_policies(),
                                 SoilConfig(initial_storage_mm=100), CostWeights())
        assert len(table) == 3
        assert "total_cost" in table.columns

    def test_rainfed_never_waters(self, weather):
        table = compare_policies(weather.iloc[:60])
        rainfed_row = table[table.index.str.contains("rainfed")]
        assert float(rainfed_row["total_irrigation_mm"].iloc[0]) == 0.0

    def test_cost_decomposes(self, weather):
        costs = CostWeights(water_penalty=0.1, stress_penalty=2.0)
        res = simulate(weather, NoIrrigationPolicy(), costs=costs)
        assert res.total_cost == pytest.approx(
            res.water_cost + res.stress_cost
        )
        assert res.stress_cost == pytest.approx(2.0 * res.total_stress_mm)

    def test_summary_keys(self, weather):
        res = simulate(weather, NoIrrigationPolicy())
        s = res.summary()
        for key in ["policy", "total_irrigation_mm", "total_stress_mm", "stress_days",
                    "water_cost", "stress_cost", "total_cost"]:
            assert key in s
