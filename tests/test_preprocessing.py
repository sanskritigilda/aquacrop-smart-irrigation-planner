"""Tests for data preparation, feature tables and chronological splitting."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    DataError,
    generate_synthetic_sample,
    missing_value_report,
    validate_date_range,
)
from src.preprocessing import (
    FEATURE_COLUMNS,
    add_lagged_features,
    build_feature_frame,
    chronological_split,
    scale_features,
)
from src.water_demand import crop_water_demand


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    return crop_water_demand(generate_synthetic_sample(n_years=2, seed=7))


class TestDataLoader:
    def test_synthetic_sample_deterministic(self):
        a = generate_synthetic_sample(n_years=1, seed=3)
        b = generate_synthetic_sample(n_years=1, seed=3)
        pd.testing.assert_frame_equal(a, b)

    def test_synthetic_values_physically_bounded(self):
        df = generate_synthetic_sample(n_years=1, seed=1)
        assert (df["tmax_c"] >= df["tmin_c"]).all()
        assert df["rh_pct"].between(0, 100).all()
        assert (df["precip_mm"] >= 0).all()
        assert (df["solar_mj"] >= 0).all()

    def test_date_range_validation(self):
        assert validate_date_range("2022-01-01", "2022-01-31")[1].day == 31
        with pytest.raises(DataError):
            validate_date_range("2022-02-01", "2022-01-01")
        with pytest.raises(DataError):
            validate_date_range("not-a-date", "2022-01-01")
        with pytest.raises(DataError):
            validate_date_range("2010-01-01", "2025-01-01")  # exceeds cap

    def test_missing_report_on_clean_frame(self):
        df = generate_synthetic_sample(n_years=1)
        rep = missing_value_report(df)
        assert (rep["missing"] == 0).all()


class TestFeatureFrame:
    def test_selects_features_and_target(self, enriched):
        frame = build_feature_frame(enriched, "et0_mm")
        assert list(frame.columns) == FEATURE_COLUMNS + ["et0_mm"]
        assert not frame.isna().any().any()

    def test_missing_column_raises(self):
        bad = generate_synthetic_sample(n_years=1).drop(columns=["wind_ms"])
        with pytest.raises(ValueError):
            build_feature_frame(bad, "et0_mm")

    def test_too_few_rows_raises(self):
        tiny = generate_synthetic_sample(n_years=1).iloc[:10]
        tiny["et0_mm"] = 3.0
        with pytest.raises(ValueError):
            build_feature_frame(tiny, "et0_mm")


class TestChronologicalSplit:
    def test_train_precedes_test(self, enriched):
        frame = build_feature_frame(enriched, "et0_mm")
        split = chronological_split(frame, "et0_mm", train_fraction=0.8)
        assert split.X_train.index.max() < split.X_test.index.min()
        assert split.n_train + split.n_test == len(frame)
        assert split.n_train / len(frame) == pytest.approx(0.8, abs=0.01)

    def test_no_target_leakage_into_features(self, enriched):
        frame = build_feature_frame(enriched, "et0_mm")
        split = chronological_split(frame, "et0_mm")
        assert "et0_mm" not in split.X_train.columns
        assert "et0_mm" not in split.X_test.columns

    def test_empty_side_raises(self):
        idx = pd.date_range("2024-01-01", periods=10)
        frame = pd.DataFrame({"a": range(10), "t": range(10)}, index=idx)
        with pytest.raises(ValueError):
            chronological_split(frame, "t", train_fraction=0.05)


class TestScaling:
    def test_scaler_fit_on_train_only(self, enriched):
        frame = build_feature_frame(enriched, "et0_mm")
        split = chronological_split(frame, "et0_mm")
        Xtr, Xte, scaler = scale_features(split.X_train, split.X_test)
        assert Xtr.mean().abs().max() == pytest.approx(0.0, abs=1e-10)
        # Test set is NOT forced to zero mean — transform uses train statistics.
        assert abs(Xte.mean().mean()) > 0


class TestLaggedFeatures:
    def test_lags_use_past_only(self, enriched):
        with_lags = add_lagged_features(enriched[FEATURE_COLUMNS], lags=(1,))
        assert "tmax_c_lag1" in with_lags.columns
        first_valid = with_lags.index[0]
        assert with_lags.loc[first_valid, "tmax_c_lag1"] == pytest.approx(
            enriched["tmax_c"].iloc[0]
        )
        # No NaNs remain after dropna
        assert not with_lags.isna().any().any()

    def test_negative_lag_rejected(self, enriched):
        with pytest.raises(ValueError):
            add_lagged_features(enriched[FEATURE_COLUMNS], lags=(-1,))
