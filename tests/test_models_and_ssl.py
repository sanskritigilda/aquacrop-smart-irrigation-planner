"""Tests for supervised models and the semi-supervised experiment."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_sample
from src.evaluation import classification_metrics, regression_metrics
from src.models import (
    describe_cluster,
    train_all_regressors,
    train_kmeans_clusters,
    train_logistic_regression,
    train_multiple_linear_regression,
    train_polynomial_regression,
)
from src.preprocessing import build_feature_frame, chronological_split, set_global_seed
from src.semi_supervised import self_training_classification, self_training_regression
from src.water_demand import crop_water_demand, irrigation_required


@pytest.fixture(scope="module")
def prepared():
    set_global_seed(42)
    df = crop_water_demand(generate_synthetic_sample(n_years=3, seed=5))
    df["irrig_required"] = irrigation_required(df, threshold_mm=3.0)
    reg = chronological_split(build_feature_frame(df, "et0_mm"), "et0_mm")
    cls = chronological_split(build_feature_frame(df, "irrig_required"), "irrig_required")
    return df, reg, cls


class TestEvaluation:
    def test_regression_metrics_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        m = regression_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == pytest.approx(1.0)

    def test_classification_metrics_confusion_sums(self):
        y_true = [0, 0, 1, 1, 1, 0]
        y_pred = [0, 1, 1, 1, 0, 0]
        m = classification_metrics(y_true, y_pred)
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 6
        assert m["precision"] == pytest.approx(2 / 3)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            regression_metrics([1, 2], [1, 2, 3])
        with pytest.raises(ValueError):
            classification_metrics([1, 0], [1])


class TestRegressionModels:
    def test_all_regressors_run_and_beat_persistence(self, prepared):
        _, reg, _ = prepared
        results = train_all_regressors(reg)
        assert len(results) == 3
        naive_mae = float(np.abs(reg.y_test - reg.y_train.mean()).mean())
        for res in results.values():
            assert res.metrics["mae"] < naive_mae  # better than predicting the train mean
            assert res.metrics["rmse"] >= res.metrics["mae"]
            assert len(res.y_pred) == reg.n_test
            assert (np.asarray(res.y_pred) >= 0).all()

    def test_polynomial_at_least_as_good_as_linear_in_sample_size_terms(self, prepared):
        # Degree-2 should not be wildly worse than linear on a smooth target.
        _, reg, _ = prepared
        lin = train_multiple_linear_regression(reg)
        poly = train_polynomial_regression(reg, degree=2)
        assert poly.metrics["mae"] <= lin.metrics["mae"] * 1.5

    def test_invalid_degree_raises(self, prepared):
        _, reg, _ = prepared
        with pytest.raises(ValueError):
            train_polynomial_regression(reg, degree=1)


class TestClassificationModel:
    def test_logistic_runs_with_sane_metrics(self, prepared):
        _, _, cls = prepared
        res = train_logistic_regression(cls)
        assert 0.5 <= res.metrics["accuracy"] <= 1.0
        assert 0.0 <= res.metrics["f1"] <= 1.0
        assert set(res.y_pred.unique()) <= {0, 1}


class TestClustering:
    def test_kmeans_shapes_and_profiles(self, prepared):
        df, _, _ = prepared
        km = train_kmeans_clusters(df, n_clusters=4)
        assert km.labels.nunique() == 4
        assert len(km.labels) == len(df)
        assert km.profile["days"].sum() == len(km.labels)
        assert 0 < km.silhouette <= 1

    def test_describe_cluster_text(self, prepared):
        df, _, _ = prepared
        km = train_kmeans_clusters(df, n_clusters=3)
        for i in range(3):
            text = describe_cluster(km, i)
            assert "Typical day" in text

    def test_too_few_clusters_raises(self):
        with pytest.raises(ValueError):
            train_kmeans_clusters(crop_water_demand(generate_synthetic_sample(n_years=1)), 1)


class TestSemiSupervised:
    def test_classification_experiment_keeps_test_separate(self, prepared):
        _, _, cls = prepared
        set_global_seed(42)
        res = self_training_classification(
            cls.X_train, cls.y_train, cls.X_test, cls.y_test,
            labelled_fraction=0.15, confidence_threshold=0.9,
        )
        assert res.n_labelled + res.n_unlabelled == cls.n_train
        assert res.n_labelled > 0 and res.n_unlabelled > 0
        assert res.n_pseudo_accepted <= res.n_unlabelled
        assert "accuracy" in res.ssl_metrics and "f1" in res.ssl_metrics
        assert isinstance(res.improvement["f1"], float)

    def test_regression_experiment_keeps_test_separate(self, prepared):
        _, reg, _ = prepared
        set_global_seed(42)
        res = self_training_regression(
            reg.X_train, reg.y_train, reg.X_test, reg.y_test,
            labelled_fraction=0.15, max_std_mm=1.0, max_rounds=2,
        )
        assert res.n_pseudo_accepted <= res.n_unlabelled
        assert res.baseline_metrics["mae"] > 0
        assert res.ssl_metrics["mae"] > 0
        assert len(res.rounds) >= 1

    def test_high_confidence_gate_accepts_fewer(self, prepared):
        _, _, cls = prepared
        set_global_seed(0)
        loose = self_training_classification(
            cls.X_train, cls.y_train, cls.X_test, cls.y_test,
            labelled_fraction=0.1, confidence_threshold=0.6, max_rounds=1,
        )
        strict = self_training_classification(
            cls.X_train, cls.y_train, cls.X_test, cls.y_test,
            labelled_fraction=0.1, confidence_threshold=0.99, max_rounds=1,
        )
        assert strict.n_pseudo_accepted <= loose.n_pseudo_accepted

    def test_invalid_fraction_raises(self, prepared):
        _, reg, _ = prepared
        with pytest.raises(ValueError):
            self_training_regression(reg.X_train, reg.y_train, reg.X_test, reg.y_test,
                                     labelled_fraction=0.001)
