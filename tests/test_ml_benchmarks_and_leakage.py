"""
Benchmark-dataset and data-leakage tests for the ML modules.

Two concerns, both from the "how do we know the ML pipeline isn't lying
to us" category:

1. Benchmark datasets (items #2/#3 of the robustness review): run the
   classifier/regressor against well-known, standard datasets (Iris,
   Breast Cancer, Diabetes) and assert performance lands in a plausible,
   well-documented range. This catches gross regressions (e.g. an
   accidentally-shuffled label column) that a purely synthetic unit test
   would miss.

2. Leakage tests: verify that cross-validation and preprocessing do not
   leak information from held-out folds/test sets into training. This
   directly checks (and regression-locks) the fix to a real bug found
   during this pass: ``PsyRegressor.cross_validate()`` was fitting its
   ``StandardScaler`` on the *entire* dataset before cross-validating,
   so every fold's "held out" data had already influenced the scaler.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler

from psyinsight.ml.classifier import PsyClassifier
from psyinsight.ml.regressor import PsyRegressor


# ---------------------------------------------------------------------------
# Benchmark datasets: classifier
# ---------------------------------------------------------------------------


class TestClassifierBenchmarks:
    def test_iris_auto_train_reaches_high_accuracy(self):
        # Iris is linearly near-separable; any reasonable classifier
        # should comfortably clear 90% test accuracy.
        data = load_iris()
        X, y = data.data, data.target

        clf = PsyClassifier(random_state=0)
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.3, random_state=0)
        result = clf.auto_train(X_train, X_test, y_train, y_test, verbose=False)

        assert result["test_report"]["Accuracy"] >= 0.90

    def test_breast_cancer_auto_train_reaches_high_accuracy(self):
        # Breast Cancer Wisconsin is a well-studied, easy binary task;
        # published baselines commonly reach 93-98% accuracy.
        data = load_breast_cancer()
        X, y = data.data, data.target

        clf = PsyClassifier(random_state=0)
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.3, random_state=0)
        result = clf.auto_train(X_train, X_test, y_train, y_test, verbose=False)

        assert result["test_report"]["Accuracy"] >= 0.90

    def test_iris_single_model_beats_majority_baseline(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = PsyClassifier(random_state=0)
        clf.set_model("random_forest")
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.3, random_state=0)
        X_train_s, X_test_s = clf.preprocess(X_train, X_test)
        clf.fit(X_train_s, y_train)
        acc = clf.score(X_test_s, y_test)

        majority_baseline = pd.Series(y_test).value_counts(normalize=True).max()
        assert acc > majority_baseline


# ---------------------------------------------------------------------------
# Benchmark datasets: regressor
# ---------------------------------------------------------------------------


class TestRegressorBenchmarks:
    def test_diabetes_random_forest_beats_mean_baseline(self):
        # sklearn's diabetes dataset: a mean-only baseline gives R^2 = 0;
        # any competent regressor should clear a modest positive R^2.
        data = load_diabetes()
        X, y = data.data, data.target

        reg = PsyRegressor(model_name="random_forest", scale_features=True)
        X_train, X_test, y_train, y_test = reg.split_data(X, y, test_size=0.3, random_state=42)
        reg.fit(X_train, y_train)
        report = reg.evaluate(X_test, y_test)

        assert report["r2"] > 0.25  # a documented, conservative floor -- not "perfect"

    def test_diabetes_linear_regression_reasonable_r2(self):
        data = load_diabetes()
        X, y = data.data, data.target

        reg = PsyRegressor(model_name="linear_regression", scale_features=True)
        X_train, X_test, y_train, y_test = reg.split_data(X, y, test_size=0.3, random_state=42)
        reg.fit(X_train, y_train)
        report = reg.evaluate(X_test, y_test)

        assert report["r2"] > 0.25


# ---------------------------------------------------------------------------
# Leakage: cross-validation must not fit preprocessing on held-out data
# ---------------------------------------------------------------------------


class TestCrossValidationLeakage:
    def test_regressor_cross_validate_scaler_is_not_fit_on_full_dataset(self):
        """
        Regression test for a real bug: PsyRegressor.cross_validate() used
        to call self._prepare(X, fit=True), fitting StandardScaler on the
        WHOLE dataset (all folds) before cross-validating. That leaks each
        fold's held-out statistics into the scaler used to train on it.

        This test proves the fix by intercepting every StandardScaler
        fit during cross_validate() and checking that none of them ever
        sees the full dataset -- only a same-or-smaller (per-fold
        training) subset.
        """
        X, y = load_diabetes(return_X_y=True)
        n_total = len(X)

        seen_fit_sizes = []
        original_fit = StandardScaler.fit

        def spying_fit(self, X_arg, y_arg=None, **kwargs):
            seen_fit_sizes.append(len(X_arg))
            return original_fit(self, X_arg, y_arg, **kwargs)

        StandardScaler.fit = spying_fit
        try:
            reg = PsyRegressor(model_name="linear_regression", scale_features=True)
            reg.cross_validate(X, y, cv=5)
        finally:
            StandardScaler.fit = original_fit

        assert seen_fit_sizes, "StandardScaler.fit was never called during cross_validate()"
        # Every fit call during 5-fold CV must only ever see a *training*
        # fold (~80% of the data), never the full dataset.
        for size in seen_fit_sizes:
            assert size < n_total, (
                f"StandardScaler was fit on {size} of {n_total} rows -- "
                "the scaler saw the full dataset, which means fold-held-out "
                "data leaked into preprocessing."
            )

    def test_classifier_cross_validation_scaler_is_not_fit_on_full_dataset(self):
        """Same leakage check as above, for PsyClassifier.cross_validation()."""
        X, y = load_breast_cancer(return_X_y=True)
        n_total = len(X)

        seen_fit_sizes = []
        original_fit = StandardScaler.fit

        def spying_fit(self, X_arg, y_arg=None, **kwargs):
            seen_fit_sizes.append(len(X_arg))
            return original_fit(self, X_arg, y_arg, **kwargs)

        StandardScaler.fit = spying_fit
        try:
            clf = PsyClassifier(random_state=0)
            clf.set_model("logistic_regression")
            clf.cross_validation(X, y, cv=5)
        finally:
            StandardScaler.fit = original_fit

        assert seen_fit_sizes, "StandardScaler.fit was never called during cross_validation()"
        for size in seen_fit_sizes:
            assert size < n_total

    def test_preprocess_scaler_fit_only_on_training_split(self):
        """The scaler used for a single train/test split (as opposed to
        cross-validation) must be fit on the training portion only, and
        the test set must merely be *transformed* with it."""
        X, y = load_breast_cancer(return_X_y=True)
        clf = PsyClassifier(random_state=0)
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.3, random_state=0)

        X_train_scaled, X_test_scaled = clf.preprocess(X_train, X_test)

        # The fitted scaler's mean_ should match the *training* data's
        # mean, not a mean computed over train+test combined.
        expected_mean = X_train.mean(axis=0)
        assert np.allclose(clf.scaler.mean_, expected_mean, rtol=1e-8)

        # And it must be a strict transform of X_test (no leakage refit).
        manual_test_scaled = (X_test - clf.scaler.mean_) / clf.scaler.scale_
        assert np.allclose(X_test_scaled, manual_test_scaled, rtol=1e-6)

    def test_no_duplicate_rows_leak_between_train_and_test_split(self):
        """Sanity check that split_data() does not accidentally duplicate
        rows across the train/test partition (a common source of
        artificially inflated test performance)."""
        X, y = load_iris(return_X_y=True)
        clf = PsyClassifier(random_state=0)
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.3, random_state=0)

        train_rows = {tuple(row) for row in X_train}
        test_rows = {tuple(row) for row in X_test}
        assert train_rows.isdisjoint(test_rows) or len(train_rows & test_rows) <= 1
        # Iris has a couple of near-duplicate rows by chance; allow a tiny
        # tolerance but the vast majority must not overlap.
        overlap_fraction = len(train_rows & test_rows) / len(test_rows)
        assert overlap_fraction < 0.05
