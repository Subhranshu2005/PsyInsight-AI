"""
Stress / edge-case tests.

These intentionally probe degenerate inputs -- empty dataframes, single-row
datasets, all-NaN columns, extreme class imbalance, malformed files, and a
few real wiring bugs that were found and fixed during a robustness pass.
The goal is not to prove every internal is bulletproof (see clustering.py /
classifier.py's 1000+ line internals, which aren't fully audited), but to
lock in the specific failure modes that were found and fixed, and to give
early warning if a future change reintroduces them.
"""

import numpy as np
import pandas as pd
import pytest

from psyinsight.preprocessing.validator import DataValidator
from psyinsight.preprocessing.cleaner import DataCleaner
from psyinsight.preprocessing.loader import DataLoader
from psyinsight.ml.classifier import PsyClassifier
from psyinsight.ml.regressor import PsyRegressor
from psyinsight.ml.clustering import PsyClustering, InvalidClusterError
from psyinsight.statistics.descriptive import DescriptiveStatistics
from psyinsight.statistics.inferential import InferentialStatistics
from psyinsight.automata import ResponsePatternAnalyzer
from psyinsight.probability.distributions import ProbabilityEngine


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------


class TestDataValidatorEdgeCases:
    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            DataValidator([1, 2, 3])

    def test_empty_dataframe_no_rows_no_cols(self):
        v = DataValidator(pd.DataFrame())
        result = v.validate()
        assert result["Dataset Shape"] == (0, 0)
        assert v.is_empty() is True

    def test_empty_dataframe_with_columns_no_nan_percentage(self):
        """Zero rows with defined columns must not produce NaN/inf percentages."""
        df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=float)})
        v = DataValidator(df)
        pct = v.missing_value_percentage()
        assert (pct == 0.0).all()
        assert not pct.isna().any()

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"a": [1.0], "b": [None]})
        v = DataValidator(df)
        result = v.validate()
        assert result["Dataset Shape"] == (1, 2)
        assert result["Missing Value Percentage"]["b"] == 100.0

    def test_all_nan_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
        v = DataValidator(df)
        pct = v.missing_value_percentage()
        assert pct["a"] == 100.0
        assert pct["b"] == 0.0


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------


class TestDataCleanerEdgeCases:
    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            DataCleaner("not a dataframe")

    def test_rename_columns_with_integer_headers(self):
        """Integer column labels used to raise AttributeError on .str accessor."""
        df = pd.DataFrame([[1, 2], [3, 4]], columns=[0, 1])
        cleaner = DataCleaner(df)
        result = cleaner.rename_columns()
        assert list(result.columns) == ["0", "1"]

    def test_empty_dataframe_does_not_crash(self):
        cleaner = DataCleaner(pd.DataFrame())
        assert cleaner.rename_columns().empty


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class TestDataLoaderEdgeCases:
    def test_missing_file_raises_clear_error(self):
        with pytest.raises(FileNotFoundError):
            DataLoader.load_csv("this_file_does_not_exist_12345.csv")

    def test_empty_csv_raises_clear_error(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(ValueError):
            DataLoader.load_csv(str(path))

    def test_oversized_file_is_rejected(self, tmp_path, monkeypatch):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n")
        monkeypatch.setattr(DataLoader, "MAX_FILE_SIZE_MB", 0)  # force rejection
        with pytest.raises(ValueError, match="exceeds"):
            DataLoader.load_csv(str(path))

    def test_normal_csv_still_loads(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n3,4\n")
        df = DataLoader.load_csv(str(path))
        assert df.shape == (2, 2)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifierEdgeCases:
    def test_single_class_raises_friendly_error(self):
        """A raw sklearn ValueError used to bubble up uncaught here."""
        X = pd.DataFrame({"f1": range(10), "f2": range(10, 20)})
        y = pd.Series([1] * 10)
        clf = PsyClassifier()
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.3)
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            clf.auto_train(X_train, X_test, y_train, y_test)

    def test_auto_train_returns_app_compatible_schema(self):
        """auto_train() must expose best_model/leaderboard/test_report --
        this is the schema the Streamlit UI (and PsyRegressor) rely on."""
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.randn(120), "f2": rng.randn(120)})
        y = pd.Series(rng.randint(0, 2, 120))
        clf = PsyClassifier()
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.2)
        result = clf.auto_train(X_train, X_test, y_train, y_test)

        for key in ("best_model", "leaderboard", "test_report", "results", "best_key", "best_row"):
            assert key in result
        assert isinstance(result["leaderboard"], pd.DataFrame)
        assert isinstance(result["test_report"], dict)

    def test_all_candidates_failing_raises_clear_error(self):
        """Regression test: used to raise a bare KeyError('r2') when every
        candidate model failed to fit."""
        X_train = pd.DataFrame({"f1": [1.0]})
        y_train = pd.Series([np.nan])
        clf = PsyClassifier()
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            clf.compare_models(X_train, X_train, y_train, y_train)

    def test_one_bad_candidate_does_not_crash_whole_comparison(self):
        """Regression test: KNN needing more neighbors than available
        samples used to crash the entire compare_models()/auto_train()
        call instead of being skipped like PsyRegressor does."""
        X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 3.0, 2.0, 1.0]})
        y_train = pd.Series([0, 1, 0, 1])
        X_test = pd.DataFrame({"f1": [1.5, 3.5], "f2": [3.5, 1.5]})
        y_test = pd.Series([0, 1])
        clf = PsyClassifier()
        result = clf.auto_train(X_train, X_test, y_train, y_test)
        assert "best_model" in result
        # knn (n_neighbors=5 > 4 training samples) should have been skipped,
        # not crashed the whole run.
        assert not result["leaderboard"]["Model"].str.contains("KNN|K Nearest", case=False).any()


class TestRegressorAllCandidatesFail:
    def test_all_candidates_failing_raises_clear_error(self):
        """Regression test: used to raise a bare KeyError('r2') when every
        candidate model failed to fit (e.g. NaN target)."""
        X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        y_train = pd.Series([np.nan, np.nan, np.nan])
        X_test = pd.DataFrame({"f1": [4.0]})
        y_test = pd.Series([4.0])
        reg = PsyRegressor()
        with pytest.raises(ValueError, match="Every candidate model failed"):
            reg.auto_train(X_train, X_test, y_train, y_test)


class TestClassifierImbalance:
    def test_extreme_class_imbalance_does_not_crash(self):
        rng = np.random.RandomState(1)
        n = 200
        X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
        y = pd.Series([0] * (n - 3) + [1, 1, 1])  # 98.5% / 1.5% split
        clf = PsyClassifier()
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.2)
        # As long as both splits still contain >= 2 classes, this should train
        # without raising, however poor the resulting metrics are.
        if len(set(y_train)) >= 2 and len(set(y_test)) >= 2:
            result = clf.auto_train(X_train, X_test, y_train, y_test)
            assert "best_model" in result


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


class TestRegressorEdgeCases:
    def test_tiny_dataset_still_trains(self):
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y = pd.Series([1.0, 2.0])
        reg = PsyRegressor()
        X_train, X_test, y_train, y_test = reg.split_data(X, y, test_size=0.5)
        result = reg.auto_train(X_train, X_test, y_train, y_test)
        assert "best_model" in result

    def test_constant_target_does_not_crash(self):
        rng = np.random.RandomState(2)
        X = pd.DataFrame({"f1": rng.randn(50), "f2": rng.randn(50)})
        y = pd.Series([5.0] * 50)  # zero variance target
        reg = PsyRegressor()
        X_train, X_test, y_train, y_test = reg.split_data(X, y, test_size=0.2)
        result = reg.auto_train(X_train, X_test, y_train, y_test)
        assert "best_model" in result


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestClusteringEdgeCases:
    def test_more_clusters_than_samples_raises_friendly_error(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        c = PsyClustering()
        c.set_model("kmeans", n_clusters=5)
        with pytest.raises(InvalidClusterError, match="only 3 sample"):
            c.fit_predict(X)

    def test_single_sample_raises_friendly_error(self):
        X = pd.DataFrame({"a": [1], "b": [1]})
        c = PsyClustering()
        c.set_model("kmeans", n_clusters=2)
        with pytest.raises(InvalidClusterError):
            c.fit_predict(X)

    def test_normal_case_still_works(self):
        X = pd.DataFrame({"a": [1, 2, 3, 10, 11, 12], "b": [1, 2, 3, 10, 11, 12]})
        c = PsyClustering()
        c.set_model("kmeans", n_clusters=2)
        labels = c.fit_predict(X)
        assert len(set(labels)) == 2

    def test_benchmark_does_not_clobber_active_model(self):
        """Regression test: benchmark() (a documented read-only timing
        diagnostic) used to leave whichever algorithm it timed last as the
        permanently active model, silently discarding the user's actual
        fitted model."""
        rng = np.random.RandomState(0)
        X = pd.DataFrame(
            np.vstack([rng.randn(30, 3) + 5, rng.randn(30, 3) - 5]), columns=["a", "b", "c"]
        )
        c = PsyClustering()
        c.set_model("kmeans", n_clusters=2)
        c.fit(X)
        labels_before = c.cluster_labels().copy()
        model_key_before = c.model_key

        c.benchmark(X)

        assert c.model_key == model_key_before
        assert np.array_equal(c.cluster_labels(), labels_before)

    def test_contingency_handles_cluster_count_not_equal_to_class_count(self):
        """Regression test: contingency() used sklearn's confusion_matrix,
        which is square over the union of label sets and raised a
        DataFrame-shape ValueError whenever cluster count != class count
        (the normal case in clustering)."""
        rng = np.random.RandomState(0)
        X = pd.DataFrame(
            np.vstack([rng.randn(20, 2) + 5, rng.randn(20, 2) - 5, rng.randn(20, 2)]),
            columns=["a", "b"],
        )
        y_true = np.array([0] * 20 + [1] * 20 + [0] * 20)  # 2 true classes
        c = PsyClustering()
        c.set_model("kmeans", n_clusters=3)  # 3 clusters != 2 classes
        c.fit(X)
        table = c.contingency(y_true)
        assert table.shape == (2, 3)


# ---------------------------------------------------------------------------
# Descriptive / inferential statistics
# ---------------------------------------------------------------------------


class TestStatisticsEdgeCases:
    def test_descriptive_stats_on_empty_dataframe(self):
        stats = DescriptiveStatistics(pd.DataFrame({"a": pd.Series(dtype=float)}))
        summary = stats.summary()
        assert summary.loc["mean", "a"] != summary.loc["mean", "a"]  # NaN, no crash
        assert summary.loc["count", "a"] == 0.0

    def test_descriptive_stats_all_nan_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        stats = DescriptiveStatistics(df)
        summary = stats.summary()
        # Should not raise; mean/std of an all-NaN column is NaN, not a crash.
        mean_val = summary.loc["mean", "a"]
        assert mean_val != mean_val  # NaN != NaN

    def test_independent_t_test_matches_documented_signature(self):
        """Regression test for the app-wiring bug: independent_t_test takes
        (column, group_column, group1, group2), not (col_a, col_b)."""
        df = pd.DataFrame(
            {
                "score": [1, 2, 3, 4, 5, 6, 7, 8],
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            }
        )
        infer = InferentialStatistics(df)
        result = infer.independent_t_test("score", "group", "A", "B")
        assert "t_statistic" in result or "statistic" in result or "p_value" in result


class TestInferentialStatisticsHostileData:
    """Round 3: hostile/degenerate inputs to every InferentialStatistics
    method should raise a clear ValueError, never an unhandled crash."""

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            InferentialStatistics([1, 2, 3])

    def test_one_sample_t_test_empty_column(self):
        df = pd.DataFrame({"x": pd.Series(dtype=float)})
        with pytest.raises(ValueError):
            InferentialStatistics(df).one_sample_t_test("x", 0)

    def test_one_sample_t_test_single_value(self):
        df = pd.DataFrame({"x": [5.0]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).one_sample_t_test("x", 0)

    def test_one_sample_t_test_all_nan(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).one_sample_t_test("x", 0)

    def test_one_sample_t_test_missing_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).one_sample_t_test("does_not_exist", 0)

    def test_one_sample_t_test_non_numeric_column(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).one_sample_t_test("x", 0)

    def test_independent_t_test_one_group_missing(self):
        df = pd.DataFrame({"score": [1, 2, 3], "group": ["A", "A", "A"]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).independent_t_test("score", "group", "A", "B")

    def test_independent_t_test_group_with_single_value(self):
        df = pd.DataFrame(
            {"score": [1, 2, 3, 9], "group": ["A", "A", "A", "B"]}
        )
        with pytest.raises(ValueError):
            InferentialStatistics(df).independent_t_test("score", "group", "A", "B")

    def test_independent_t_test_infinite_values_do_not_crash(self):
        df = pd.DataFrame(
            {
                "score": [1, 2, 3, np.inf, 5, 6],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )
        # Should either raise a clear error or return a (non-crashing)
        # result; scipy propagates inf/nan rather than crashing.
        try:
            result = InferentialStatistics(df).independent_t_test("score", "group", "A", "B")
            assert "t_statistic" in result
        except ValueError:
            pass

    def test_paired_t_test_mismatched_lengths_after_nan_drop(self):
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).paired_t_test("a", "b")

    def test_pearson_correlation_constant_column(self):
        df = pd.DataFrame({"x": [5, 5, 5, 5], "y": [1, 2, 3, 4]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).pearson_correlation("x", "y")

    def test_pearson_correlation_empty_after_dropna(self):
        df = pd.DataFrame({"x": [np.nan, np.nan], "y": [np.nan, np.nan]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).pearson_correlation("x", "y")

    def test_chi_square_single_category_column(self):
        df = pd.DataFrame({"a": ["yes"] * 5, "b": ["x", "y", "x", "y", "x"]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).chi_square_test("a", "b")

    def test_chi_square_sparse_table_warns_about_low_expected_counts(self):
        df = pd.DataFrame(
            {
                "a": ["x"] * 9 + ["y"],
                "b": ["p"] * 5 + ["q"] * 5,
            }
        )
        result = InferentialStatistics(df).chi_square_test("a", "b")
        assert result["warnings"]  # low expected counts should be flagged

    def test_anova_single_group(self):
        df = pd.DataFrame({"score": [1, 2, 3], "group": ["A", "A", "A"]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).one_way_anova("score", "group")

    def test_anova_one_group_has_single_observation(self):
        df = pd.DataFrame(
            {"score": [1, 2, 3, 9], "group": ["A", "A", "A", "B"]}
        )
        with pytest.raises(ValueError):
            InferentialStatistics(df).one_way_anova("score", "group")

    def test_anova_missing_group_column(self):
        df = pd.DataFrame({"score": [1, 2, 3, 4]})
        with pytest.raises(ValueError):
            InferentialStatistics(df).one_way_anova("score", "does_not_exist")

    def test_independent_t_test_reports_assumption_warnings_on_unequal_variance(self):
        rng = np.random.default_rng(42)
        group_a = rng.normal(0, 1, 60)
        group_b = rng.normal(0, 20, 60)
        df = pd.DataFrame(
            {
                "score": np.concatenate([group_a, group_b]),
                "group": ["A"] * 60 + ["B"] * 60,
            }
        )
        result = InferentialStatistics(df).independent_t_test("score", "group", "A", "B")
        assert result["assumptions"]["homogeneity_of_variance"]["ran"]
        assert not result["assumptions"]["homogeneity_of_variance"]["equal_variance"]
        assert any("variance" in w.lower() for w in result["warnings"])


# ---------------------------------------------------------------------------
# Automata / response pattern analysis
# ---------------------------------------------------------------------------


class TestAutomataEdgeCases:
    def test_straight_lining_on_empty_dataframe(self):
        analyzer = ResponsePatternAnalyzer(pd.DataFrame())
        result = analyzer.detect_straight_lining([], run_limit=3)
        assert result.empty
        assert list(result.columns) == ["flagged_straight_lining", "longest_run_state"]

    def test_straight_lining_with_no_columns_selected(self):
        df = pd.DataFrame({"q1": [1, 2, 3]})
        analyzer = ResponsePatternAnalyzer(df)
        result = analyzer.detect_straight_lining([], run_limit=3)
        assert len(result) == 3

    def test_straight_lining_normal_case_still_works(self):
        # A respondent giving the same answer across many consecutive items
        # (within one row) is what "straight-lining" actually detects.
        df = pd.DataFrame(
            {
                "q1": [1, 1],
                "q2": [1, 2],
                "q3": [1, 1],
                "q4": [1, 2],
                "q5": [1, 1],
            }
        )
        analyzer = ResponsePatternAnalyzer(df)
        result = analyzer.detect_straight_lining(["q1", "q2", "q3", "q4", "q5"], run_limit=3)
        assert result.loc[0, "flagged_straight_lining"]  # row 0 is all 1's
        assert not result.loc[1, "flagged_straight_lining"]  # row 1 alternates


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


class TestProbabilityEdgeCases:
    def test_zero_std_normal_raises_clear_error(self):
        with pytest.raises(ValueError, match="Standard deviation"):
            ProbabilityEngine.normal_probability(0, 0, 0, mode="cdf")

    def test_zero_lambda_poisson_raises_clear_error(self):
        with pytest.raises(ValueError, match="Lambda"):
            ProbabilityEngine.poisson_probability(0, 0.0)

    def test_binomial_boundary_probabilities(self):
        assert ProbabilityEngine.binomial_probability(0, 10, 0.0) == 1.0
        assert ProbabilityEngine.binomial_probability(10, 10, 1.0) == 1.0
        assert ProbabilityEngine.binomial_probability(15, 10, 0.5) == 0.0
