"""
Reference-value correctness tests.

These do NOT merely check that a function runs -- they check that the
answer is *numerically correct* against independently-computed reference
values: classic textbook examples with known results, hand-derived
formulas evaluated with plain Python/NumPy (a second, independent code
path), and, where available, an alternate library implementation.

This is "mathematical correctness" testing, as distinct from the
"does it crash" edge-case testing in test_edge_cases.py / test_fuzz.py.
"""

import math

import numpy as np
import pandas as pd
import pytest

from psyinsight.statistics.descriptive import DescriptiveStatistics
from psyinsight.statistics.inferential import InferentialStatistics
from psyinsight.ml.metrics import PsyMetrics


# ---------------------------------------------------------------------------
# Descriptive statistics: compare against plain-Python / NumPy re-derivation
# ---------------------------------------------------------------------------


class TestDescriptiveCorrectness:
    def setup_method(self):
        self.values = [4.0, 8.0, 6.0, 5.0, 3.0, 2.0, 8.0, 9.0, 2.0, 5.0]
        self.df = pd.DataFrame({"x": self.values})
        self.desc = DescriptiveStatistics(self.df)

    def test_mean_matches_hand_computation(self):
        expected = sum(self.values) / len(self.values)
        assert self.desc.mean()["x"] == pytest.approx(expected, rel=1e-9)

    def test_variance_matches_hand_computation(self):
        # Sample variance (ddof=1), the pandas/PsyInsight default.
        mean = sum(self.values) / len(self.values)
        expected = sum((v - mean) ** 2 for v in self.values) / (len(self.values) - 1)
        assert self.desc.variance()["x"] == pytest.approx(expected, rel=1e-9)

    def test_std_is_sqrt_of_variance(self):
        var = self.desc.variance()["x"]
        std = self.desc.standard_deviation()["x"]
        assert std == pytest.approx(math.sqrt(var), rel=1e-12)

    def test_median_matches_hand_computation(self):
        sorted_vals = sorted(self.values)
        n = len(sorted_vals)
        expected = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        assert self.desc.median()["x"] == pytest.approx(expected)

    def test_range_matches_max_minus_min(self):
        assert self.desc.data_range()["x"] == pytest.approx(max(self.values) - min(self.values))

    def test_iqr_matches_quartile_difference(self):
        q = self.desc.quartiles()
        expected = q.loc[0.75, "x"] - q.loc[0.25, "x"]
        assert self.desc.interquartile_range()["x"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Inferential statistics: classic textbook examples with known answers
# ---------------------------------------------------------------------------


class TestTTestCorrectness:
    def test_one_sample_t_test_known_result(self):
        # Textbook example: sample with a known, hand-computable t-statistic.
        sample = [51, 55, 45, 42, 53, 50, 47, 49, 52, 48]
        df = pd.DataFrame({"score": sample})
        result = InferentialStatistics(df).one_sample_t_test("score", population_mean=50)

        mean = np.mean(sample)
        sd = np.std(sample, ddof=1)
        n = len(sample)
        expected_t = (mean - 50) / (sd / np.sqrt(n))

        assert result["t_statistic"] == pytest.approx(expected_t, rel=1e-9)
        assert result["n"] == n
        assert result["mean"] == pytest.approx(mean)

    def test_independent_t_test_matches_manual_welch_formula(self):
        group_a = [23, 25, 21, 24, 22, 26, 20]
        group_b = [30, 28, 32, 29, 31, 27, 33]
        df = pd.DataFrame(
            {
                "score": group_a + group_b,
                "group": ["A"] * len(group_a) + ["B"] * len(group_b),
            }
        )
        result = InferentialStatistics(df).independent_t_test("score", "group", "A", "B", equal_var=False)

        a, b = np.array(group_a, dtype=float), np.array(group_b, dtype=float)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        n_a, n_b = len(a), len(b)
        expected_t = (np.mean(a) - np.mean(b)) / np.sqrt(var_a / n_a + var_b / n_b)

        assert result["t_statistic"] == pytest.approx(expected_t, rel=1e-9)

    def test_cohens_d_matches_hand_formula(self):
        group_a = [10, 12, 11, 13, 9]
        group_b = [15, 14, 16, 13, 17]
        df = pd.DataFrame(
            {
                "score": group_a + group_b,
                "group": ["A"] * len(group_a) + ["B"] * len(group_b),
            }
        )
        result = InferentialStatistics(df).independent_t_test("score", "group", "A", "B")

        a, b = np.array(group_a, dtype=float), np.array(group_b, dtype=float)
        n_a, n_b = len(a), len(b)
        pooled_sd = np.sqrt(
            ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
        )
        expected_d = (np.mean(a) - np.mean(b)) / pooled_sd

        assert result["effect_size"]["cohens_d"] == pytest.approx(expected_d, rel=1e-9)

    def test_paired_t_test_matches_manual_formula(self):
        before = [68, 70, 65, 72, 69, 71, 66]
        after = [65, 68, 63, 70, 66, 69, 64]
        df = pd.DataFrame({"before": before, "after": after})
        result = InferentialStatistics(df).paired_t_test("before", "after")

        diff = np.array(before, dtype=float) - np.array(after, dtype=float)
        expected_t = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))

        assert result["t_statistic"] == pytest.approx(expected_t, rel=1e-9)


class TestCorrelationCorrectness:
    def test_pearson_matches_numpy_corrcoef(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 1, 4, 3, 7, 6, 8, 9, 11, 10]
        df = pd.DataFrame({"x": x, "y": y})
        result = InferentialStatistics(df).pearson_correlation("x", "y")

        expected_r = np.corrcoef(x, y)[0, 1]
        assert result["correlation"] == pytest.approx(expected_r, rel=1e-9)
        assert result["effect_size"]["r_squared"] == pytest.approx(expected_r ** 2, rel=1e-9)

    def test_perfect_positive_correlation_is_one(self):
        x = list(range(1, 11))
        y = [v * 2 + 3 for v in x]
        df = pd.DataFrame({"x": x, "y": y})
        result = InferentialStatistics(df).pearson_correlation("x", "y")
        assert result["correlation"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_correlation_is_minus_one(self):
        x = list(range(1, 11))
        y = [-v for v in x]
        df = pd.DataFrame({"x": x, "y": y})
        result = InferentialStatistics(df).pearson_correlation("x", "y")
        assert result["correlation"] == pytest.approx(-1.0, abs=1e-9)

    def test_spearman_matches_pandas_rank_correlation(self):
        x = [10, 20, 30, 40, 50]
        y = [15, 10, 40, 35, 60]
        df = pd.DataFrame({"x": x, "y": y})
        result = InferentialStatistics(df).spearman_correlation("x", "y")

        expected = pd.Series(x).corr(pd.Series(y), method="spearman")
        assert result["correlation"] == pytest.approx(expected, rel=1e-9)


class TestChiSquareCorrectness:
    def test_chi_square_matches_hand_computed_contingency_table(self):
        # Classic 2x2 table with a known chi-square statistic.
        # Observed:      Yes   No
        #        Group A   10   20
        #        Group B   20   10
        rows = (
            ["A"] * 30 + ["B"] * 30
        )
        cols = (
            ["Yes"] * 10 + ["No"] * 20 + ["Yes"] * 20 + ["No"] * 10
        )
        df = pd.DataFrame({"group": rows, "response": cols})

        result = InferentialStatistics(df).chi_square_test("group", "response")

        contingency = pd.crosstab(df["group"], df["response"]).to_numpy()
        row_totals = contingency.sum(axis=1, keepdims=True)
        col_totals = contingency.sum(axis=0, keepdims=True)
        total = contingency.sum()
        expected_counts = row_totals @ col_totals / total
        # scipy applies the Yates continuity correction by default for 2x2
        # tables -- match that here rather than the uncorrected formula.
        expected_chi2 = np.sum(
            (np.abs(contingency - expected_counts) - 0.5) ** 2 / expected_counts
        )

        assert result["chi_square"] == pytest.approx(expected_chi2, rel=1e-9)
        assert result["n"] == 60


class TestAnovaCorrectness:
    def test_one_way_anova_matches_hand_computed_f_statistic(self):
        group_a = [4, 5, 6, 5, 4]
        group_b = [7, 8, 7, 9, 8]
        group_c = [2, 3, 2, 1, 3]
        df = pd.DataFrame(
            {
                "score": group_a + group_b + group_c,
                "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            }
        )
        result = InferentialStatistics(df).one_way_anova("score", "group")

        groups = [np.array(group_a), np.array(group_b), np.array(group_c)]
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
        df_between = len(groups) - 1
        df_within = sum(len(g) for g in groups) - len(groups)
        expected_f = (ss_between / df_between) / (ss_within / df_within)

        assert result["f_statistic"] == pytest.approx(expected_f, rel=1e-9)

        expected_eta_sq = ss_between / (ss_between + ss_within)
        assert result["effect_size"]["eta_squared"] == pytest.approx(expected_eta_sq, rel=1e-9)


class TestCronbachAlphaCorrectness:
    def test_cronbach_alpha_matches_hand_computed_value(self):
        # 4 items, 6 respondents -- small enough to hand-verify.
        data = pd.DataFrame(
            {
                "item1": [4, 3, 5, 2, 4, 3],
                "item2": [4, 2, 5, 2, 3, 3],
                "item3": [3, 3, 4, 1, 4, 2],
                "item4": [4, 3, 5, 2, 4, 4],
            }
        )
        alpha = PsyMetrics.cronbach_alpha(data)

        k = data.shape[1]
        item_variances = data.var(axis=0, ddof=1).sum()
        total_variance = data.sum(axis=1).var(ddof=1)
        expected_alpha = (k / (k - 1)) * (1 - item_variances / total_variance)

        assert alpha == pytest.approx(expected_alpha, rel=1e-9)
        # Sanity bound: Cronbach's alpha for positively-correlated items
        # with this k should sit in a plausible, non-degenerate range.
        assert -1.0 <= alpha <= 1.0

    def test_cronbach_alpha_of_identical_items_is_one(self):
        # Alpha = 1 requires tau-equivalence (equal variance, perfect
        # correlation) -- items that are merely perfectly *correlated* but
        # differently scaled (e.g. base, 2*base) are NOT tau-equivalent and
        # alpha will be < 1, which is mathematically correct, not a bug.
        base = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        data = pd.DataFrame({f"item{i}": base for i in range(4)})
        alpha = PsyMetrics.cronbach_alpha(data)
        assert alpha == pytest.approx(1.0, abs=1e-6)

    def test_cronbach_alpha_of_rescaled_perfectly_correlated_items_is_below_one(self):
        # Documents the tau-equivalence subtlety above: perfect correlation
        # with unequal variances yields alpha < 1, not alpha == 1.
        base = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        data = pd.DataFrame({f"item{i}": base * (i + 1) for i in range(4)})
        alpha = PsyMetrics.cronbach_alpha(data)
        assert alpha < 1.0
