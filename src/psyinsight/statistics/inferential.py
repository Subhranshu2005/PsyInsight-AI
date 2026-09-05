"""
PsyInsight AI
Inferential Statistics Module

Author: Subhranshu Ranjan Sahoo

This module provides inferential statistical methods.

Round 3 robustness pass
------------------------
Every test below now returns, alongside the statistic and p-value:

  * ``n`` (sample size(s)) so results can never be read without knowing
    how much data they're based on.
  * an **effect size** (Cohen's d, eta-squared, or Cramer's V) --
    p-values alone don't tell you whether an effect is practically
    meaningful.
  * a **confidence interval** where one has a standard closed-form
    (t-tests, correlations).
  * an ``assumptions`` dict recording whether normality /
    homogeneity-of-variance checks were run, whether they passed, and
    a human-readable ``warnings`` list -- so a researcher is told,
    e.g., "Levene's test suggests unequal variances" instead of
    silently getting a result that assumes otherwise.

Every method also now validates its inputs up front (empty samples,
too few observations, non-numeric columns, constant/zero-variance
data) and raises a clear ``ValueError`` naming the problem, rather
than letting scipy raise a cryptic exception or -- worse -- silently
return ``nan``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    pearsonr,
    spearmanr,
    chi2_contingency,
    f_oneway,
    shapiro,
    levene,
    t as t_dist,
)

__all__ = ["InferentialStatistics"]

# Shapiro-Wilk is only well-defined/reliable in this range; outside it we
# skip the check rather than report a misleading result.
_SHAPIRO_MIN_N = 3
_SHAPIRO_MAX_N = 5000


def _as_numeric_series(series: pd.Series, label: str) -> pd.Series:
    """Coerce a column to numeric, dropping NaNs, with a clear error on
    failure instead of a downstream scipy exception."""
    numeric = pd.to_numeric(series, errors="coerce")
    n_before = series.notna().sum()
    numeric = numeric.dropna()
    if n_before and len(numeric) == 0:
        raise ValueError(
            f"Column '{label}' contains no numeric values after coercion."
        )
    return numeric


def _check_min_n(n: int, minimum: int, what: str) -> None:
    if n < minimum:
        raise ValueError(
            f"{what} needs at least {minimum} observation(s); got {n}."
        )


def _normality_check(sample: np.ndarray, label: str) -> Dict[str, Any]:
    """Shapiro-Wilk normality check. Returns a dict describing whether the
    check ran, its result, and (if relevant) a warning string."""
    n = len(sample)
    if n < _SHAPIRO_MIN_N:
        return {
            "ran": False,
            "reason": f"n={n} too small for a normality check (need >= {_SHAPIRO_MIN_N}).",
        }
    if n > _SHAPIRO_MAX_N:
        return {
            "ran": False,
            "reason": f"n={n} exceeds Shapiro-Wilk's recommended range (<= {_SHAPIRO_MAX_N}); skipped.",
        }
    if np.all(sample == sample[0]):
        return {
            "ran": False,
            "reason": f"'{label}' is constant (zero variance); normality is undefined.",
        }
    try:
        stat, p_value = shapiro(sample)
    except Exception as exc:  # noqa: BLE001 -- never let this crash the caller
        return {"ran": False, "reason": f"Shapiro-Wilk failed to run: {exc}"}

    return {
        "ran": True,
        "test": "Shapiro-Wilk",
        "statistic": float(stat),
        "p_value": float(p_value),
        "normal": bool(p_value >= 0.05),
    }


def _variance_homogeneity_check(*samples: np.ndarray) -> Dict[str, Any]:
    """Levene's test for equality of variances across 2+ groups."""
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2:
        return {"ran": False, "reason": "Need at least two non-empty groups."}
    if any(len(s) < 2 for s in samples):
        return {"ran": False, "reason": "Every group needs at least 2 observations."}
    try:
        stat, p_value = levene(*samples)
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "reason": f"Levene's test failed to run: {exc}"}

    return {
        "ran": True,
        "test": "Levene",
        "statistic": float(stat),
        "p_value": float(p_value),
        "equal_variance": bool(p_value >= 0.05),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_var = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d for a paired/one-sample design (mean difference / SD of differences)."""
    if len(diff) < 2:
        return float("nan")
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def _mean_ci(sample: np.ndarray, confidence: float = 0.95) -> Sequence[float]:
    n = len(sample)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = np.mean(sample)
    sem = np.std(sample, ddof=1) / np.sqrt(n)
    if sem == 0:
        return (float(mean), float(mean))
    t_crit = t_dist.ppf(1 - (1 - confidence) / 2, df=n - 1)
    return (float(mean - t_crit * sem), float(mean + t_crit * sem))


def _diff_ci(a: np.ndarray, b: np.ndarray, equal_var: bool, confidence: float = 0.95) -> Sequence[float]:
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return (float("nan"), float("nan"))

    diff = np.mean(a) - np.mean(b)
    if equal_var:
        pooled_var = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
        se = np.sqrt(pooled_var * (1 / n_a + 1 / n_b))
        df = n_a + n_b - 2
    else:
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        se = np.sqrt(var_a / n_a + var_b / n_b)
        # Welch-Satterthwaite degrees of freedom.
        denom = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
        df = (se ** 4) / denom if denom > 0 else (n_a + n_b - 2)

    if se == 0:
        return (float(diff), float(diff))
    t_crit = t_dist.ppf(1 - (1 - confidence) / 2, df=df)
    return (float(diff - t_crit * se), float(diff + t_crit * se))


def _correlation_ci(r: float, n: int, confidence: float = 0.95) -> Sequence[float]:
    """Fisher z-transform confidence interval for a correlation coefficient."""
    if n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    from scipy.stats import norm

    z_crit = norm.ppf(1 - (1 - confidence) / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def _interpret_d(d: float) -> str:
    if np.isnan(d):
        return "undefined"
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def _interpret_eta_sq(eta_sq: float) -> str:
    if np.isnan(eta_sq):
        return "undefined"
    if eta_sq < 0.01:
        return "negligible"
    if eta_sq < 0.06:
        return "small"
    if eta_sq < 0.14:
        return "medium"
    return "large"


def _interpret_cramers_v(v: float, df_min: int) -> str:
    """Cohen's (1988) benchmarks for Cramer's V, adjusted for the smaller
    of (rows-1, cols-1) -- the conventional adjustment for tables larger
    than 2x2."""
    if np.isnan(v):
        return "undefined"
    thresholds = {1: (0.1, 0.3, 0.5), 2: (0.07, 0.21, 0.35), 3: (0.06, 0.17, 0.29)}
    small, medium, large = thresholds.get(df_min, thresholds[3])
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


def _interpret_correlation(r: float) -> str:
    magnitude = abs(r)
    if magnitude < 0.1:
        return "negligible"
    if magnitude < 0.3:
        return "small"
    if magnitude < 0.5:
        return "moderate"
    if magnitude < 0.7:
        return "strong"
    return "very strong"


class InferentialStatistics:
    """
    Performs inferential statistical analysis.

    All methods validate their inputs and raise ``ValueError`` with a
    clear message on invalid/insufficient data rather than propagating a
    raw exception from pandas/scipy. All results include sample size(s),
    an effect size, and (where applicable) a confidence interval and
    assumption-check diagnostics.
    """

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(dataframe).__name__}.")
        self.dataframe = dataframe

    def _column(self, column: str) -> pd.Series:
        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' not found in the dataframe.")
        return self.dataframe[column]

    # ------------------------------------------------------------------
    # One-sample t-test
    # ------------------------------------------------------------------

    def one_sample_t_test(
        self,
        column: str,
        population_mean: float,
    ) -> Dict[str, Any]:
        """
        Perform a one-sample t-test comparing the mean of ``column``
        against ``population_mean``.
        """
        sample = _as_numeric_series(self._column(column), column).to_numpy()
        _check_min_n(len(sample), 2, "One-sample t-test")

        statistic, p_value = ttest_1samp(sample, population_mean)

        diff = sample - population_mean
        effect_size = _cohens_d_paired(diff)
        normality = _normality_check(sample, column)

        warnings: List[str] = []
        if normality.get("ran") and not normality.get("normal"):
            warnings.append(
                "Shapiro-Wilk suggests the sample is not normally distributed "
                "(p < 0.05); consider a Wilcoxon signed-rank test instead."
            )

        return {
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "n": len(sample),
            "mean": float(np.mean(sample)),
            "population_mean": float(population_mean),
            "confidence_interval_95": _mean_ci(sample),
            "effect_size": {"cohens_d": effect_size, "interpretation": _interpret_d(effect_size)},
            "assumptions": {"normality": normality},
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Independent-samples t-test
    # ------------------------------------------------------------------

    def independent_t_test(
        self,
        column: str,
        group_column: str,
        group1,
        group2,
        equal_var: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Perform an independent two-sample t-test.

        By default (``equal_var=None``) Levene's test is used to decide
        automatically between Welch's t-test (unequal variances -- the
        safer default when in doubt) and the classic Student's t-test.
        Pass ``equal_var=True``/``False`` to force one or the other.
        """
        if group_column not in self.dataframe.columns:
            raise ValueError(f"Column '{group_column}' not found in the dataframe.")

        sample1 = _as_numeric_series(
            self.dataframe[self.dataframe[group_column] == group1][column], column
        ).to_numpy()
        sample2 = _as_numeric_series(
            self.dataframe[self.dataframe[group_column] == group2][column], column
        ).to_numpy()

        _check_min_n(len(sample1), 2, f"Group '{group1}'")
        _check_min_n(len(sample2), 2, f"Group '{group2}'")

        variance_check = _variance_homogeneity_check(sample1, sample2)
        auto_equal_var = variance_check.get("equal_variance", False) if variance_check.get("ran") else False
        use_equal_var = auto_equal_var if equal_var is None else equal_var

        statistic, p_value = ttest_ind(sample1, sample2, equal_var=use_equal_var)

        normality1 = _normality_check(sample1, str(group1))
        normality2 = _normality_check(sample2, str(group2))
        effect_size = _cohens_d(sample1, sample2)

        warnings: List[str] = []
        if variance_check.get("ran") and not variance_check.get("equal_variance"):
            warnings.append(
                "Levene's test suggests unequal variances between groups; "
                "Welch's t-test (equal_var=False) was used." if not use_equal_var else
                "Levene's test suggests unequal variances, but equal_var=True was forced."
            )
        for label, norm in ((group1, normality1), (group2, normality2)):
            if norm.get("ran") and not norm.get("normal"):
                warnings.append(
                    f"Shapiro-Wilk suggests group '{label}' is not normally distributed "
                    "(p < 0.05); consider the Mann-Whitney U test instead."
                )

        return {
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "n1": len(sample1),
            "n2": len(sample2),
            "mean1": float(np.mean(sample1)),
            "mean2": float(np.mean(sample2)),
            "equal_var_used": bool(use_equal_var),
            "confidence_interval_95": _diff_ci(sample1, sample2, equal_var=use_equal_var),
            "effect_size": {"cohens_d": effect_size, "interpretation": _interpret_d(effect_size)},
            "assumptions": {
                "normality_group1": normality1,
                "normality_group2": normality2,
                "homogeneity_of_variance": variance_check,
            },
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Paired t-test
    # ------------------------------------------------------------------

    def paired_t_test(
        self,
        column1: str,
        column2: str,
    ) -> Dict[str, Any]:
        """
        Perform a paired t-test between two related measurements.
        """
        paired = self.dataframe[[column1, column2]].apply(pd.to_numeric, errors="coerce").dropna()
        _check_min_n(len(paired), 2, "Paired t-test")

        sample1 = paired[column1].to_numpy()
        sample2 = paired[column2].to_numpy()

        statistic, p_value = ttest_rel(sample1, sample2)

        diff = sample1 - sample2
        effect_size = _cohens_d_paired(diff)
        normality = _normality_check(diff, "difference scores")

        warnings: List[str] = []
        if normality.get("ran") and not normality.get("normal"):
            warnings.append(
                "Shapiro-Wilk suggests the difference scores are not normally "
                "distributed (p < 0.05); consider a Wilcoxon signed-rank test instead."
            )

        return {
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "n": len(paired),
            "mean_difference": float(np.mean(diff)),
            "confidence_interval_95": _mean_ci(diff),
            "effect_size": {"cohens_d": effect_size, "interpretation": _interpret_d(effect_size)},
            "assumptions": {"normality_of_differences": normality},
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------

    def pearson_correlation(
        self,
        column1: str,
        column2: str,
    ) -> Dict[str, Any]:
        """
        Compute Pearson correlation between two numeric columns.
        """
        paired = self.dataframe[[column1, column2]].apply(pd.to_numeric, errors="coerce").dropna()
        _check_min_n(len(paired), 2, "Pearson correlation")

        x, y = paired[column1].to_numpy(), paired[column2].to_numpy()
        if np.all(x == x[0]) or np.all(y == y[0]):
            raise ValueError(
                "Pearson correlation is undefined when one column is constant "
                "(zero variance)."
            )

        correlation, p_value = pearsonr(x, y)

        normality_x = _normality_check(x, column1)
        normality_y = _normality_check(y, column2)
        warnings: List[str] = []
        for label, norm in ((column1, normality_x), (column2, normality_y)):
            if norm.get("ran") and not norm.get("normal"):
                warnings.append(
                    f"Shapiro-Wilk suggests '{label}' is not normally distributed "
                    "(p < 0.05); consider Spearman's rank correlation instead."
                )

        return {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "n": len(paired),
            "confidence_interval_95": _correlation_ci(float(correlation), len(paired)),
            "effect_size": {
                "r_squared": float(correlation ** 2),
                "interpretation": _interpret_correlation(correlation),
            },
            "assumptions": {"normality_x": normality_x, "normality_y": normality_y},
            "warnings": warnings,
        }

    def spearman_correlation(
        self,
        column1: str,
        column2: str,
    ) -> Dict[str, Any]:
        """
        Compute Spearman rank correlation between two columns.
        """
        paired = self.dataframe[[column1, column2]].apply(pd.to_numeric, errors="coerce").dropna()
        _check_min_n(len(paired), 2, "Spearman correlation")

        x, y = paired[column1].to_numpy(), paired[column2].to_numpy()
        if np.all(x == x[0]) or np.all(y == y[0]):
            raise ValueError(
                "Spearman correlation is undefined when one column is constant "
                "(zero variance)."
            )

        correlation, p_value = spearmanr(x, y)

        return {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "n": len(paired),
            "confidence_interval_95": _correlation_ci(float(correlation), len(paired)),
            "effect_size": {
                "r_squared": float(correlation ** 2),
                "interpretation": _interpret_correlation(correlation),
            },
        }

    # ------------------------------------------------------------------
    # Chi-square test of independence
    # ------------------------------------------------------------------

    def chi_square_test(
        self,
        column1: str,
        column2: str,
    ) -> Dict[str, Any]:
        """
        Perform a Chi-Square test of independence between two categorical
        columns.
        """
        subset = self.dataframe[[column1, column2]].dropna()
        _check_min_n(len(subset), 1, "Chi-square test")

        contingency = pd.crosstab(subset[column1], subset[column2])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            raise ValueError(
                "Chi-square test of independence needs at least 2 categories "
                "in each column."
            )

        statistic, p_value, dof, expected = chi2_contingency(contingency)

        n = int(contingency.to_numpy().sum())
        df_min = min(contingency.shape) - 1
        cramers_v = float(np.sqrt(statistic / (n * df_min))) if n > 0 and df_min > 0 else float("nan")

        low_expected_cells = int(np.sum(expected < 5))
        total_cells = expected.size
        warnings: List[str] = []
        if total_cells and low_expected_cells / total_cells > 0.2:
            warnings.append(
                f"{low_expected_cells}/{total_cells} expected cell counts are below 5; "
                "the chi-square approximation may be unreliable (consider Fisher's exact test)."
            )

        return {
            "chi_square": float(statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "expected": expected,
            "n": n,
            "effect_size": {
                "cramers_v": cramers_v,
                "interpretation": _interpret_cramers_v(cramers_v, df_min),
            },
            "assumptions": {
                "min_expected_count": float(np.min(expected)) if expected.size else float("nan"),
                "low_expected_cells": low_expected_cells,
                "total_cells": int(total_cells),
            },
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # One-way ANOVA
    # ------------------------------------------------------------------

    def one_way_anova(
        self,
        column: str,
        group_column: str,
    ) -> Dict[str, Any]:
        """
        Perform one-way ANOVA of ``column`` across the groups defined by
        ``group_column``.
        """
        if group_column not in self.dataframe.columns:
            raise ValueError(f"Column '{group_column}' not found in the dataframe.")

        working = self.dataframe[[column, group_column]].copy()
        working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.dropna()

        groups: List[np.ndarray] = [
            g[column].to_numpy() for _, g in working.groupby(group_column) if len(g) > 0
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            raise ValueError(
                "One-way ANOVA needs at least 2 non-empty groups; "
                f"found {len(groups)}."
            )
        for g in groups:
            _check_min_n(len(g), 2, "Each group in one-way ANOVA")

        statistic, p_value = f_oneway(*groups)

        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
        eta_squared = float(ss_between / ss_total) if ss_total > 0 else float("nan")

        variance_check = _variance_homogeneity_check(*groups)
        normality_checks = {
            f"group_{i}": _normality_check(g, f"group {i}") for i, g in enumerate(groups)
        }

        warnings: List[str] = []
        if variance_check.get("ran") and not variance_check.get("equal_variance"):
            warnings.append(
                "Levene's test suggests unequal variances across groups "
                "(p < 0.05); consider Welch's ANOVA or the Kruskal-Wallis test."
            )
        if any(nc.get("ran") and not nc.get("normal") for nc in normality_checks.values()):
            warnings.append(
                "At least one group fails the Shapiro-Wilk normality check "
                "(p < 0.05); consider the Kruskal-Wallis test instead."
            )

        return {
            "f_statistic": float(statistic),
            "p_value": float(p_value),
            "n_total": int(sum(len(g) for g in groups)),
            "n_groups": len(groups),
            "group_sizes": [len(g) for g in groups],
            "effect_size": {"eta_squared": eta_squared, "interpretation": _interpret_eta_sq(eta_squared)},
            "assumptions": {
                "homogeneity_of_variance": variance_check,
                "normality_by_group": normality_checks,
            },
            "warnings": warnings,
        }
