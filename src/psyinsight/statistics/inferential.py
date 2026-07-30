"""
PsyInsight AI
Inferential Statistics Module

Author: Subhranshu Ranjan Sahoo

This module provides inferential statistical methods.
"""

import pandas as pd

from scipy.stats import (
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    pearsonr,
    spearmanr,
    chi2_contingency,
    f_oneway,
)


class InferentialStatistics:
    """
    Performs inferential statistical analysis.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def one_sample_t_test(
        self,
        column: str,
        population_mean: float,
    ):
        """
        Perform a one-sample t-test.
        """

        sample = self.dataframe[column].dropna()

        statistic, p_value = ttest_1samp(
            sample,
            population_mean,
        )

        return {
            "t_statistic": statistic,
            "p_value": p_value,
        }

    def independent_t_test(
        self,
        column: str,
        group_column: str,
        group1,
        group2,
    ):
        """
        Perform an independent two-sample t-test.
        """

        sample1 = self.dataframe[
            self.dataframe[group_column] == group1
        ][column].dropna()

        sample2 = self.dataframe[
            self.dataframe[group_column] == group2
        ][column].dropna()

        statistic, p_value = ttest_ind(
            sample1,
            sample2,
            equal_var=False,
        )

        return {
            "t_statistic": statistic,
            "p_value": p_value,
        }

    def paired_t_test(
        self,
        column1: str,
        column2: str,
    ):
        """
        Perform a paired t-test.
        """

        statistic, p_value = ttest_rel(
            self.dataframe[column1].dropna(),
            self.dataframe[column2].dropna(),
        )

        return {
            "t_statistic": statistic,
            "p_value": p_value,
        }

    def pearson_correlation(
        self,
        column1: str,
        column2: str,
    ):
        """
        Compute Pearson correlation.
        """

        correlation, p_value = pearsonr(
            self.dataframe[column1],
            self.dataframe[column2],
        )

        return {
            "correlation": correlation,
            "p_value": p_value,
        }

    def spearman_correlation(
        self,
        column1: str,
        column2: str,
    ):
        """
        Compute Spearman correlation.
        """

        correlation, p_value = spearmanr(
            self.dataframe[column1],
            self.dataframe[column2],
        )

        return {
            "correlation": correlation,
            "p_value": p_value,
        }

    def chi_square_test(
        self,
        column1: str,
        column2: str,
    ):
        """
        Perform Chi-Square test of independence.
        """

        contingency = pd.crosstab(
            self.dataframe[column1],
            self.dataframe[column2],
        )

        statistic, p_value, dof, expected = chi2_contingency(
            contingency
        )

        return {
            "chi_square": statistic,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "expected": expected,
        }

    def one_way_anova(
        self,
        column: str,
        group_column: str,
    ):
        """
        Perform one-way ANOVA.
        """

        groups = [
            group[column].dropna()
            for _, group in self.dataframe.groupby(group_column)
        ]

        statistic, p_value = f_oneway(*groups)

        return {
            "f_statistic": statistic,
            "p_value": p_value,
        }
