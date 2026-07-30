"""
PsyInsight AI
Inferential Statistics Module

Author: Subhranshu Ranjan Sahoo

This module provides inferential statistical methods.
"""

import pandas as pd
from scipy.stats import ttest_ind


class InferentialStatistics:
    """
    Performs inferential statistical tests.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def independent_t_test(
        self,
        column: str,
        group_column: str,
        group1,
        group2,
    ):
        """
        Perform an independent two-sample t-test.

        Parameters
        ----------
        column : str
            Numeric column.
        group_column : str
            Grouping column.
        group1 :
            First group label.
        group2 :
            Second group label.

        Returns
        -------
        dict
            t-statistic and p-value.
        """

        sample1 = self.dataframe[
            self.dataframe[group_column] == group1
        ][column].dropna()

        sample2 = self.dataframe[
            self.dataframe[group_column] == group2
        ][column].dropna()

        t_statistic, p_value = ttest_ind(
            sample1,
            sample2,
            equal_var=False,
        )

        return {
            "t_statistic": t_statistic,
            "p_value": p_value,
        }
