"""
PsyInsight AI
Descriptive Statistics Module

Author: Subhranshu Ranjan Sahoo

This module provides descriptive statistical analysis for datasets.
"""

import pandas as pd


class DescriptiveStatistics:
    """
    A class for performing descriptive statistical analysis.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def mean(self):
        """
        Return the mean of all numeric columns.
        """
        return self.dataframe.mean(numeric_only=True)

    def median(self):
        """
        Return the median of all numeric columns.
        """
        return self.dataframe.median(numeric_only=True)

    def mode(self):
        """
        Return the mode of all columns.
        """
        return self.dataframe.mode()

    def variance(self):
        """
        Return the variance of numeric columns.
        """
        return self.dataframe.var(numeric_only=True)

    def standard_deviation(self):
        """
        Return the standard deviation of numeric columns.
        """
        return self.dataframe.std(numeric_only=True)
