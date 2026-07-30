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

    def minimum(self):
        """
        Return the minimum value of numeric columns.
        """
        return self.dataframe.min(numeric_only=True)

    def maximum(self):
        """
        Return the maximum value of numeric columns.
        """
        return self.dataframe.max(numeric_only=True)

    def data_range(self):
        """
        Return the range of numeric columns.
        """
        numeric = self.dataframe.select_dtypes(include="number")
        return numeric.max() - numeric.min()

    def quartiles(self):
        """
        Return the first, second, and third quartiles.
        """
        return self.dataframe.quantile(
            [0.25, 0.50, 0.75],
            numeric_only=True
        )

    def interquartile_range(self):
        """
        Return the Interquartile Range (IQR).
        """
        numeric = self.dataframe.select_dtypes(include="number")

        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)

        return q3 - q1

    def skewness(self):
        """
        Return the skewness of numeric columns.
        """
        return self.dataframe.skew(numeric_only=True)

    def kurtosis(self):
        """
        Return the kurtosis of numeric columns.
        """
        return self.dataframe.kurt(numeric_only=True)

    def count(self):
        """
        Return the number of non-missing values.
        """
        return self.dataframe.count()

    def summary(self):
        """
        Return a statistical summary of numeric columns.
        """
        return self.dataframe.describe()
