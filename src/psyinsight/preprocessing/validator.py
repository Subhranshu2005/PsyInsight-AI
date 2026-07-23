"""
PsyInsight AI
Data Validation Engine

Author: Subhranshu Ranjan Sahoo

This module validates psychology datasets before
machine learning and statistical analysis.
"""

import pandas as pd


class DataValidator:
    """
    Validates datasets before preprocessing.
    """

    def __init__(self, dataframe):
        """
        Store the dataset.
        """
        self.df = dataframe

    def dataset_shape(self):
        """
        Returns the number of rows and columns.
        """
        return self.df.shape

    def missing_values(self):
        """
        Returns missing values in every column.
        """
        return self.df.isnull().sum()

    def duplicate_rows(self):
        """
        Returns duplicate row count.
        """
        return self.df.duplicated().sum()

    def data_types(self):
        """
        Returns datatype of each column.
        """
        return self.df.dtypes
