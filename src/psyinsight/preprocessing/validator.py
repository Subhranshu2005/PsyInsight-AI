"""
PsyInsight AI
Data Validation Engine

Author: Subhranshu Ranjan Sahoo

This module validates datasets before
statistical analysis and machine learning.
"""

import pandas as pd


class DataValidator:
    """
    Validates datasets before analysis.
    """

    def __init__(self, dataframe):
        """
        Initialize the validator with a pandas DataFrame.
        """
        self.dataframe = dataframe

    def dataset_shape(self):
        """
        Return the number of rows and columns.
        """
        return self.dataframe.shape

    def missing_values(self):
        """
        Return the number of missing values in each column.
        """
        return self.dataframe.isnull().sum()

    def missing_value_percentage(self):
        """
        Return the percentage of missing values in each column.
        """
        percentage = (
            self.dataframe.isnull().sum()
            / len(self.dataframe)
        ) * 100

        return percentage.round(2)

    def duplicate_rows(self):
        """
        Return the number of duplicate rows.
        """
        return self.dataframe.duplicated().sum()

    def data_types(self):
        """
        Return the data type of every column.
        """
        return self.dataframe.dtypes

    def validate(self):
        """
        Run all validation checks and return a report.
        """
        report = {
            "Dataset Shape": self.dataset_shape(),
            "Missing Values": self.missing_values().to_dict(),
            "Missing Value Percentage": self.missing_value_percentage().to_dict(),
            "Duplicate Rows": self.duplicate_rows(),
            "Data Types": self.data_types().astype(str).to_dict(),
        }

        return report
