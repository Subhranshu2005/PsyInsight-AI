"""
validator.py
------------

Data Validation Engine for PsyInsight AI

Author: Subhranshu Ranjan Sahoo

This module validates datasets before statistical analysis
or machine learning.
"""

import pandas as pd


class DataValidator:
    """
    Validates a pandas DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def check_missing_values(self):
        """
        Returns missing values in each column.
        """
        return self.dataframe.isnull().sum()

    def check_duplicate_rows(self):
        """
        Returns the number of duplicate rows.
        """
        return self.dataframe.duplicated().sum()

    def check_empty_columns(self):
        """
        Returns columns containing only missing values.
        """
        return self.dataframe.columns[
            self.dataframe.isnull().all()
        ].tolist()

    def check_constant_columns(self):
        """
        Returns columns containing only one unique value.
        """
        constant_columns = []

        for column in self.dataframe.columns:
            if self.dataframe[column].nunique(dropna=False) == 1:
                constant_columns.append(column)

        return constant_columns

    def validate(self):
        """
        Runs all validation checks.
        """
        report = {
            "Rows": self.dataframe.shape[0],
            "Columns": self.dataframe.shape[1],
            "Missing Values": self.check_missing_values().to_dict(),
            "Duplicate Rows": self.check_duplicate_rows(),
            "Empty Columns": self.check_empty_columns(),
            "Constant Columns": self.check_constant_columns(),
        }

        return report
