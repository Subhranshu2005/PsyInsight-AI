"""
PsyInsight AI
Data Validation Engine
"""

import pandas as pd


class DataValidator:
    """
    Validates datasets before analysis.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def dataset_shape(self):
        """Return dataset shape."""
        return self.dataframe.shape

    def missing_values(self):
        """Return missing values in each column."""
        return self.dataframe.isnull().sum()

    def duplicate_rows(self):
        """Return duplicate row count."""
        return self.dataframe.duplicated().sum()

    def data_types(self):
        """Return data types."""
        return self.dataframe.dtypes

    def validate(self):
        """
        Runs all validation checks and returns a report.
        """
        report = {
            "Dataset Shape": self.dataset_shape(),
            "Missing Values": self.missing_values().to_dict(),
            "Duplicate Rows": self.duplicate_rows(),
            "Data Types": self.data_types().astype(str).to_dict(),
        }

        return report
