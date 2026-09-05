"""
PsyInsight AI
Data Cleaning Module

Author: Subhranshu Ranjan Sahoo

This module provides utilities for cleaning datasets.
"""

import pandas as pd


class DataCleaner:
    """
    A class for cleaning datasets.
    """

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"DataCleaner expects a pandas DataFrame, got {type(dataframe).__name__}."
            )
        self.dataframe = dataframe

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows.
        """
        self.dataframe = self.dataframe.drop_duplicates()
        return self.dataframe

    def fill_missing_values(self, value=0) -> pd.DataFrame:
        """
        Fill missing values.
        """
        self.dataframe = self.dataframe.fillna(value)
        return self.dataframe

    def rename_columns(self) -> pd.DataFrame:
        """
        Standardize column names.

        Coerces column labels to strings first, so datasets with
        integer or mixed-type column headers don't raise an
        AttributeError on the `.str` accessor.
        """
        self.dataframe.columns = (
            self.dataframe.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return self.dataframe

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the cleaned DataFrame.
        """
        return self.dataframe
