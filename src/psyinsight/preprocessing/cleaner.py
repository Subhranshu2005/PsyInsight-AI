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
        """
        self.dataframe.columns = (
            self.dataframe.columns
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
