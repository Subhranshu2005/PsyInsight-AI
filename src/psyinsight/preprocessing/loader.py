"""
PsyInsight AI
Data Loader Module

Author: Subhranshu Ranjan Sahoo

This module loads datasets from different file formats.
"""

from pathlib import Path

import pandas as pd


class DataLoader:
    """
    A class to load datasets into pandas DataFrames.
    """

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path: Path to the CSV file.

        Returns:
            pandas.DataFrame
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV file: {e}")
