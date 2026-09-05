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

    # Guard against accidentally loading a huge file into memory from a
    # script/CLI context (the Streamlit app applies its own, separate
    # limit on uploaded files -- see app/main.py).
    MAX_FILE_SIZE_MB: float = 200

    @staticmethod
    def _check_file_size(file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            return  # let the format-specific reader raise FileNotFoundError
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > DataLoader.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File '{file_path}' is {size_mb:.1f} MB, which exceeds the "
                f"{DataLoader.MAX_FILE_SIZE_MB} MB limit. Load a smaller file, "
                "increase DataLoader.MAX_FILE_SIZE_MB, or stream/chunk it yourself."
            )

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path: Path to the CSV file.

        Returns:
            pandas.DataFrame
        """
        DataLoader._check_file_size(file_path)
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"'{file_path}' is empty or has no parseable columns.")
        except Exception as e:
            raise Exception(f"Error loading CSV file: {e}")

        if df.shape[1] == 0:
            raise ValueError(f"'{file_path}' has no columns after parsing.")
        return df

    @staticmethod
    def load_excel(file_path: str) -> pd.DataFrame:
        """
        Load an Excel file into a pandas DataFrame.

        Args:
            file_path: Path to the Excel file.

        Returns:
            pandas.DataFrame
        """
        DataLoader._check_file_size(file_path)
        try:
            return pd.read_excel(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading Excel file: {e}")

    @staticmethod
    def load_json(file_path: str) -> pd.DataFrame:
        """
        Load a JSON file into a pandas DataFrame.

        Args:
            file_path: Path to the JSON file.

        Returns:
            pandas.DataFrame
        """
        DataLoader._check_file_size(file_path)
        try:
            return pd.read_json(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except ValueError as e:
            raise ValueError(f"'{file_path}' is not valid JSON: {e}")
        except Exception as e:
            raise Exception(f"Error loading JSON file: {e}")

    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """
        Detect the file extension.

        Args:
            file_path: Path to the dataset.

        Returns:
            File extension as a string.
        """
        return Path(file_path).suffix.lower()

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        """
        Automatically load a dataset based on its file extension.

        Supported formats:
        - CSV
        - Excel (.xlsx, .xls)
        - JSON

        Args:
            file_path: Path to the dataset.

        Returns:
            pandas.DataFrame
        """
        extension = DataLoader.detect_file_type(file_path)

        if extension == ".csv":
            return DataLoader.load_csv(file_path)

        elif extension in [".xlsx", ".xls"]:
            return DataLoader.load_excel(file_path)

        elif extension == ".json":
            return DataLoader.load_json(file_path)

        else:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )
