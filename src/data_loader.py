"""
data_loader.py
--------------
Handles loading the SUV dataset and doing a quick sanity check on it.
"""

import os
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the SUV dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame with the raw data.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist at the given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    dataframe = pd.read_csv(file_path)
    print(f"[INFO] Dataset loaded successfully. Shape: {dataframe.shape}")
    return dataframe


def explore_dataset(dataframe: pd.DataFrame) -> None:
    """
    Prints basic information about the dataset — first few rows, shape,
    column names, data types, and missing value counts.

    Args:
        dataframe: The raw DataFrame to inspect.
    """
    print("\n===== First 5 Rows =====")
    print(dataframe.head())

    print(f"\n===== Shape =====")
    print(f"Rows: {dataframe.shape[0]}, Columns: {dataframe.shape[1]}")

    print("\n===== Column Names =====")
    print(list(dataframe.columns))

    print("\n===== Data Types =====")
    print(dataframe.dtypes)

    print("\n===== Missing Values =====")
    missing = dataframe.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing[missing > 0])

    print("\n===== Basic Statistics =====")
    print(dataframe.describe())
