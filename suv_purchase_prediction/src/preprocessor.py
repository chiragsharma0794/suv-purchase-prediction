"""
preprocessor.py
---------------
Handles all preprocessing steps: encoding categorical variables,
selecting features, splitting into train/test sets, and scaling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def encode_categorical_columns(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Label-encodes a categorical column in the DataFrame.
    Male -> 1, Female -> 0 (or whatever order LabelEncoder picks up).

    Args:
        dataframe: The DataFrame containing the column.
        column_name: Name of the categorical column to encode.

    Returns:
        DataFrame with the column encoded as integers.
    """
    encoder = LabelEncoder()
    dataframe = dataframe.copy()
    dataframe[column_name] = encoder.fit_transform(dataframe[column_name])
    print(f"[INFO] Encoded '{column_name}' column. Classes: {list(encoder.classes_)}")
    return dataframe


def select_features_and_target(
    dataframe: pd.DataFrame,
    feature_columns: list,
    target_column: str
):
    """
    Separates the DataFrame into feature matrix (X) and target vector (y).

    Args:
        dataframe: The preprocessed DataFrame.
        feature_columns: List of column names to use as features.
        target_column: Column name for the target variable.

    Returns:
        A tuple (X, y) as DataFrames/Series.

    Raises:
        ValueError: If any specified column is missing from the DataFrame.
    """
    missing_cols = [col for col in feature_columns + [target_column] if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")

    features = dataframe[feature_columns]
    target = dataframe[target_column]

    print(f"[INFO] Features selected: {feature_columns}")
    print(f"[INFO] Target column: {target_column}")
    print(f"[INFO] Class distribution:\n{target.value_counts()}")
    return features, target


def split_train_test(features, target, test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target into training and test sets.

    Args:
        features: Feature matrix (X).
        target: Target vector (y).
        test_size: Proportion of data to hold out for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Applies StandardScaler to the feature sets.
    Scaler is fit only on training data to avoid data leakage.

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.

    Returns:
        Tuple: (X_train_scaled, X_test_scaled, fitted scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] Feature scaling applied (StandardScaler).")
    return X_train_scaled, X_test_scaled, scaler
