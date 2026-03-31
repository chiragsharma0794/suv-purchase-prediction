"""
model.py
--------
Trains a Logistic Regression model using scikit-learn.
"""

from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train,
    y_train,
    max_iter: int = 200,
    random_state: int = 42
) -> LogisticRegression:
    """
    Fits a Logistic Regression model on the training data.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training target vector.
        max_iter: Maximum number of iterations for the solver.
        random_state: Seed for reproducibility.

    Returns:
        A trained LogisticRegression object.
    """
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    print("[INFO] Logistic Regression model trained successfully.")
    print(f"       Model intercept: {model.intercept_}")
    print(f"       Model coefficients: {model.coef_}")
    return model
