"""
evaluate.py
-----------
Functions to evaluate the trained model and generate visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Generates predictions and computes accuracy and confusion matrix.

    Args:
        model: A trained scikit-learn classifier.
        X_test: Scaled test feature matrix.
        y_test: True labels for the test set.

    Returns:
        A dictionary with 'accuracy', 'confusion_matrix', and 'predictions'.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f"\n===== Model Evaluation =====")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nInterpretation:")
    print(f"  True Negatives  (predicted No,  actual No):  {conf_matrix[0][0]}")
    print(f"  False Positives (predicted Yes, actual No):  {conf_matrix[0][1]}")
    print(f"  False Negatives (predicted No,  actual Yes): {conf_matrix[1][0]}")
    print(f"  True Positives  (predicted Yes, actual Yes): {conf_matrix[1][1]}")

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "predictions": predictions
    }


def plot_confusion_matrix(conf_matrix, save_path: str = None) -> None:
    """
    Plots and optionally saves the confusion matrix as a heatmap.

    Args:
        conf_matrix: The confusion matrix array.
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Not Purchased", "Purchased"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — SUV Purchase Prediction")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Confusion matrix saved to: {save_path}")
    plt.show()


def plot_decision_boundary(model, X_test_scaled, y_test, save_path: str = None) -> None:
    """
    Plots the decision boundary of the logistic regression model in 2D.
    Only works when there are exactly 2 features.

    Args:
        model: Trained LogisticRegression model.
        X_test_scaled: Scaled test features (2 features required).
        y_test: True labels.
        save_path: If provided, saves the figure to this path.
    """
    if X_test_scaled.shape[1] != 2:
        print("[WARN] Decision boundary plot only supports 2 features. Skipping.")
        return

    y_test_array = np.array(y_test)

    # Build a mesh grid to evaluate the model across the feature space
    x_min = X_test_scaled[:, 0].min() - 1
    x_max = X_test_scaled[:, 0].max() + 1
    y_min = X_test_scaled[:, 1].min() - 1
    y_max = X_test_scaled[:, 1].max() + 1

    step = 0.01
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    grid_predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
    grid_predictions = grid_predictions.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, grid_predictions, alpha=0.3, cmap="coolwarm")
    scatter = ax.scatter(
        X_test_scaled[:, 0],
        X_test_scaled[:, 1],
        c=y_test_array,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.5,
        s=40
    )

    not_purchased_patch = mpatches.Patch(color="blue", alpha=0.4, label="Not Purchased (0)")
    purchased_patch = mpatches.Patch(color="red", alpha=0.4, label="Purchased (1)")
    ax.legend(handles=[not_purchased_patch, purchased_patch], loc="upper left")

    ax.set_xlabel("Age (scaled)")
    ax.set_ylabel("Estimated Salary (scaled)")
    ax.set_title("Decision Boundary — Logistic Regression on SUV Dataset")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Decision boundary plot saved to: {save_path}")
    plt.show()


def compare_test_sizes(dataframe, feature_columns: list, target_column: str) -> None:
    """
    Trains and evaluates the model with different train/test split ratios
    to see the effect on accuracy.

    Args:
        dataframe: The fully preprocessed DataFrame (encoded, scaled NOT applied yet).
        feature_columns: List of feature column names.
        target_column: Target column name.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    test_sizes = [0.20, 0.25, 0.30]
    split_labels = ["80/20", "75/25", "70/30"]

    print("\n===== Accuracy Comparison Across Split Ratios =====")
    print(f"{'Split':<12} {'Train Samples':<16} {'Test Samples':<14} {'Accuracy':>10}")
    print("-" * 55)

    for test_size, label in zip(test_sizes, split_labels):
        features = dataframe[feature_columns]
        target = dataframe[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_train_sc, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_sc))

        print(f"{label:<12} {X_train.shape[0]:<16} {X_test.shape[0]:<14} {acc * 100:>9.2f}%")
