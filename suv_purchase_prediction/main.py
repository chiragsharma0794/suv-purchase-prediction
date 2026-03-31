"""
main.py
-------
Entry point for the SUV Purchase Prediction project.

Runs the full ML pipeline:
  1. Load and explore data
  2. Preprocess (encode, select features, split, scale)
  3. Train Logistic Regression
  4. Evaluate model (accuracy + confusion matrix)
  5. Visualize (decision boundary + confusion matrix plot)
  6. Compare different train/test split sizes

Usage:
    python main.py
"""

import os
from src.data_loader import load_dataset, explore_dataset
from src.preprocessor import (
    encode_categorical_columns,
    select_features_and_target,
    split_train_test,
    scale_features
)
from src.model import train_logistic_regression
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_decision_boundary,
    compare_test_sizes
)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "suv_data.csv")
FEATURE_COLUMNS = ["Age", "EstimatedSalary"]
TARGET_COLUMN = "Purchased"
TEST_SIZE = 0.20
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline():
    """
    Runs the complete SUV purchase prediction ML pipeline.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1 — Load and explore data
    print("\n" + "=" * 60)
    print("STEP 1: Data Loading & Exploration")
    print("=" * 60)
    raw_data = load_dataset(DATA_PATH)
    explore_dataset(raw_data)

    # Step 2 — Preprocess
    print("\n" + "=" * 60)
    print("STEP 2: Data Preprocessing")
    print("=" * 60)

    # Encode Gender (Male/Female -> 1/0)
    processed_data = encode_categorical_columns(raw_data, column_name="Gender")

    # Drop User ID — it's just an identifier, not a useful feature
    if "User ID" in processed_data.columns:
        processed_data = processed_data.drop(columns=["User ID"])

    # Select features and target
    features, target = select_features_and_target(
        processed_data,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN
    )

    # Step 3 — Train/Test Split
    print("\n" + "=" * 60)
    print("STEP 3: Train-Test Split (80/20)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = split_train_test(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Step 4 — Feature Scaling
    print("\n" + "=" * 60)
    print("STEP 4: Feature Scaling")
    print("=" * 60)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 5 — Model Training
    print("\n" + "=" * 60)
    print("STEP 5: Model Training")
    print("=" * 60)
    model = train_logistic_regression(X_train_scaled, y_train, random_state=RANDOM_STATE)

    # Step 6 — Evaluation
    print("\n" + "=" * 60)
    print("STEP 6: Model Evaluation")
    print("=" * 60)
    results = evaluate_model(model, X_test_scaled, y_test)

    # Step 7 — Visualizations
    print("\n" + "=" * 60)
    print("STEP 7: Visualizations")
    print("=" * 60)
    cm_save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], save_path=cm_save_path)

    db_save_path = os.path.join(OUTPUT_DIR, "decision_boundary.png")
    plot_decision_boundary(model, X_test_scaled, y_test, save_path=db_save_path)

    # Step 8 — Compare split sizes (Stretch)
    print("\n" + "=" * 60)
    print("STEP 8: Comparing Different Train/Test Splits")
    print("=" * 60)
    compare_test_sizes(processed_data, FEATURE_COLUMNS, TARGET_COLUMN)

    print("\n[DONE] Pipeline complete. Plots saved in the outputs/ folder.")


if __name__ == "__main__":
    run_pipeline()
