# SUV Purchase Prediction — Logistic Regression

**Week 06 · Day 30 Assignment**  
**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**

---

## Overview

This project builds an end-to-end Logistic Regression pipeline to predict whether a social network user will purchase an SUV based on their **Age** and **Estimated Salary**.

Dataset source: [Kaggle — SUV Purchase Dataset](https://www.kaggle.com/datasets/bittupanchal/logistics-regression-on-suv-dataset)

**Result:** Achieved **86.25% accuracy** on a held-out 20% test set.

---

## Project Structure

```
suv_purchase_prediction/
│
├── data/
│   └── suv_data.csv              # Raw dataset
│
├── notebooks/
│   └── suv_logistic_regression.ipynb   # Full notebook with analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Loading + exploration functions
│   ├── preprocessor.py           # Encoding, splitting, scaling
│   ├── model.py                  # Model training
│   └── evaluate.py               # Evaluation + visualizations
│
├── outputs/
│   ├── confusion_matrix.png      # Generated at runtime
│   └── decision_boundary.png     # Generated at runtime
│
├── main.py                       # Entry point — runs full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & How to Run

### 1. Clone the repository

```bash
git clone https://github.com/chiragsharma0794/suv-purchase-prediction.git
cd suv-purchase-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

Plots will be saved automatically in the `outputs/` folder.

### 5. Open the notebook (optional)

```bash
jupyter notebook notebooks/suv_logistic_regression.ipynb
```

---

## What the Pipeline Does

| Step | Description |
|------|-------------|
| Load | Reads CSV, prints shape, types, missing values |
| Preprocess | Encodes Gender, drops User ID, selects Age + Salary |
| Split | 80/20 train-test split (random_state=42) |
| Scale | StandardScaler — fit on train only (no leakage) |
| Train | Logistic Regression (max_iter=200) |
| Evaluate | Accuracy + confusion matrix with TP/TN/FP/FN breakdown |
| Visualize | Decision boundary + confusion matrix heatmap |
| Compare | Accuracy across 80/20, 75/25, 70/30 splits |

---

## Results

| Split | Train Samples | Test Samples | Accuracy |
|-------|--------------|-------------|----------|
| 80/20 | 320 | 80 | **86.25%** |
| 75/25 | 300 | 100 | 86.00% |
| 70/30 | 280 | 120 | 85.00% |

**Confusion Matrix (80/20 split):**

|  | Predicted: No | Predicted: Yes |
|--|--|--|
| **Actual: No** | 50 (TN) | 2 (FP) |
| **Actual: Yes** | 9 (FN) | 19 (TP) |

---

## Interview Q&A (Part C)

**Q1 — What is Logistic Regression? Is it classification or regression?**  
It is a **classification** algorithm. It applies the sigmoid function to a linear combination of features to output a probability between 0 and 1. If the probability ≥ 0.5, it predicts class 1, otherwise class 0. The "regression" in the name refers to fitting a linear equation to the log-odds, not to predicting a continuous value.

**Q3 — What is a Confusion Matrix?**  
A confusion matrix is a 2x2 table (for binary classification) that shows how many predictions the model made correctly and incorrectly, broken down by actual class. It contains True Positives, True Negatives, False Positives, and False Negatives — giving a much more complete picture than accuracy alone.

---

## Dependencies

- Python 3.11+
- pandas
- numpy
- scikit-learn
- matplotlib
- notebook (for Jupyter)

See `requirements.txt` for pinned versions.
