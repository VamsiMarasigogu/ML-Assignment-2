"""
Training script for ML Assignment 2.

Run this script in BITS Virtual Lab:
    python train_models.py

It trains the five mandatory models on the same dataset,
computes the six required metrics, saves the trained models,
and creates test_data.csv and metrics.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# Load the UCI WDBC-compatible dataset available through scikit-learn.
# The assignment README should cite the original UCI source.
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
X.columns = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

# scikit-learn target: 0=malignant, 1=benign.
# Convert to assignment/UCI convention: 1=malignant, 0=benign.
y = (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
}

Path("model").mkdir(exist_ok=True)
rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    rows.append({
        "ML Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred)
    })

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, Path("model") / filename)

pd.DataFrame(rows).to_csv("metrics.csv", index=False)

test_output = X_test.copy()
test_output["diagnosis"] = np.where(y_test == 1, "M", "B")
test_output.to_csv("test_data.csv", index=False)

print(pd.DataFrame(rows).round(4))
print("\nCreated model/*.joblib, metrics.csv and test_data.csv")
