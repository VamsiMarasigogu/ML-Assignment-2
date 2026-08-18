import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="📊",
    layout="wide"
)

FEATURES = [
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

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest.joblib"
}

st.title("📊 Machine Learning Classification Model Comparison")
st.caption("Breast Cancer Wisconsin (Diagnostic) — UCI dataset")

st.markdown("""
### Objective
Compare five classification algorithms on the same dataset using
Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).
""")

uploaded = st.file_uploader(
    "Upload test data CSV",
    type=["csv"],
    help="Upload test_data.csv. It must contain the 30 feature columns and diagnosis."
)

if uploaded is None:
    st.info("Please upload the test CSV to evaluate the models.")
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in FEATURES + ["diagnosis"] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

X = df[FEATURES]
y_raw = df["diagnosis"].astype(str).str.upper().str.strip()

# UCI convention: M = malignant, B = benign
if not y_raw.isin(["M", "B"]).all():
    st.error("The diagnosis column must contain only M (malignant) or B (benign).")
    st.stop()

y = (y_raw == "M").astype(int)

# Load all models and compare them on the uploaded test data.
all_results = {}
loaded_models = {}

for model_name, model_file in MODEL_FILES.items():
    model_path = Path(model_file)
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)
    loaded_models[model_name] = model

    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    all_results[model_name] = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred)
    }

comparison_df = pd.DataFrame(all_results).T
comparison_df.index.name = "ML Model"

st.subheader("Model Comparison on Uploaded Test Data")
st.dataframe(comparison_df.round(4), use_container_width=True)

selected_model_name = st.selectbox(
    "Select a model for detailed evaluation",
    list(MODEL_FILES.keys())
)

model = loaded_models[selected_model_name]
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

metrics = all_results[selected_model_name]

st.subheader(f"Evaluation Metrics — {selected_model_name}")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, pred, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["Actual B", "Actual M"],
    columns=["Predicted B", "Predicted M"]
)
st.dataframe(cm_df, use_container_width=True)

st.subheader("Classification Report")
report = classification_report(
    y, pred, target_names=["Benign (B)", "Malignant (M)"],
    output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("Test Data Preview")
st.dataframe(df.head(10), use_container_width=True)

st.success(f"All five models were evaluated on {len(df)} test records.")
