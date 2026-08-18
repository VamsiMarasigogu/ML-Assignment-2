# Machine Learning Assignment - 2

## 1. Problem Statement

The objective is to implement and compare multiple classification models on a common
binary classification dataset and demonstrate the trained models through an interactive
Streamlit application.

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and
Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository, Dataset ID 17.

The dataset contains **569 instances and 30 numerical features**. The target is binary:
**M = malignant** and **B = benign**. There are no missing values in the original dataset.

Dataset source:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset was selected because it satisfies the assignment requirement of at least
12 features and at least 500 instances.

## 3. GitHub Repository Link

**Paste your GitHub repository URL here after creating the repository.**

Repository contents:
- `app.py`
- `train_models.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- `metrics.csv`
- `model/*.joblib`

## 4. Models Used and Evaluation

The five named models in the assignment are implemented on the **same dataset and same
train/test split**.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |\n| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |\n| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |\n| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |\n| Random Forest | 0.9649 | 0.9970 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |\n
### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline. StandardScaler is used because the feature magnitudes vary substantially. |
| Decision Tree | Captures non-linear decision boundaries and is easy to interpret. The tree is limited to a maximum depth of 5 to reduce overfitting. |
| kNN | Performance depends strongly on feature scale and neighborhood size; therefore StandardScaler is used before kNN. |
| Naive Bayes | Gaussian Naive Bayes is computationally simple and works by modelling each feature distribution conditionally on the class. |
| Random Forest (Ensemble) | Combines many decision trees and generally provides robust performance on non-linear tabular data. |
| Overall Winner | Based on the highest F1 score in this run, **Logistic Regression** is the overall winner. If your BITS Lab run gives different results after you modify the split/hyperparameters, update this row using your actual output. |

### Reproducibility

- Test size: 20%
- `random_state = 42`
- Stratified train/test split
- Positive class for AUC/Precision/Recall/F1/MCC: **M (malignant)**
- Scaling is applied inside pipelines for Logistic Regression and kNN to prevent data leakage.

## 5. Streamlit App

The application provides:

1. CSV test-data upload.
2. Model selection dropdown.
3. Display of Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Confusion matrix.
5. Classification report.
6. Test-data preview.

**Live Streamlit App Link:**  
Paste your deployed Streamlit Community Cloud URL here.

## 6. How to Run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

For Streamlit Community Cloud, set the main file to `app.py`.

## Academic Integrity Note

This repository is a reference implementation for learning. Before submission, run the
code yourself in BITS Virtual Lab, verify all numerical outputs, make meaningful
customizations, and maintain your own Git commit history as required by the assignment.
