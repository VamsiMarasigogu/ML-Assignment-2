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

https://github.com/VamsiMarasigogu/ML-Assignment-2

Repository contents:
- `app.py`
- `train_models.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- `metrics.csv`
- `model/*.joblib`

## 4. Models Used and Evaluation

The five classification models specified in the assignment are implemented on the
same dataset using the same stratified train/test split.

ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC
--- | --- | --- | --- | --- | --- | ---
Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245
Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299
kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058
Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715
Random Forest (Ensemble) | 0.9649 | 0.9970 | 1.0000 | 0.9048 | 0.9500 | 0.9258

### Observations

ML Model Name | Observation about model performance
--- | ---
Logistic Regression | Achieved 96.49% accuracy and the highest F1 score of 0.9512. It also achieved the highest recall of 0.9286, making it a strong linear baseline for this dataset.
Decision Tree | Achieved 92.11% accuracy, which was the lowest among the five models. Its lower F1 and MCC indicate weaker overall classification performance compared with the other models.
kNN | Achieved 95.61% accuracy with an F1 score of 0.9383. Its performance benefits from feature scaling because kNN is sensitive to differences in feature magnitude.
Naive Bayes | Achieved perfect precision of 1.0000 but lower recall of 0.8333. This indicates that while its positive predictions were highly precise, it missed some malignant cases.
Random Forest (Ensemble) | Achieved the highest AUC of 0.9970 and highest MCC of 0.9258. It also achieved perfect precision of 1.0000 and tied for the highest accuracy at 0.9649.

### Overall Winner

Based on the evaluation results, **Random Forest (Ensemble)** is considered the
overall best-performing model for this dataset.

Random Forest achieved the highest AUC (0.9970) and MCC (0.9258), perfect
precision (1.0000), and tied with Logistic Regression for the highest accuracy
(0.9649). Although Logistic Regression achieved slightly higher recall (0.9286)
and F1 score (0.9512), Random Forest provides the strongest overall balance
across the evaluation metrics.

Therefore, **Random Forest (Ensemble) is selected as the overall winner** for
this classification task.

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
https://vamsimarasigogu-ml-assignment-2-app-fmqdrz.streamlit.app/

The application allows users to upload test data, select a classification
model, and view the corresponding evaluation metrics, confusion matrix,
classification report and test-data preview.

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
