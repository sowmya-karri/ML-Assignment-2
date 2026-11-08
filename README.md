# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification models to classify dry bean varieties based on their physical measurements.

The models are trained and evaluated using the Dry Bean Dataset. Their performance is compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

- **Dataset:** Dry Bean Dataset
- **Source:** UCI Machine Learning Repository
- **Problem Type:** Multiclass Classification
- **Target Variable:** `Class`
- **Number of Classes:** 7
- **Features:** 16 numerical features
- **Dataset Size After Duplicate Removal:** 13,543 records
- **Training Data:** 10,834 records
- **Testing Data:** 2,709 records

The dataset contains physical measurements of dry bean samples, such as area, perimeter, major axis length, minor axis length, eccentricity, and other shape-related features.

The target variable identifies the dry bean variety. The seven classes are:
- Dermason
- Sira
- Seker
- Horoz
- Cali
- Barbunya
- Bombay

## 3. GitHub Repository Link

[ML Assignment 2 - GitHub Repository](https://github.com/sowmya-karri/ML-Assignment-2)

The repository contains the notebook/source code, dataset, model implementation, evaluation results, and Streamlit application files.

## 4. Models Used

The following five classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### 4.1 Evaluation Metrics Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9947 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.8966 | 0.9465 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| K-Nearest Neighbors | 0.9155 | 0.9832 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9670 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Support Vector Machine | 0.9210 | 0.9945 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

### 4.2 Observations

| ML Model | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved 91.95% accuracy and the highest AUC score of 99.47%. Its overall performance was very close to SVM. |
| Decision Tree | Achieved 89.66% accuracy. It performed reasonably well but had lower scores than SVM, Logistic Regression, and KNN. |
| K-Nearest Neighbors | Achieved 91.55% accuracy and 91.57% weighted F1-score. It performed well, although it was slightly behind SVM and Logistic Regression. |
| Gaussian Naive Bayes | Achieved 76.30% accuracy, the lowest among the five models. Its AUC was 96.70%, indicating good class-score discrimination despite lower final classification accuracy. |
| Support Vector Machine | Achieved the highest accuracy of 92.10%, weighted F1-score of 92.11%, and MCC of 0.9045. It performed best based on these metrics. |

### Overall Winner for the Dataset

Support Vector Machine (SVM) was the best-performing model based on Accuracy (92.10%), weighted F1 Score (92.11%), and MCC (0.9045).

Logistic Regression achieved the highest AUC score (99.47%). The difference in accuracy between SVM and Logistic Regression was small, at approximately 0.15 percentage points.

These results are based on the selected stratified train-test split.

