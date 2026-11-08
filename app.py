
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Dry Bean Classification", layout="wide")

st.title("Dry Bean Classification")
st.write("Evaluate five machine learning models on the Dry Bean dataset.")

MODEL_PATH = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

model_name = st.selectbox(
    "Select a Classification Model",
    list(model_files.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    if "Class" not in test_df.columns:
        st.error("The uploaded CSV must contain the target column 'Class'.")
    else:
        X_test = test_df.drop(columns=["Class"])
        y_test = test_df["Class"]

        model_file = os.path.join(MODEL_PATH, model_files[model_name])

        if not os.path.exists(model_file):
            st.error(f"Model file not found: {model_file}")
        else:
            model = joblib.load(model_file)

            y_pred = model.predict(X_test)

            st.subheader("Evaluation Metrics")

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            mcc = matthews_corrcoef(y_test, y_pred)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                auc = roc_auc_score(
                    y_test,
                    y_prob,
                    multi_class="ovr",
                    average="macro"
                )
            else:
                auc = None

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Recall", f"{recall:.4f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("F1 Score", f"{f1:.4f}")
            col5.metric("MCC", f"{mcc:.4f}")
            col6.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")

            st.subheader("Confusion Matrix")

            labels = sorted(y_test.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            cm_df = pd.DataFrame(
                cm,
                index=labels,
                columns=labels
            )

            st.dataframe(cm_df)

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                zero_division=0,
                output_dict=True
            )

            st.dataframe(pd.DataFrame(report).transpose())
