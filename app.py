from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET_COL = "target"
DATASET_INSTANCE_COUNT = 569
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="AIML Assignment 2 | Classifier Lab", page_icon="🧪", layout="wide")
logo_col, title_col, student_col = st.columns([1, 4, 2])
st.caption("M.Tech AIML 2026 · Machine Learning Assignment 2")
with student_col:
    st.markdown("**Name**  \nVaibhav Kaushik")
    st.markdown("**BITS ID**  \n2026AJ05812")
st.write(
    "Evaluate five saved classification pipelines on a labelled CSV test set. "
    "The app **does not retrain** models on uploaded data."
)

with st.sidebar:
    st.header("Evaluation setup")
    selected_model = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    compare_all = st.checkbox("Compare all five models", value=True)
    st.markdown("---")
    st.markdown("**Dataset:** UCI Wisconsin Diagnostic Breast Cancer")
    st.markdown("**Target column:** `target` (0 = malignant, 1 = benign)")
    st.markdown("Upload labelled test data containing all 30 model input features plus `target`.")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def get_expected_columns():
    model = load_model(MODEL_DIR / MODEL_FILES["Logistic Regression"])
    return list(model.feature_names_in_)

expected_columns = get_expected_columns()
st.subheader("Dataset Overview")
dataset_col, features_col = st.columns(2)
dataset_col.metric("Full dataset instances", f"{DATASET_INSTANCE_COUNT:,}")
features_col.metric("Model input features", len(expected_columns))
st.caption("These counts describe the complete UCI dataset. The uploaded CSV is the labelled test split used for evaluation.")

uploaded = st.file_uploader("Upload test dataset (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload `test_data.csv` from this project to run an evaluation.")
    with open(ROOT / "test_data.csv", "rb") as f:
        st.download_button("Download sample test_data.csv", f, file_name="test_data.csv", mime="text/csv")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if TARGET_COL not in data.columns:
    st.error(f"Missing required target column `{TARGET_COL}`.")
    st.stop()

missing = [c for c in expected_columns if c not in data.columns]
if missing:
    st.error("The CSV is missing required feature columns:")
    st.code(", ".join(missing))
    st.stop()

if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

X_eval = data[expected_columns].copy()
y_eval = data[TARGET_COL]
if y_eval.isna().any():
    st.error("The target column contains missing values. Please remove or fill them before evaluation.")
    st.stop()

try:
    y_eval = y_eval.astype(int)
except Exception:
    st.error("Target values must be encoded as integers 0 or 1 (0 = malignant, 1 = benign).")
    st.stop()

if not set(pd.unique(y_eval)).issubset({0, 1}):
    st.error("This app is configured for the binary target values 0 and 1.")
    st.stop()

st.subheader("Uploaded test data")
c1, c2, c3 = st.columns(3)
c1.metric("Test rows", f"{len(data):,}")
c2.metric("Input features", len(expected_columns))
c3.metric("Target classes", data[TARGET_COL].nunique())
with st.expander("Preview uploaded CSV"):
    st.dataframe(data.head(20), use_container_width=True)

def evaluate(model, X, y):
    predictions = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, list(model.classes_).index(1)]
    else:
        raw = model.decision_function(X)
        score = raw if list(model.classes_)[1] == 1 else -raw
    try:
        auc = roc_auc_score(y, score) if y.nunique() == 2 else np.nan
    except ValueError:
        auc = np.nan
    return predictions, {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }

model_names = list(MODEL_FILES.keys()) if compare_all else [selected_model]
results, predictions_by_model = {}, {}
for name in model_names:
    try:
        model = load_model(MODEL_DIR / MODEL_FILES[name])
        preds, values = evaluate(model, X_eval, y_eval)
        results[name] = values
        predictions_by_model[name] = preds
    except Exception as exc:
        st.error(f"Could not evaluate {name}: {exc}")
        st.stop()

st.subheader("Evaluation metrics")
metrics_df = pd.DataFrame(results).T
st.dataframe(
    metrics_df.style.format("{:.4f}"),
    use_container_width=True
)
if compare_all:
    best_name = metrics_df["F1"].idxmax()
    st.success(f"Highest F1 on this uploaded test set: **{best_name}** ({metrics_df.loc[best_name, 'F1']:.4f}).")
else:
    st.caption(f"Selected model: {selected_model}")

detail_name = selected_model if selected_model in predictions_by_model else model_names[0]
detail_preds = predictions_by_model[detail_name]
left, right = st.columns(2)
with left:
    st.subheader(f"Confusion matrix · {detail_name}")
    cm = confusion_matrix(y_eval, detail_preds, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader(f"Classification report · {detail_name}")
    st.code(classification_report(
        y_eval, detail_preds,
        labels=[0, 1],
        target_names=["Malignant (0)", "Benign (1)"],
        zero_division=0
    ))

pred_output = data.copy()
for name, preds in predictions_by_model.items():
    pred_output[f"prediction_{name.lower().replace(' ', '_').replace('-', '')}"] = preds
st.download_button(
    "Download predictions CSV",
    pred_output.to_csv(index=False).encode("utf-8"),
    file_name="classifier_predictions.csv",
    mime="text/csv"
)
st.caption("AUC is undefined and shown as NaN if the uploaded test set contains only one target class.")
