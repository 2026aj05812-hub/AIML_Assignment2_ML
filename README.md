# AIML Machine Learning Assignment 2 — Classifier Lab

## a. Problem statement
Build an interactive Streamlit application that loads pre-trained classification models, accepts a labelled test CSV, and reports Accuracy, AUC, Precision, Recall, F1, MCC, a confusion matrix, and a classification report. The app evaluates uploaded data without retraining models.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); `data/breast_cancer_full_dataset.csv` is included in this repository.
- **Task:** Binary classification of breast-cancer diagnostic observations.
- **Size:** 569 observations and 30 numeric input features.
- **Target:** `target` — `0` = malignant, `1` = benign.
- **Split:** 75% training / 25% test, stratified, `random_state=42`.
- **Files:** `test_data.csv` is the fixed labelled test set; `data/breast_cancer_full_dataset.csv` is included for reference and local reproducibility.

## c. GitHub Repository Link
**[AIML Assignment 2 GitHub Repository](https://github.com/2026aj05812-hub/AIML_Assignment2_ML)**

## d. Models used

All five models use the same training/test split. Preprocessing is stored in each saved scikit-learn pipeline so the same transformations are applied during inference.

|                        |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9860 | 0.9977 |      0.9889 |   0.9889 | 0.9889 | 0.9700 |
| Decision Tree          |     0.9231 | 0.9234 |      0.9540 |   0.9222 | 0.9379 | 0.8378 |
| K-Nearest Neighbors    |     0.9790 | 0.9845 |      0.9677 |   1.0000 | 0.9836 | 0.9555 |
| Gaussian Naive Bayes   |     0.9371 | 0.9893 |      0.9263 |   0.9778 | 0.9514 | 0.8650 |
| Support Vector Machine |     0.9790 | 0.9969 |      0.9888 |   0.9778 | 0.9832 | 0.9553 |

### Observations on this hold-out split
- **Logistic Regression:** Accuracy 0.986, AUC 0.998, F1 0.989, MCC 0.970 on the fixed 25% hold-out split. Interpret results alongside precision/recall and the class balance; results can change with a different split.
- **Decision Tree:** Accuracy 0.923, AUC 0.923, F1 0.938, MCC 0.838 on the fixed 25% hold-out split. Interpret results alongside precision/recall and the class balance; results can change with a different split.
- **K-Nearest Neighbors:** Accuracy 0.979, AUC 0.984, F1 0.984, MCC 0.955 on the fixed 25% hold-out split. Interpret results alongside precision/recall and the class balance; results can change with a different split.
- **Gaussian Naive Bayes:** Accuracy 0.937, AUC 0.989, F1 0.951, MCC 0.865 on the fixed 25% hold-out split. Interpret results alongside precision/recall and the class balance; results can change with a different split.
- **Support Vector Machine:** Accuracy 0.979, AUC 0.997, F1 0.983, MCC 0.955 on the fixed 25% hold-out split. Interpret results alongside precision/recall and the class balance; results can change with a different split.

**Overall winner for this split:** Logistic Regression by F1 score (0.9889). This is only a split-specific result, not a guarantee of generalization; discuss other metrics and the cost of false positives/false negatives in your final report.

## Run locally
From a terminal, change to the project folder and create/activate a Python 3.11 or newer environment.

**macOS / Linux**
```bash
cd AIML_assignment2_streamlit_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

**Windows PowerShell**
```powershell
cd AIML_assignment2_streamlit_project
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) in the terminal; open it in a browser. To reproduce the saved models and test split before starting the app, run `python train.py`. This writes five pipelines under `model/` and recreates `test_data.csv`. Upload `test_data.csv` in the app to check the metrics and confusion matrix.

## Deploy on Streamlit Community Cloud
**Live app:** [Open the deployed Streamlit app](https://majorvaibhavkaushikml.streamlit.app/)

Docker is not required for Streamlit Community Cloud. It installs the packages listed in `requirements.txt` and runs the app entry point you select.

1. Push the project to GitHub, including `app.py`, `requirements.txt`, and the `model/` files used by the app.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, set the app file path to `app.py`, then deploy.
4. Copy the live app URL for your submission PDF.

Use a Dockerfile only if you need custom system packages or runtime configuration that Community Cloud's standard Python environment does not provide.

## Repository structure
```text
aiml_assignment2_streamlit/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── bits-pilani-logo.png
├── test_data.csv
├── data/
│   └── breast_cancer_full_dataset.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## Important submission checklist
- Replace the GitHub placeholder with your actual repository URL.
- Deploy the app and add the live Streamlit URL to your assignment PDF.
- Run it on BITS Virtual Lab and capture your own screenshot as required by the assignment.
- Include the final README content in the submitted PDF.
- Review the app, code, model results, and observations yourself. The assignment brief explicitly warns against direct copy-paste submissions; use this as a learning scaffold and customize/understand it before submission.
