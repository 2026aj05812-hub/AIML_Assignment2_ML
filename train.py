import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
TARGET_COLUMN = "target"


def build_models():
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DecisionTreeClassifier(random_state=42)),
            ]
        ),
        "knn": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier()),
            ]
        ),
        "naive_bayes": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GaussianNB()),
            ]
        ),
        "svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", SVC(probability=True, random_state=42)),
            ]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the five classifier pipelines.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for model/ and test_data.csv outputs (defaults to the project folder).",
    )
    args = parser.parse_args()

    dataset = pd.read_csv(ROOT / "data" / "breast_cancer_full_dataset.csv")
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Dataset must contain a `{TARGET_COLUMN}` target column.")

    features = dataset.drop(columns=[TARGET_COLUMN])
    target = dataset[TARGET_COLUMN]
    if set(target.dropna().unique()) != {0, 1}:
        raise ValueError("The target column must contain both binary labels 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        positive_class_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_test)[:, positive_class_index]
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(y_test, probabilities),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "MCC": matthews_corrcoef(y_test, predictions),
            }
        )
        joblib.dump(model, model_dir / f"{model_name}.joblib")

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(args.output_dir / "test_data.csv", index=False)
    print(pd.DataFrame(results).set_index("Model").to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved five models to {model_dir}")
    print(f"Saved hold-out data to {args.output_dir / 'test_data.csv'}")


if __name__ == "__main__":
    main()