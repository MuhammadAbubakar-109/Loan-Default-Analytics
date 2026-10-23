
"""
train_model.py
──────────────
Loan Default Prediction — Training Pipeline

Reads:
    data/Loan_Default.csv

Saves:
    artifacts/model.pkl
    artifacts/model_metrics.json

Includes:
    - Held-out test evaluation
    - Logistic Regression and majority-class baseline
    - Accuracy, precision, recall, F1 and ROC-AUC
    - Complete ROC curve coordinates and thresholds
    - Confusion matrix (TN, FP, FN, TP)
    - Classification report
    - JSON-safe metrics for the FastAPI analysis endpoint

Usage:
    python train_model.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ═══════════════════════════════════════════════════════════
#  1. CONFIGURATION
# ═══════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "Loan_Default.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = OUTPUT_DIR / "model.pkl"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"

TARGET = "Status"
EXCLUDED_COLUMNS = []

TEST_SIZE = 0.20
RANDOM_STATE = 42
MINIMUM_ROWS = 100


# ═══════════════════════════════════════════════════════════
#  2. LOAD & CLEAN
# ═══════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Read CSV, validate target and remove invalid rows."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not in CSV. "
            f"Available: {list(df.columns)}"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]
    df = df.drop_duplicates().reset_index(drop=True)

    if len(df) < MINIMUM_ROWS:
        raise ValueError(
            f"Only {len(df)} usable rows "
            f"(need >= {MINIMUM_ROWS})."
        )

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "The target must contain both classes 0 and 1."
        )

    print(
        f"Loaded {len(df):,} rows x "
        f"{len(df.columns)} columns"
    )
    print(
        f"Class distribution: "
        f"0 = {(df[TARGET] == 0).sum():,} | "
        f"1 = {(df[TARGET] == 1).sum():,}"
    )

    return df


# ═══════════════════════════════════════════════════════════
#  3. FEATURE PREPARATION
# ═══════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame):
    """Split X/y and automatically detect feature types."""

    drop = [
        TARGET,
        *[c for c in EXCLUDED_COLUMNS if c in df.columns],
    ]

    X = df.drop(columns=drop).copy()
    y = df[TARGET].astype(int).copy()

    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after exclusions.")

    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    for col in categorical_cols:
        X[col] = X[col].apply(
            lambda value: (
                str(value) if pd.notna(value) else np.nan
            )
        )

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(
        f"Categorical features ({len(categorical_cols)}): "
        f"{categorical_cols}"
    )

    return X, y, numeric_cols, categorical_cols


# ═══════════════════════════════════════════════════════════
#  4. PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════

def build_preprocessor(numeric_cols, categorical_cols):
    """Impute and scale numeric data; impute and encode categories."""

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(
        transformers,
        remainder="drop",
    )


# ═══════════════════════════════════════════════════════════
#  5. EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, y_prob) -> dict:
    """Compute binary classification metrics."""

    return {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)), 4
        ),
        "precision": round(
            float(precision_score(
                y_true, y_pred, zero_division=0
            )), 4
        ),
        "recall": round(
            float(recall_score(
                y_true, y_pred, zero_division=0
            )), 4
        ),
        "f1": round(
            float(f1_score(
                y_true, y_pred, zero_division=0
            )), 4
        ),
        "roc_auc": round(
            float(roc_auc_score(y_true, y_prob)), 4
        ),
    }


def build_roc_model(y_true, y_prob) -> dict:
    """
    Generate the complete held-out-test ROC curve.

    Positive class: 1 = loan default.
    All coordinates and thresholds are saved for frontend plotting.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob,
        pos_label=1,
        drop_intermediate=False,
    )

    return {
        "positive_class": 1,
        "positive_label": "default",
        "negative_label": "non_default",
        "fpr": [float(v) for v in fpr],
        "tpr": [float(v) for v in tpr],
        "thresholds": [
            None if not np.isfinite(v) else float(v)
            for v in thresholds
        ],
        "roc_auc": round(
            float(roc_auc_score(y_true, y_prob)), 4
        ),
        "n_points": int(len(fpr)),
        "x_axis": "False Positive Rate",
        "y_axis": "True Positive Rate",
        "evaluation_set": "held_out_test",
    }


def build_confusion_matrix(y_true, y_pred) -> dict:
    """Return the matrix and named counts in a JSON-friendly format."""

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    return {
        "labels": ["non_default", "default"],
        "matrix": cm.tolist(),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }


# ═══════════════════════════════════════════════════════════
#  6. TRAIN → EVALUATE → SAVE
# ═══════════════════════════════════════════════════════════

def train():
    print("\n" + "=" * 55)
    print("  LOAN DEFAULT PREDICTION — TRAINING")
    print("=" * 55 + "\n")

    # ── data ─────────────────────────────────────────────

    df = load_data()
    X, y, num_cols, cat_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\nTrain rows: {len(X_train):,} | "
        f"Test rows: {len(X_test):,}"
    )

    # ── pipeline ─────────────────────────────────────────

    preprocessor = build_preprocessor(
        num_cols, cat_cols
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
            C=10.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    # ── evaluate trained model on held-out test data ─────

    y_pred = pipeline.predict(X_test)

    # Explicitly select probability for class 1 (default).
    classifier = pipeline.named_steps["classifier"]
    positive_index = list(classifier.classes_).index(1)

    y_prob = pipeline.predict_proba(X_test)[:, positive_index]

    model_metrics = evaluate(
        y_test, y_pred, y_prob
    )

    roc_model = build_roc_model(
        y_test, y_prob
    )

    cm_data = build_confusion_matrix(
        y_test, y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["non_default", "default"],
        zero_division=0,
        output_dict=True,
    )

    # ── evaluate baseline on the same held-out test data ─

    baseline = DummyClassifier(
        strategy="most_frequent"
    )
    baseline.fit(X_train, y_train)

    b_pred = baseline.predict(X_test)

    baseline_classifier = baseline
    baseline_positive_index = list(
        baseline_classifier.classes_
    ).index(1)

    b_prob = baseline.predict_proba(X_test)[
        :, baseline_positive_index
    ]

    baseline_metrics = evaluate(
        y_test, b_pred, b_prob
    )

    baseline_cm = build_confusion_matrix(
        y_test, b_pred
    )

    baseline_report = classification_report(
        y_test,
        b_pred,
        labels=[0, 1],
        target_names=["non_default", "default"],
        zero_division=0,
        output_dict=True,
    )

    baseline_roc = build_roc_model(
        y_test, b_prob
    )

    # ── build model artifact ─────────────────────────────

    artifact = {
        "pipeline": pipeline,
        "feature_cols": X.columns.tolist(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "target": TARGET,
        "positive_class": 1,
        "metrics": model_metrics,
        "roc_model": roc_model,
        "confusion_matrix": cm_data,
        "classification_report": report,
        "baseline_metrics": baseline_metrics,
        "baseline_roc_model": baseline_roc,
        "baseline_confusion_matrix": baseline_cm,
        "baseline_classification_report": baseline_report,
        "evaluation": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(X.shape[1]),
            "positive_class": "default (1)",
            "method": "stratified held-out test split",
        },
    }

    # ── save model artifact ──────────────────────────────

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(artifact, MODEL_PATH)

    # ── save metrics artifact for FastAPI/frontend ───────

    full_metrics = {
        "model": "LogisticRegression",
        "target": TARGET,
        "positive_class": 1,
        "evaluation": artifact["evaluation"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),

        **{
            f"model_{key}": value
            for key, value in model_metrics.items()
        },
        **{
            f"baseline_{key}": value
            for key, value in baseline_metrics.items()
        },

        # Nested objects for analysis API consumers.
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "roc_model": roc_model,
        "baseline_roc_model": baseline_roc,
        "confusion_matrix": cm_data,
        "classification_report": report,
        "baseline_confusion_matrix": baseline_cm,
        "baseline_classification_report": baseline_report,
    }

    METRICS_PATH.write_text(
        json.dumps(
            full_metrics,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # ── report ──────────────────────────────────────────

    print("\n" + "=" * 55)
    print("  RESULTS — HELD-OUT TEST SET")
    print("=" * 55)

    print("\nTrained Model:")
    for key, value in model_metrics.items():
        print(f"  {key:<12} {value:.4f}")

    print("\nBaseline (most_frequent):")
    for key, value in baseline_metrics.items():
        print(f"  {key:<12} {value:.4f}")

    print("\nConfusion Matrix:")
    print(
        f"  TN={cm_data['true_negative']:>6}  "
        f"FP={cm_data['false_positive']:>6}"
    )
    print(
        f"  FN={cm_data['false_negative']:>6}  "
        f"TP={cm_data['true_positive']:>6}"
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["non_default", "default"],
        zero_division=0,
    ))

    print(
        f"\nROC-AUC: {roc_model['roc_auc']:.4f}"
        f" | ROC points: {roc_model['n_points']}"
    )

    print(f"\nModel   -> {MODEL_PATH}")
    print(f"Metrics -> {METRICS_PATH}")
    print("\nTraining complete.\n")

    return artifact


# ═══════════════════════════════════════════════════════════
#  7. ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
