from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
)

# 1. Paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Loan_Default.csv"
OUT = BASE / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# 2. Load and clean data
df = pd.read_csv(DATA)
df["Status"] = pd.to_numeric(df["Status"], errors="coerce")
df = df.dropna(subset=["Status"])
df = df[df["Status"].isin([0, 1])].drop_duplicates()

X = df.drop(columns="Status")
y = df["Status"].astype(int)
feature_cols = X.columns.tolist()

# 3. Numeric / categorical columns
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()

# 4. Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num),
    ("cat", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), cat),
])

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train
model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)),
])
model.fit(X_train, y_train)


def get_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


# 7. Evaluate model on the held-out test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
model_metrics = get_metrics(y_test, y_pred, y_prob)

# 8. Baseline (always predicts the most common class)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
base_pred = base.predict(X_test)
base_prob = base.predict_proba(X_test)[:, 1]
baseline_metrics = get_metrics(y_test, base_pred, base_prob)

# 9. Confusion matrix and ROC curve
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # [[TN, FP], [FN, TP]]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Keep ~200 points for plotting; cap the inf threshold (invalid in JSON)
idx = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
fpr, tpr, thresholds = fpr[idx], tpr[idx], np.minimum(thresholds[idx], 1.0)

evaluation_results = {
    "model": "LogisticRegression",
    "evaluation": "held_out_test_set",
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": len(feature_cols),
    "model_metrics": model_metrics,
    "baseline_metrics": baseline_metrics,
    "confusion_matrix": {
        "labels": ["non_default", "default"],
        "matrix": cm.tolist(),
    },
    "classification_report": classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["non_default", "default"],
        output_dict=True, zero_division=0,
    ),
    "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": model_metrics["roc_auc"],
    },
}

# 10. Save model
model_path = OUT / "model.joblib"
joblib.dump(
    {
        "pipeline": model,
        "feature_cols": feature_cols,
        "target": "Status",
        "metrics": model_metrics,
    },
    model_path,
)

# 11. Save metrics
metrics_path = OUT / "model_metrics.json"
metrics_path.write_text(
    json.dumps(evaluation_results, indent=2, allow_nan=False),
    encoding="utf-8",
)

# 12. Print results
print("\nLogistic Regression Results:")
print(json.dumps(model_metrics, indent=2))
print("\nBaseline Results:")
print(json.dumps(baseline_metrics, indent=2))
print("\nConfusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())
print(f"\nModel saved:   {model_path.resolve()}")
print(f"Metrics saved: {metrics_path.resolve()}")
print("Training complete!")