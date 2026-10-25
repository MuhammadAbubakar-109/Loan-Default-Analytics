
"""
main.py
-------
Loan Default Prediction API

Features:
- Health and information routes
- User management stub
- Loan default prediction
- Model analysis: held-out metrics, ROC curve, confusion matrix,
  and classification report
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "model_metrics.json"

pipeline = None
feature_cols = None
model_artifact = {}


def _first_available(source, *keys):
    """Return the first non-None value found for the given keys."""
    if not isinstance(source, dict):
        return None

    for key in keys:
        value = source.get(key)

        if value is not None:
            return value

    return None


# ============================================================
# 2. HELPERS
# ============================================================
def _normalise_analysis(data):
    """
    Normalize saved held-out evaluation results for the frontend.

    This function only reads saved metrics. It does not retrain
    the model or modify results when a new applicant is predicted.
    """

    nested_model = data.get("model_metrics", {})
    if not isinstance(nested_model, dict):
        nested_model = {}

    nested_baseline = data.get("baseline_metrics", {})
    if not isinstance(nested_baseline, dict):
        nested_baseline = {}

    # Read model metrics from nested or flattened saved formats.
    model_metrics = {}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        model_metrics[key] = _first_available(
            nested_model,
            key,
        )

        if model_metrics[key] is None:
            model_metrics[key] = _first_available(
                data,
                f"model_{key}",
                key,
            )

    baseline_metrics = {}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        baseline_metrics[key] = _first_available(
            nested_baseline,
            key,
        )

        if baseline_metrics[key] is None:
            baseline_metrics[key] = _first_available(
                data,
                f"baseline_{key}",
            )

    # Confusion matrix: support dictionary and list formats.
    cm = data.get("confusion_matrix")
    if cm is None:
        cm = model_artifact.get("confusion_matrix")

    if isinstance(cm, dict):
        matrix = cm.get("matrix")
        if matrix is None:
            matrix = cm.get("values")

        confusion_matrix_data = {
            **cm,
            "labels": cm.get(
                "labels",
                ["non_default", "default"],
            ),
            "matrix": matrix,
        }

    elif isinstance(cm, list):
        confusion_matrix_data = {
            "labels": ["non_default", "default"],
            "matrix": cm,
        }

    else:
        confusion_matrix_data = None

    # IMPORTANT: the current saved JSON uses "roc_curve".
    # Older artifacts may use "roc_model".
    roc = data.get("roc_curve")
    if roc is None:
        roc = data.get("roc_model")
    if roc is None:
        roc = model_artifact.get("roc_curve")
    if roc is None:
        roc = model_artifact.get("roc_model")

    if isinstance(roc, dict):
        roc_auc = _first_available(
            roc,
            "roc_auc",
            "auc",
        )

        if roc_auc is None:
            roc_auc = model_metrics["roc_auc"]

        roc_curve = {
            "fpr": roc.get("fpr", []),
            "tpr": roc.get("tpr", []),
            "thresholds": roc.get("thresholds", []),
            "roc_auc": roc_auc,
        }

    else:
        roc_curve = None

    report = data.get("classification_report")
    if report is None:
        report = model_artifact.get("classification_report")

    return {
        "model": data.get("model", "LogisticRegression"),
        "evaluation": "held_out_test_set",
        "n_train": data.get("n_train"),
        "n_test": data.get("n_test"),
        "n_features": data.get(
            "n_features",
            len(feature_cols) if feature_cols else None,
        ),
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "confusion_matrix": confusion_matrix_data,
        "classification_report": report,
        "roc_curve": roc_curve,
    }



# ============================================================
# 3. APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, feature_cols, model_artifact

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python train_model.py` first."
        )

    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load model artifact: {exc}"
        ) from exc

    if not isinstance(artifact, dict):
        raise RuntimeError(
            "Invalid model artifact: expected a dictionary."
        )

    loaded_pipeline = artifact.get("pipeline")
    loaded_features = artifact.get("feature_cols")

    if loaded_pipeline is None or not loaded_features:
        raise RuntimeError(
            "model.pkl must contain 'pipeline' and 'feature_cols'."
        )

    pipeline = loaded_pipeline
    feature_cols = loaded_features
    model_artifact = artifact

    print(f"Model loaded: {len(feature_cols)} features")
    print(f"Metrics file: {METRICS_PATH}")
    yield

    pipeline = None
    feature_cols = None
    model_artifact = {}
    print("Model unloaded")


# ============================================================
# 4. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "Loan default prediction and held-out model evaluation API."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def strip_api_prefix(request, call_next):
    path = request.scope["path"]
    if path == "/api" or path.startswith("/api/"):
        request.scope["path"] = path[4:] or "/"
    return await call_next(request)

# ============================================================
# 5. REQUEST / RESPONSE SCHEMAS
# ============================================================

class LoanFeatures(BaseModel):
    """Loan input features matching the training dataset."""

    # Categorical features
    loan_limit: str = Field(..., examples=["cf"])
    Gender: str = Field(..., examples=["Male"])
    approv_in_adv: str = Field(..., examples=["pre"])
    loan_type: str = Field(..., examples=["conventional"])
    loan_purpose: str = Field(..., examples=["p1"])
    Credit_Worthiness: str = Field(..., examples=["I1"])
    open_credit: str = Field(..., examples=["op"])
    business_or_commercial: str = Field(..., examples=["b/c"])
    Neg_ammortization: str = Field(..., examples=["not_neg"])
    interest_only: str = Field(..., examples=["not_int"])
    lump_sum_payment: str = Field(..., examples=["not_lps"])
    construction_type: str = Field(..., examples=["sb"])
    occupancy_type: str = Field(..., examples=["pr"])
    Secured_by: str = Field(..., examples=["home"])
    total_units: str = Field(..., examples=["1U"])
    credit_type: str = Field(..., examples=["CIB"])
    co_applicant_credit_type: str = Field(..., examples=["CIB"])
    submission_of_application: str = Field(..., examples=["to_inst"])
    Region: str = Field(..., examples=["south"])
    Security_Type: str = Field(..., examples=["direct"])

    # Required numeric features
    loan_amount: float = Field(..., ge=0, examples=[250000])
    term: float = Field(..., ge=0, examples=[360])
    income: float = Field(..., ge=0, examples=[75000])
    Credit_Score: float = Field(..., ge=300, le=850, examples=[720])
    age: float = Field(..., ge=18, le=100, examples=[35])
    LTV: float = Field(..., ge=0, examples=[80.0])

    # Optional numeric features
    rate_of_interest: Optional[float] = Field(
        None, ge=0, examples=[4.5]
    )
    Interest_rate_spread: Optional[float] = Field(
        None, examples=[1.5]
    )
    Upfront_charges: Optional[float] = Field(
        None, ge=0, examples=[2500]
    )
    property_value: Optional[float] = Field(
        None, ge=0, examples=[300000]
    )
    dtir1: Optional[float] = Field(
        None, ge=0, examples=[35.0]
    )


class LoanResponse(BaseModel):
    prediction: int = Field(
        ..., description="0 = No Default, 1 = Default"
    )
    default_probability: float = Field(
        ..., description="Probability of default (0.0–1.0)"
    )
    risk_level: str = Field(
        ..., description="Low / Medium / High"
    )


class UserResponse(BaseModel):
    user_id: int
    name: str
    message: str


# ============================================================
# 6. INFO ROUTES
# ============================================================

@app.get("/", tags=["Info"])
def home():
    return {
        "message": "Loan Default Prediction API is running!",
        "docs": "/docs",
        "health": "/health",
        "model_analysis": "/api/model-analysis",
    }


@app.get("/health", tags=["Info"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "features_expected": len(feature_cols) if feature_cols else 0,
        "metrics_file_exists": METRICS_PATH.is_file(),
    }


# ============================================================
# 7. USER ROUTES
# ============================================================

@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    tags=["Users"],
)
def get_user(user_id: int):
    if user_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="user_id must be positive",
        )

    return {
        "user_id": user_id,
        "name": "Test User",
        "message": "User fetched successfully",
    }


# ============================================================
# 8. PREDICTION ROUTES
# ============================================================

def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.6:
        return "Medium"
    return "High"


@app.post(
    "/predict",
    response_model=LoanResponse,
    tags=["Prediction"],
)
def predict_default(features: LoanFeatures):
    """Predict default using the trained pipeline."""

    if pipeline is None or feature_cols is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    # Pydantic v2 uses model_dump(); v1 uses dict().
    if hasattr(features, "model_dump"):
        feature_data = features.model_dump()
    else:
        feature_data = features.dict()

    input_df = pd.DataFrame(
        [feature_data],
        columns=feature_cols,
    )

    try:
        prediction = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]

        # Use the actual positive-class index rather than assuming
        # that class 1 is always the second probability.
        classes = list(pipeline.classes_)
        if 1 not in classes:
            raise ValueError(
                "The trained model does not contain class 1."
            )

        probability = float(probabilities[classes.index(1)])

    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return LoanResponse(
        prediction=prediction,
        default_probability=round(probability, 4),
        risk_level=_risk_level(probability),
    )


def _load_metrics():
    """
    Load saved evaluation metrics from model_metrics.json.
    Fall back to the metrics stored in model.pkl if necessary.
    """
    import json

    # First, try the saved JSON metrics file.
    if METRICS_PATH.exists():
        try:
            with METRICS_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict) and data:
                return data

        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Could not load metrics JSON: {exc}")

    # Fall back to the loaded model artifact.
    artifact = model_artifact

    if isinstance(artifact, dict):
        # Support common metric formats saved in the model artifact.
        for key in ("evaluation_metrics", "metrics"):
            saved_metrics = artifact.get(key)

            if isinstance(saved_metrics, dict) and saved_metrics:
                return saved_metrics

        return artifact

    return {}


# ============================================================
# 9. MODEL ANALYSIS — ROC / METRICS / CONFUSION MATRIX
# ============================================================

@app.get(
    "/api/model-analysis",
    tags=["Model Analysis"],
    summary="Get held-out test evaluation and ROC curve",
)
def get_model_analysis():
    """
    Return evaluation data saved by train_model.py.

    This route does not retrain the model or evaluate training rows.
    """
    data = _load_metrics()
    result = _normalise_analysis(data)

    if not result["model_metrics"]:
        raise HTTPException(
            status_code=503,
            detail="No model evaluation metrics are available.",
        )

    return result


# Optional alias for clients that use a shorter route.
@app.get(
    "/model-analysis",
    tags=["Model Analysis"],
    include_in_schema=False,
)
def get_model_analysis_alias():
    return get_model_analysis()


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
