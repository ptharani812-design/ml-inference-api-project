"""FastAPI application for customer churn inference."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).parent / "model" / "champion_model.pkl"
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Educational REST API serving a trained Random Forest pipeline.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenure_months: int = Field(ge=0, le=600, examples=[5])
    support_tickets: int = Field(ge=0, le=100, examples=[3])
    monthly_spend_inr: float = Field(ge=0, le=1_000_000, examples=[799])
    last_login_days: int = Field(ge=0, le=3650, examples=[15])
    plan_type: Literal["Basic", "Standard", "Pro"] = Field(examples=["Basic"])

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: Literal["No churn", "Churn"]
    probabilities: dict[str, float]
    model_name: str
    model_version: str
    disclaimer: str


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Run train_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    return bundle

try:
    MODEL_BUNDLE = load_model()
except Exception as exc:  # Fail clearly at startup rather than returning misleading predictions.
    MODEL_BUNDLE = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None

@app.get("/", tags=["General"])
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs", "health": "/health"}

@app.get("/health", tags=["General"])
def health():
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {MODEL_LOAD_ERROR}")
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: PredictionRequest):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {MODEL_LOAD_ERROR}")
    try:
        row = pd.DataFrame([payload.model_dump()])[MODEL_BUNDLE["features"]]
        pipeline = MODEL_BUNDLE["pipeline"]
        predicted_class = int(pipeline.predict(row)[0])
        raw_probabilities = pipeline.predict_proba(row)[0]
        classes = pipeline.named_steps["classifier"].classes_
        probabilities = {("churn" if int(c) == 1 else "no_churn"): float(p)
                         for c, p in zip(classes, raw_probabilities)}
        return PredictionResponse(
            prediction=predicted_class,
            prediction_label="Churn" if predicted_class == 1 else "No churn",
            probabilities=probabilities,
            model_name=MODEL_BUNDLE["model_name"],
            model_version=MODEL_BUNDLE["model_version"],
            disclaimer="Educational demonstration only. Do not use for automated customer decisions.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
