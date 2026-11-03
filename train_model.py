"""Train and save the customer-churn inference pipeline.

Usage: python train_model.py --csv customer-churn-training.csv
The source CSV is intentionally not included in the public repository.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure_months", "support_tickets", "monthly_spend_inr", "last_login_days"]
CATEGORICAL_FEATURES = ["plan_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churned"


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    classifier = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to approved training CSV")
    parser.add_argument("--output", default="model/champion_model.pkl")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    X = df[FEATURES].copy()
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}) or y.nunique() < 2:
        raise ValueError("The churned target must contain both binary classes 0 and 1.")
    model = build_pipeline()
    model.fit(X, y)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": FEATURES, "target": TARGET,
                 "model_name": "Customer Churn Random Forest", "model_version": "1.0.0"}, output)
    print(f"Saved trained pipeline to {output.resolve()}")
    print("IMPORTANT: this model is for educational demonstration only.")

if __name__ == "__main__":
    main()
