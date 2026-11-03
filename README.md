# Real-Time ML Inference REST API & Capstone

A container-ready FastAPI service that serves a trained Random Forest customer-churn classification pipeline.

> **Educational-use warning:** The source training file used for this demonstration contains only 12 records. Predictions and any performance conclusions are not reliable for real-world decisions. Do not use this model to make automated customer decisions. Obtain a larger, representative, authorized dataset and complete independent validation before any operational use.

## Project contents

```text
ML-Inference-API/
├── model/
│   └── champion_model.pkl
├── main.py
├── train_model.py
├── test_main.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

The trained preprocessing and classifier pipeline is stored together in `model/champion_model.pkl`. The raw customer CSV is intentionally excluded to avoid publishing customer-level records.

## Input features

| Field | Type | Validation | Example |
|---|---|---|---|
| `tenure_months` | integer | 0–600 | 5 |
| `support_tickets` | integer | 0–100 | 3 |
| `monthly_spend_inr` | number | 0–1,000,000 | 799 |
| `last_login_days` | integer | 0–3,650 | 15 |
| `plan_type` | string | Basic, Standard, Pro | Basic |

`customer_id` and the target field `churned` are not accepted as prediction inputs.

## Run locally (Windows PowerShell)

1. Install Python 3.11.
2. Open PowerShell inside this project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   python -m pip install pytest==9.0.2 httpx==0.28.1
   ```
5. Start the API:

   ```powershell
   uvicorn main:app --reload
   ```
6. Open `http://127.0.0.1:8000/docs` to test the interactive API documentation.

Health endpoint: `GET http://127.0.0.1:8000/health`

Prediction endpoint: `POST http://127.0.0.1:8000/predict`

## Example request

```json
{
  "tenure_months": 5,
  "support_tickets": 3,
  "monthly_spend_inr": 799,
  "last_login_days": 15,
  "plan_type": "Basic"
}
```

The response includes a binary prediction (`1` = churn, `0` = no churn), a readable label, class probabilities, model version, and an educational-use disclaimer. Exact probability values depend on the saved model artifact.

## API documentation

FastAPI automatically provides:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI schema: `/openapi.json`

## Run unit tests

```powershell
python -m pip install -r requirements-dev.txt
pytest -v
```

The tests cover the root route, health route, valid predictions, invalid plan categories, negative values, and unexpected input fields.

## Build and run with Docker

```powershell
docker build -t customer-churn-api .
docker run --rm -p 8000:8000 customer-churn-api
```

Then visit `http://localhost:8000/docs`.

## Retrain the model (optional)

The original training CSV is not distributed in this repository. If you have permission to use the original approved training data, place it locally and run:

```powershell
python train_model.py --csv path\to\customer-churn-training.csv
```

The CSV must contain `tenure_months`, `support_tickets`, `monthly_spend_inr`, `last_login_days`, `plan_type`, and `churned`. The script excludes `customer_id` by selecting only the approved feature columns. Never commit raw customer records to a public repository.

## Architecture

```text
Client / API caller
        |
        | POST /predict (JSON)
        v
FastAPI + Pydantic validation
        |
        v
Pandas one-row feature frame
        |
        v
Saved scikit-learn Pipeline
  - numeric imputation + scaling
  - categorical imputation + one-hot encoding
  - Random Forest classifier
        |
        v
JSON prediction + class probabilities
```

Docker packages the API and its saved model artifact into a reproducible container image. Dependencies are pinned in `requirements.txt`.

## Responsible-use notes

- The training sample has only 12 rows; this is a software integration demonstration, not a validated churn model.
- Verify data provenance, licence, consent/legal basis, label meaning, and feature timestamps before retraining or reuse.
- Keep raw customer data and customer identifiers private.
- Review predictions with a human; do not automate punitive or consequential customer actions.
- Evaluate on a larger, representative, independent dataset and monitor data quality and model drift before considering deployment.

## Deliverable checklist

- [x] Trained model artifact included
- [x] FastAPI `/predict` endpoint with JSON validation and probabilities
- [x] Dockerfile with pinned dependencies
- [x] Unit tests
- [x] README with setup, API examples, architecture, and responsible-use notes
- [ ] Push this folder to a public GitHub repository and submit its URL
