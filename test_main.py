from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
VALID_PAYLOAD = {
    "tenure_months": 5,
    "support_tickets": 3,
    "monthly_spend_inr": 799,
    "last_login_days": 15,
    "plan_type": "Basic",
}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_predict_endpoint():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] in ("Churn", "No churn")
    assert "churn" in body["probabilities"]
    assert "no_churn" in body["probabilities"]
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6

def test_invalid_plan_is_rejected():
    payload = {**VALID_PAYLOAD, "plan_type": "Premium"}
    assert client.post("/predict", json=payload).status_code == 422

def test_negative_tenure_is_rejected():
    payload = {**VALID_PAYLOAD, "tenure_months": -1}
    assert client.post("/predict", json=payload).status_code == 422

def test_extra_fields_are_rejected():
    payload = {**VALID_PAYLOAD, "customer_id": "C999"}
    assert client.post("/predict", json=payload).status_code == 422
