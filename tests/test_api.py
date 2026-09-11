"""
Automated Test Suite — Smart Healthcare Diagnosis API
Covers smoke, inference, validation, observability, and UI tests.
"""

import pytest


# ─── Smoke Tests ───────────────────────────────────────────────────────────────

@pytest.mark.smoke
def test_root_endpoint(client):
    """Root endpoint returns 200 with valid navigation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"
    assert data["predict_url"] == "/predict"


@pytest.mark.smoke
def test_root_request_id_header(client):
    """Every response must include X-Request-ID for correlation tracing."""
    response = client.get("/")
    assert "x-request-id" in response.headers, "Missing X-Request-ID header"
    request_id = response.headers["x-request-id"]
    # Must be a valid 36-char UUID string (8-4-4-4-12)
    assert len(request_id) == 36
    assert request_id.count("-") == 4


# ─── Health / Observability Tests ─────────────────────────────────────────────

@pytest.mark.observability
def test_health_endpoint(client):
    """Health check confirms models and preprocessors are loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert "selected_model" in data
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.observability
def test_metrics_endpoint(client):
    """Metrics endpoint returns uptime, version, and timestamp."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], float)
    assert data["uptime_seconds"] >= 0.0
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.observability
def test_latency_header(client):
    """Response includes X-Process-Time-Ms header and latency is under 500ms SLA."""
    payload = {
        "Pregnancies": 2,
        "Glucose": 120.0,
        "BloodPressure": 70.0,
        "SkinThickness": 25.0,
        "Insulin": 90.0,
        "BMI": 26.5,
        "DiabetesPedigreeFunction": 0.40,
        "Age": 32,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "x-process-time-ms" in response.headers
    latency_ms = float(response.headers["x-process-time-ms"])
    assert latency_ms < 500.0, f"Latency {latency_ms:.2f}ms exceeds 500ms SLA"


# ─── UI Tests ─────────────────────────────────────────────────────────────────

@pytest.mark.ui
def test_dashboard_endpoint_returns_html(client):
    """Dashboard endpoint returns valid HTML with 200 status."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert b"<!DOCTYPE html>" in response.content or b"<!doctype html>" in response.content.lower()


@pytest.mark.ui
def test_ui_alias_endpoint(client):
    """/ui alias also returns the dashboard HTML."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


# ─── Inference Tests ──────────────────────────────────────────────────────────

@pytest.mark.inference
def test_predict_diabetic_patient(client):
    """High-risk patient vitals should predict Diabetes with Moderate or High Risk."""
    payload = {
        "Pregnancies": 6,
        "Glucose": 180.0,
        "BloodPressure": 88.0,
        "SkinThickness": 32.0,
        "Insulin": 190.0,
        "BMI": 38.2,
        "DiabetesPedigreeFunction": 0.85,
        "Age": 52,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"] == "Diabetes"
    assert data["confidence"].endswith("%")
    assert data["risk_level"] in ["Moderate Risk", "High Risk"]
    assert 0.0 <= data["probability"] <= 1.0
    assert "prediction_id" in data
    assert len(data["prediction_id"]) == 36  # UUID format


@pytest.mark.inference
def test_predict_healthy_patient(client):
    """Low-risk patient vitals should predict No Diabetes with Low Risk."""
    payload = {
        "Pregnancies": 1,
        "Glucose": 82.0,
        "BloodPressure": 68.0,
        "SkinThickness": 20.0,
        "Insulin": 55.0,
        "BMI": 21.4,
        "DiabetesPedigreeFunction": 0.18,
        "Age": 22,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"] == "No Diabetes"
    assert data["risk_level"] == "Low Risk"
    assert data["probability"] < 0.50


@pytest.mark.inference
def test_predict_male_patient_omitted_pregnancies(client):
    """Male patient with Pregnancies omitted should default to 0 and succeed."""
    payload = {
        "Gender": "male",
        "Glucose": 165.0,
        "BloodPressure": 85.0,
        "SkinThickness": 28.0,
        "Insulin": 150.0,
        "BMI": 32.5,
        "DiabetesPedigreeFunction": 0.55,
        "Age": 48,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["patient_gender"] == "male"
    assert data["prediction"] in ["Diabetes", "No Diabetes"]
    assert 0.0 <= data["probability"] <= 1.0


@pytest.mark.inference
def test_predict_male_patient_explicit_zero(client):
    """Male patient with explicit Pregnancies=0 should return a valid prediction."""
    payload = {
        "Gender": "male",
        "Pregnancies": 0,
        "Glucose": 90.0,
        "BloodPressure": 70.0,
        "SkinThickness": 19.0,
        "Insulin": 65.0,
        "BMI": 23.0,
        "DiabetesPedigreeFunction": 0.25,
        "Age": 30,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["patient_gender"] == "male"
    assert data["prediction"] == "No Diabetes"


@pytest.mark.inference
def test_predict_female_patient_explicit_gender(client):
    """Female patient with explicit Gender and Pregnancies should succeed."""
    payload = {
        "Gender": "female",
        "Pregnancies": 3,
        "Glucose": 130.0,
        "BloodPressure": 76.0,
        "SkinThickness": 26.0,
        "Insulin": 105.0,
        "BMI": 27.5,
        "DiabetesPedigreeFunction": 0.42,
        "Age": 36,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["patient_gender"] == "female"


@pytest.mark.inference
def test_predict_response_has_prediction_id(client):
    """Every prediction response must carry a unique prediction_id UUID."""
    payload = {
        "Glucose": 120.0,
        "BloodPressure": 75.0,
        "SkinThickness": 23.0,
        "Insulin": 85.0,
        "BMI": 26.0,
        "DiabetesPedigreeFunction": 0.33,
        "Age": 35,
    }
    r1 = client.post("/predict", json=payload)
    r2 = client.post("/predict", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Each call should produce a distinct prediction_id
    assert r1.json()["prediction_id"] != r2.json()["prediction_id"]


# ─── Validation Tests ─────────────────────────────────────────────────────────

@pytest.mark.validation
def test_validation_missing_required_field(client):
    """Omitting a required field (Glucose) should return 422 ValidationError."""
    payload = {
        "Pregnancies": 2,
        # Glucose is intentionally missing
        "BloodPressure": 80.0,
        "SkinThickness": 25.0,
        "Insulin": 100.0,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 30,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_type"] == "ValidationError"


@pytest.mark.validation
def test_validation_negative_age(client):
    """Negative age should fail Pydantic validation with 422."""
    payload = {
        "Pregnancies": 1,
        "Glucose": 110.0,
        "BloodPressure": 75.0,
        "SkinThickness": 22.0,
        "Insulin": 80.0,
        "BMI": 24.5,
        "DiabetesPedigreeFunction": 0.30,
        "Age": -5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("Age" in str(err) for err in data["details"])


@pytest.mark.validation
def test_validation_glucose_out_of_range(client):
    """Glucose value exceeding the allowed maximum (500) should return 422."""
    payload = {
        "Pregnancies": 1,
        "Glucose": 999.0,
        "BloodPressure": 75.0,
        "SkinThickness": 22.0,
        "Insulin": 80.0,
        "BMI": 24.5,
        "DiabetesPedigreeFunction": 0.30,
        "Age": 35,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.validation
def test_validation_wrong_data_type(client):
    """String passed for a numeric field should return 422."""
    payload = {
        "Pregnancies": "two",  # invalid type
        "Glucose": 110.0,
        "BloodPressure": 75.0,
        "SkinThickness": 22.0,
        "Insulin": 80.0,
        "BMI": 24.5,
        "DiabetesPedigreeFunction": 0.30,
        "Age": 35,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.validation
def test_validation_empty_body(client):
    """Empty request body should return 422 for all missing required fields."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.validation
def test_validation_male_with_pregnancies_greater_than_zero(client):
    """Male patient assigned Pregnancies > 0 should fail with 422."""
    payload = {
        "Gender": "male",
        "Pregnancies": 3,
        "Glucose": 140.0,
        "BloodPressure": 80.0,
        "SkinThickness": 25.0,
        "Insulin": 110.0,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 40,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert "Male patients cannot have Pregnancies > 0" in str(data["details"])
