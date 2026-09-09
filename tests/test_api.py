"""
Automated Test Suite for Smart Healthcare Diagnosis API
Covers functional tests, validation boundaries, edge cases, and latency.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns 200 and valid links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"
    assert data["predict_url"] == "/predict"


def test_health_endpoint():
    """Test health check confirms models and preprocessors are loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert "selected_model" in data


def test_predict_diabetic_patient():
    """Test prediction for typical high-risk diabetic patient vitals."""
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
    assert "confidence" in data
    assert data["confidence"].endswith("%")
    assert data["risk_level"] in ["Moderate Risk", "High Risk"]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_healthy_patient():
    """Test prediction for typical low-risk healthy patient vitals."""
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


def test_validation_missing_field():
    """Test validation fails with 422 if a required field is missing."""
    payload = {
        "Pregnancies": 2,
        # Glucose is missing
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


def test_validation_negative_age():
    """Test validation fails with 422 when age is negative."""
    payload = {
        "Pregnancies": 1,
        "Glucose": 110.0,
        "BloodPressure": 75.0,
        "SkinThickness": 22.0,
        "Insulin": 80.0,
        "BMI": 24.5,
        "DiabetesPedigreeFunction": 0.30,
        "Age": -5,  # Invalid
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("Age" in str(err) for err in data["details"])


def test_validation_out_of_range_glucose():
    """Test validation fails with 422 when glucose is unrealistically high."""
    payload = {
        "Pregnancies": 1,
        "Glucose": 999.0,  # Exceeds max 400
        "BloodPressure": 75.0,
        "SkinThickness": 22.0,
        "Insulin": 80.0,
        "BMI": 24.5,
        "DiabetesPedigreeFunction": 0.30,
        "Age": 35,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_validation_invalid_datatype():
    """Test validation fails with 422 when string is passed for float/int."""
    payload = {
        "Pregnancies": "two",  # Invalid
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


def test_validation_empty_body():
    """Test validation fails with 422 on empty body."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_latency_header():
    """Test that response includes process time header and latency is < 500ms."""
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
    assert latency_ms < 500.0, f"Latency {latency_ms}ms exceeds 500ms SLA"
