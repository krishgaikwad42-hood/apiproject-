"""
Pydantic Schemas — Request and Response Models
Smart Healthcare Diagnosis API
"""

from typing import Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Request Schema ────────────────────────────────────────────────────────────

class PatientVitalsInput(BaseModel):
    """Input payload containing patient clinical and demographic vitals."""

    Gender: Optional[Literal["female", "male", "other"]] = Field(
        default="female",
        description="Biological sex / gender of the patient ('male', 'female', or 'other')",
        examples=["male"],
    )
    Pregnancies: int = Field(
        default=0,
        ge=0,
        le=25,
        description="Number of times pregnant (automatically 0 for male patients)",
        examples=[0],
    )
    Glucose: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Plasma glucose concentration (2 hours in an oral glucose tolerance test)",
        examples=[150.0],
    )
    BloodPressure: float = Field(
        ...,
        ge=0.0,
        le=250.0,
        description="Diastolic blood pressure (mm Hg)",
        examples=[85.0],
    )
    SkinThickness: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Triceps skin fold thickness (mm)",
        examples=[30.0],
    )
    Insulin: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="2-Hour serum insulin (mu U/ml)",
        examples=[120.0],
    )
    BMI: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Body mass index (weight in kg / (height in m)^2)",
        examples=[33.5],
    )
    DiabetesPedigreeFunction: float = Field(
        ...,
        ge=0.0,
        le=3.5,
        description="Diabetes pedigree function (genetic score based on family history)",
        examples=[0.45],
    )
    Age: int = Field(
        ...,
        ge=1,
        le=130,
        description="Age of the patient in years",
        examples=[45],
    )

    @field_validator("Gender", mode="before")
    @classmethod
    def normalize_gender(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ["m", "male", "man", "boy"]:
                return "male"
            if v_clean in ["f", "female", "woman", "girl"]:
                return "female"
            return v_clean
        return v

    @field_validator("Age", mode="before")
    @classmethod
    def validate_age_positive(cls, v):
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError("Age cannot be negative")
        return v

    @model_validator(mode="after")
    def validate_gender_and_pregnancies(self):
        if self.Gender == "male" and self.Pregnancies > 0:
            raise ValueError(
                "Male patients cannot have Pregnancies > 0. Biological pregnancy count must be 0 for male patients."
            )
        return self

    @classmethod
    def model_validate(cls, obj: Any, *args, **kwargs):
        if isinstance(obj, dict):
            key_map = {
                "gender": "Gender",
                "sex": "Gender",
                "pregnancies": "Pregnancies",
                "glucose": "Glucose",
                "bloodpressure": "BloodPressure",
                "blood_pressure": "BloodPressure",
                "skinthickness": "SkinThickness",
                "skin_thickness": "SkinThickness",
                "insulin": "Insulin",
                "bmi": "BMI",
                "diabetespedigreefunction": "DiabetesPedigreeFunction",
                "diabetes_pedigree": "DiabetesPedigreeFunction",
                "diabetes_pedigree_function": "DiabetesPedigreeFunction",
                "age": "Age",
            }
            normalized = {}
            for k, val in obj.items():
                norm_key = key_map.get(k.lower(), k)
                normalized[norm_key] = val
            obj = normalized
        return super().model_validate(obj, *args, **kwargs)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Gender": "male",
                    "Pregnancies": 0,
                    "Glucose": 140.0,
                    "BloodPressure": 80.0,
                    "SkinThickness": 25.0,
                    "Insulin": 115.0,
                    "BMI": 29.5,
                    "DiabetesPedigreeFunction": 0.38,
                    "Age": 42,
                },
                {
                    "Gender": "female",
                    "Pregnancies": 2,
                    "Glucose": 150.0,
                    "BloodPressure": 85.0,
                    "SkinThickness": 30.0,
                    "Insulin": 120.0,
                    "BMI": 33.5,
                    "DiabetesPedigreeFunction": 0.45,
                    "Age": 45,
                },
            ]
        }
    }


# ─── Response Schemas ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Structured response returned by the /predict diagnosis endpoint."""

    status: str = Field(
        default="success",
        description="Status of the prediction request ('success' or 'error')",
    )
    prediction_id: str = Field(
        ...,
        description="Unique UUID assigned to this prediction run for traceability",
    )
    prediction: str = Field(
        ...,
        description="Predicted diagnosis label: 'Diabetes' or 'No Diabetes'",
    )
    confidence: str = Field(
        ...,
        description="Model confidence in the prediction, formatted as a percentage string (e.g. '87.3%')",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw probability of a positive (Diabetes) outcome, in range [0.0, 1.0]",
    )
    risk_level: str = Field(
        ...,
        description="Stratified clinical risk level: 'Low Risk', 'Moderate Risk', or 'High Risk'",
    )
    model_used: str = Field(
        ...,
        description="Name of the machine learning algorithm used for this inference",
    )
    patient_gender: Optional[str] = Field(
        default=None,
        description="Reported patient biological sex: 'male' or 'female'",
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 timestamp of when the inference was executed (UTC)",
    )


class HealthResponse(BaseModel):
    """System health check response with model and service status."""

    status: str = Field(
        ...,
        description="Overall service status: 'healthy' or 'degraded'",
    )
    model_loaded: bool = Field(
        ...,
        description="Indicates whether the ML model artifact is loaded into memory",
    )
    preprocessor_loaded: bool = Field(
        ...,
        description="Indicates whether the scaler and imputer preprocessors are loaded",
    )
    selected_model: str = Field(
        ...,
        description="Name of the ML model selected during training and loaded at runtime",
    )
    version: str = Field(
        ...,
        description="API semantic version string (e.g. '1.0.0')",
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 timestamp of when the health check was executed",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "preprocessor_loaded": True,
                    "selected_model": "Decision Tree Classifier",
                    "version": "1.0.0",
                    "timestamp": "2026-09-10T08:00:00.000000",
                }
            ]
        }
    }


class RootResponse(BaseModel):
    """Root endpoint response with service metadata and navigation URLs."""

    message: str = Field(
        ...,
        description="Welcome message describing the service",
    )
    version: str = Field(
        ...,
        description="API semantic version string",
    )
    docs_url: str = Field(
        ...,
        description="Path to the interactive Swagger UI documentation",
    )
    health_url: str = Field(
        ...,
        description="Path to the system health check endpoint",
    )
    predict_url: str = Field(
        ...,
        description="Path to the disease prediction (inference) endpoint",
    )


class MetricsResponse(BaseModel):
    """Operational metrics snapshot for monitoring and observability."""

    uptime_seconds: float = Field(
        ...,
        description="Seconds elapsed since the API process started",
    )
    version: str = Field(
        ...,
        description="API semantic version string",
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 timestamp of when metrics were captured",
    )
