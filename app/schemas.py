"""
Pydantic Schemas for Request and Response Models
Smart Healthcare Diagnosis API
"""

from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class PatientVitalsInput(BaseModel):
    """Input payload containing patient clinical and demographic vitals."""

    Pregnancies: int = Field(
        ...,
        ge=0,
        le=25,
        description="Number of times pregnant",
        examples=[2],
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

    @field_validator("Age", mode="before")
    @classmethod
    def validate_age_positive(cls, v):
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError("Age cannot be negative")
        return v

    @classmethod
    def model_validate(cls, obj: Any, *args, **kwargs):
        if isinstance(obj, dict):
            key_map = {
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
                    "Pregnancies": 2,
                    "Glucose": 150.0,
                    "BloodPressure": 85.0,
                    "SkinThickness": 30.0,
                    "Insulin": 120.0,
                    "BMI": 33.5,
                    "DiabetesPedigreeFunction": 0.45,
                    "Age": 45,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Structured response returned by the diagnosis endpoint."""

    status: str = Field(default="success", description="Status of prediction request")
    prediction: str = Field(..., description="Predicted diagnosis ('Diabetes' or 'No Diabetes')")
    confidence: str = Field(..., description="Prediction confidence formatted as percentage string")
    probability: float = Field(..., description="Calculated probability of positive outcome (0.0 to 1.0)")
    risk_level: str = Field(..., description="Categorized clinical risk level: Low Risk, Moderate Risk, or High Risk")
    model_used: str = Field(..., description="Machine learning algorithm name used for inference")
    timestamp: str = Field(..., description="Inference execution timestamp in ISO format")


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    selected_model: str
    version: str
    timestamp: str


class RootResponse(BaseModel):
    """Root endpoint response."""

    message: str
    version: str
    docs_url: str
    health_url: str
    predict_url: str
