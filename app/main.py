"""
FastAPI Main Application
Smart Healthcare Diagnosis API (Disease Prediction System)
Exposes RESTful endpoints for early disease risk detection.
"""

import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure base directory in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.schemas import (
    PatientVitalsInput,
    PredictionResponse,
    HealthResponse,
    RootResponse,
)
from app.predict import inference_service
from app.logger import logger, RequestLoggingMiddleware

API_VERSION = "1.0.0"

app = FastAPI(
    title="Smart Healthcare Diagnosis API",
    description="""
A production-grade Machine Learning REST API for early clinical disease diagnosis.
Trained on clinical health indicators (Glucose, BMI, Blood Pressure, Insulin, Age, etc.)
to provide real-time risk classification and confidence scores.

### Endpoints
* **GET /**: Welcome message & service metadata.
* **GET /health**: System health check & ML model status.
* **POST /predict**: Predicts diabetes risk given patient vitals.
    """,
    version=API_VERSION,
    contact={
        "name": "Krish",
        "url": "https://github.com/krish",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for cross-origin web app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom timing and request logger middleware
app.add_middleware(RequestLoggingMiddleware)


# Custom Error Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats validation errors into clean, informative JSON response."""
    error_details = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        error_details.append({
            "field": field,
            "message": err.get("msg"),
            "type": err.get("type"),
        })
    logger.warning(f"Validation failure on {request.url.path}: {error_details}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "ValidationError",
            "message": "Invalid input vitals provided. Please check the ranges and field types.",
            "details": error_details,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches unhandled errors gracefully without crashing the server."""
    logger.critical(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred during processing. Please try again later.",
            "detail": str(exc),
        },
    )


STATIC_HTML_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")


# API Endpoints
@app.get("/", tags=["General"])
async def root(request: Request):
    """
    Root endpoint.
    Returns the interactive visual dashboard if accessed via browser (text/html),
    or structured JSON metadata if accessed via API client.
    """
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header and os.path.exists(STATIC_HTML_PATH):
        with open(STATIC_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)

    return RootResponse(
        message="Welcome to the Smart Healthcare Diagnosis API (Disease Prediction System).",
        version=API_VERSION,
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict",
    )


@app.get("/dashboard", response_class=HTMLResponse, tags=["UI"])
@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def interactive_dashboard():
    """Returns the rich interactive clinical diagnosis dashboard."""
    if os.path.exists(STATIC_HTML_PATH):
        with open(STATIC_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard UI file not found.")



@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to monitor API status and ML model availability."""
    model_loaded = bool(inference_service.is_loaded and inference_service.model is not None)
    preprocessor_loaded = bool(
        inference_service.is_loaded and inference_service.scaler is not None and inference_service.imputer is not None
    )
    selected_model_name = inference_service.metadata.get("selected_model", "Unknown")

    if not model_loaded:
        logger.error("Health check failed: Model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded.",
        )

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        preprocessor_loaded=preprocessor_loaded,
        selected_model=selected_model_name,
        version=API_VERSION,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_disease(vitals: PatientVitalsInput):
    """
    Main disease risk prediction endpoint.
    Accepts clinical patient vitals and returns the predicted diagnosis,
    confidence score, probability, and stratified risk level.
    """
    try:
        payload = vitals.model_dump()
        result = inference_service.predict(payload)
        return PredictionResponse(**result)
    except RuntimeError as re:
        logger.error(f"Inference runtime error: {str(re)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re),
        )
    except Exception as e:
        logger.error(f"Unexpected prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )
