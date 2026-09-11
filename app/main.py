"""
FastAPI Application — Smart Healthcare Diagnosis API
Production-grade REST API for early clinical disease risk prediction.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure base directory is resolvable from any working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.schemas import (
    PatientVitalsInput,
    PredictionResponse,
    HealthResponse,
    RootResponse,
    MetricsResponse,
)
from app.predict import inference_service
from app.logger import logger, RequestLoggingMiddleware

# ─── Versioning ────────────────────────────────────────────────────────────────
API_VERSION = "1.0.0"

# Monotonic start time for uptime calculation
_START_TIME: float = time.monotonic()

# ─── OpenAPI Tag Groups ────────────────────────────────────────────────────────
TAGS_METADATA = [
    {
        "name": "General",
        "description": "Root endpoint and service navigation.",
    },
    {
        "name": "Health",
        "description": "Service health check — confirms ML model and preprocessors are loaded.",
    },
    {
        "name": "Inference",
        "description": "Core disease risk prediction endpoint. Accepts patient vitals and returns diagnosis.",
    },
    {
        "name": "Observability",
        "description": "Operational metrics for monitoring and SRE observability.",
    },
    {
        "name": "UI",
        "description": "Serves the interactive clinical decision support dashboard.",
    },
]


# ─── Lifespan Context (startup / shutdown) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and graceful shutdown events.
    Replaces deprecated @app.on_event("startup") / ("shutdown") pattern.
    """
    # Startup
    logger.info(f"Smart Healthcare Diagnosis API v{API_VERSION} starting up...")
    if inference_service.is_loaded:
        model_name = inference_service.metadata.get("selected_model", "Unknown")
        logger.info(f"ML model ready: {model_name}")
    else:
        logger.warning("ML model is NOT loaded — /predict will be unavailable until model is trained.")

    yield  # Application runs here

    # Shutdown
    logger.info("Smart Healthcare Diagnosis API shutting down. Goodbye.")


# ─── Application Instance ──────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Healthcare Diagnosis API",
    description="""
A **production-grade Machine Learning REST API** for early clinical disease diagnosis.

Trained on clinical health indicators (Glucose, BMI, Blood Pressure, Insulin, Age, etc.)
to provide real-time risk classification and confidence scores.

### Key Features
- **Sub-50ms inference latency** with scikit-learn serialized models
- **Pydantic v2 strict validation** with descriptive error messages
- **Risk stratification** into Low / Moderate / High tiers
- **Dual-gender clinical support** with automatic pregnancy constraint enforcement
- **Correlation tracing** via `X-Request-ID` on every response

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service metadata |
| `GET` | `/health` | Model & service status |
| `POST` | `/predict` | Disease risk prediction |
| `GET` | `/metrics` | Operational metrics |
| `GET` | `/dashboard` | Interactive UI |
    """,
    version=API_VERSION,
    contact={
        "name": "Krish Gaikwad",
        "url": "https://github.com/krishgaikwad42",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing, IP logging, and X-Request-ID injection
app.add_middleware(RequestLoggingMiddleware)


# ─── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats Pydantic validation errors into clean, structured JSON."""
    error_details = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        error_details.append({
            "field": field,
            "message": err.get("msg"),
            "type": err.get("type"),
        })
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {len(error_details)} issue(s)"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "ValidationError",
            "message": "Invalid input vitals provided. Please check the field ranges and types.",
            "details": error_details,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches unhandled exceptions and returns a structured 500 response."""
    logger.critical(
        f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred. Please try again or contact support.",
            "detail": str(exc),
        },
    )


# ─── Static File Path ──────────────────────────────────────────────────────────
STATIC_HTML_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get(
    "/",
    tags=["General"],
    response_model=RootResponse,
    summary="Service Root",
    description=(
        "Returns the interactive visual dashboard when accessed via a browser (Accept: text/html), "
        "or structured JSON service metadata when accessed via an API client."
    ),
)
async def root(request: Request):
    accept_header = request.headers.get("accept", "")
    # Serve HTML if browser requests text/html and not JSON
    if "text/html" in accept_header and "application/json" not in accept_header:
        if os.path.exists(STATIC_HTML_PATH):
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


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Interactive Dashboard",
    description="Serves the rich interactive clinical diagnosis dashboard with real-time risk gauge.",
)
@app.get("/ui", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def interactive_dashboard():
    if os.path.exists(STATIC_HTML_PATH):
        with open(STATIC_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Dashboard UI file not found.",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="System Health Check",
    description="Verifies that the API is operational and that the ML model and preprocessors are loaded in memory.",
)
async def health_check():
    model_loaded = bool(inference_service.is_loaded and inference_service.model is not None)
    preprocessor_loaded = bool(
        inference_service.is_loaded
        and inference_service.scaler is not None
        and inference_service.imputer is not None
    )
    selected_model_name = inference_service.metadata.get("selected_model", "Unknown")

    if not model_loaded:
        logger.error("Health check failed: ML model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded. Run model training first.",
        )

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        preprocessor_loaded=preprocessor_loaded,
        selected_model=selected_model_name,
        version=API_VERSION,
        timestamp=datetime.now().isoformat(),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Observability"],
    summary="Operational Metrics",
    description="Returns lightweight operational metrics: API uptime and version. Suitable for monitoring dashboards.",
)
async def metrics():
    uptime = time.monotonic() - _START_TIME
    return MetricsResponse(
        uptime_seconds=round(uptime, 3),
        version=API_VERSION,
        timestamp=datetime.now().isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict Disease Risk",
    description=(
        "Accepts clinical patient vitals and returns: predicted diagnosis, confidence score, "
        "probability, stratified risk level, and a unique prediction ID for audit tracing."
    ),
)
async def predict_disease(vitals: PatientVitalsInput):
    try:
        payload = vitals.model_dump()
        result = inference_service.predict(payload)
        return PredictionResponse(**result)
    except RuntimeError as re:
        logger.error(f"Inference service unavailable: {str(re)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re),
        )
    except ValueError as ve:
        logger.warning(f"Invalid inference input: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Unexpected prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )
