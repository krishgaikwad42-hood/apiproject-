"""
Logging Configuration and Middleware
Smart Healthcare Diagnosis API
"""

import os
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")

# Setup logger
logger = logging.getLogger("smart_healthcare_api")
logger.setLevel(logging.INFO)

# Formatting
formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Avoid duplicate handlers if reloaded
if not logger.handlers:
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming requests, status codes, and execution duration in milliseconds."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url_path = request.url.path

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
            logger.info(
                f"{client_ip} - {method} {url_path} - Status: {response.status_code} - Latency: {duration_ms:.2f}ms"
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{client_ip} - {method} {url_path} - Exception: {str(exc)} - Latency: {duration_ms:.2f}ms"
            )
            raise exc
