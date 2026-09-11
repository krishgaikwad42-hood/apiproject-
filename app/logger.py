"""
Logging Configuration and Request Middleware
Smart Healthcare Diagnosis API
"""

import os
import time
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# ─── Logger Setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("smart_healthcare_api")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False  # Prevent duplicate output in parent loggers

formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if not logger.handlers:
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# ─── Request Logging Middleware ────────────────────────────────────────────────
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    - Assigns a unique X-Request-ID UUID to every incoming request.
    - Logs method, path, status code, latency, and client IP.
    - Injects X-Process-Time-Ms and X-Request-ID headers into responses.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url_path = request.url.path

        # Make request_id available downstream via request.state
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
            logger.info(
                f"[{request_id[:8]}] {client_ip} — {method} {url_path} "
                f"→ {response.status_code} ({duration_ms:.2f}ms)"
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id[:8]}] {client_ip} — {method} {url_path} "
                f"→ EXCEPTION: {str(exc)} ({duration_ms:.2f}ms)"
            )
            raise exc
