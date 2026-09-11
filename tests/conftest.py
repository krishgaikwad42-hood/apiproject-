"""
pytest configuration — shared fixtures and custom markers.
Smart Healthcare Diagnosis API Test Suite
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app


def pytest_configure(config):
    """Register custom pytest markers to suppress PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "smoke: Core functionality smoke tests")
    config.addinivalue_line("markers", "inference: ML prediction endpoint tests")
    config.addinivalue_line("markers", "validation: Input validation and boundary tests")
    config.addinivalue_line("markers", "observability: Health, metrics, and header tests")
    config.addinivalue_line("markers", "ui: Dashboard and HTML endpoint tests")


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Session-scoped FastAPI test client.
    Shared across all tests to avoid repeated app startup overhead.
    """
    with TestClient(app) as c:
        yield c
