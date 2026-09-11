"""
Application Entrypoint
Run using: python main.py  (or via run.bat on Windows)
"""

import os
import uvicorn

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can also be set externally

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    print(f"  Smart Healthcare Diagnosis API  v1.0.0")
    print(f"  -----------------------------------------")
    print(f"  Dashboard  ->  http://127.0.0.1:{port}/")
    print(f"  Swagger    ->  http://127.0.0.1:{port}/docs")
    print(f"  Health     ->  http://127.0.0.1:{port}/health\n")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )
