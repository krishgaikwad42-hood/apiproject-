"""
Application Entrypoint
Run using: python main.py
"""

import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Smart Healthcare Diagnosis API on http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
