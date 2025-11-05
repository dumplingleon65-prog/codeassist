#!/usr/bin/env python3
"""
Service entry point for the solution tester.
"""

import uvicorn

from src.config import settings
from src.api.server import app

if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
