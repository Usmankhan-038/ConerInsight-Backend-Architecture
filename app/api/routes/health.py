"""
Health Check Routes
"""

from fastapi import APIRouter

from app.core import model_registry

router = APIRouter()


@router.get("/")
async def health_check():
    """API health check — returns status, model availability, and version."""
    return {
        "status": "online",
        "model_loaded": model_registry.get("svm") is not None,
        "version": "2.0.0",
    }
