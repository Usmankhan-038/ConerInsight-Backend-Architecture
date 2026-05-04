"""
Model Registry — Centralized ML Model Access

Provides a singleton registry for ML models and FastAPI dependency
injection functions. Models are registered at startup by main.py
and consumed by route handlers via Depends().
"""

import logging
from typing import Any, Dict

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_registry: Dict[str, Any] = {}


def register(name: str, model: Any) -> None:
    """Register a model in the global registry."""
    _registry[name] = model
    logger.info("Registered model: %s", name)


def get(name: str) -> Any:
    """Retrieve a model by name. Returns None if not found."""
    return _registry.get(name)


# ── FastAPI Dependency Injection ─────────────────────────────────────────────

def get_svm_model():
    """Dependency: Inject the trained SVM classifier."""
    model = _registry.get("svm")
    if model is None:
        raise HTTPException(status_code=503, detail="SVM model not loaded")
    return model


def get_feature_extractor():
    """Dependency: Inject the EfficientNet-B0 feature extractor."""
    model = _registry.get("feature_extractor")
    if model is None:
        raise HTTPException(status_code=503, detail="Feature extractor not loaded")
    return model


def get_grad_model():
    """Dependency: Inject the Grad-CAM gradient model."""
    model = _registry.get("grad_model")
    if model is None:
        raise HTTPException(status_code=503, detail="Grad-CAM model not loaded")
    return model
