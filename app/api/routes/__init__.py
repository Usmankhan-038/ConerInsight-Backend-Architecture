"""
API Routes Package

Organizes route modules for the Keratoconus Detection API.
"""

from app.api.routes import health, predict, predictions, auth

__all__ = ["health", "predict", "predictions", "auth"]
