"""
Configuration — Centralized Settings for the Keratoconus Detection API

All application constants, paths, and Pydantic settings are defined here.
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings

# ── Corneal Map Configuration ────────────────────────────────────────────────

MAP_CODES = ["CT_A", "Elv_A", "Elv_P", "EC_A", "EC_P", "Sag_A", "Sag_P"]

MAP_DISPLAY_NAMES = {
    "CT_A": "Corneal Thickness (Anterior)",
    "Elv_A": "Elevation (Anterior)",
    "Elv_P": "Elevation (Posterior)",
    "EC_A": "Eccentricity (Anterior)",
    "EC_P": "Eccentricity (Posterior)",
    "Sag_A": "Sagittal Curvature (Anterior)",
    "Sag_P": "Sagittal Curvature (Posterior)",
}

# Keep old names as aliases for backward compatibility
MAP_SUFFIXES = MAP_CODES
MAP_NAMES = MAP_DISPLAY_NAMES

# ── Image & Model Constants ─────────────────────────────────────────────────

IMG_SIZE = (224, 224)
FEATURES_PER_MAP = 1000
TOTAL_FEATURES = FEATURES_PER_MAP * len(MAP_CODES)

# ── Path Configuration ───────────────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

HEATMAP_DIR = os.path.join(ROOT_DIR, "heatmaps")
SVM_MODEL_PATH = os.path.join(ROOT_DIR, "svm_model_2class.pkl")
ONNX_MODEL_PATH = os.path.join(ROOT_DIR, "keratoconus_model.onnx")

os.makedirs(HEATMAP_DIR, exist_ok=True)

# ── API Configuration ────────────────────────────────────────────────────────

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# ── JWT Configuration ────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "keratoconus-api-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7


# ── Pydantic Settings ────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    app_name: str = "Keratoconus Detection API"
    app_version: str = "2.0.0"
    app_description: str = "API for detecting Keratoconus from Pentacam corneal topography images"

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    model_path: str = SVM_MODEL_PATH
    upload_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_uploads")
    heatmap_dir: str = HEATMAP_DIR
    image_size: tuple = IMG_SIZE

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
