"""
Keratoconus Detection API — Main Entry Point

FastAPI application for detecting Keratoconus from Pentacam corneal
topography maps. Uses EfficientNet-B0 for feature extraction and SVM
for classification with Grad-CAM explainability.
"""

import os
import logging
import pickle

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model

tf.config.set_visible_devices([], "GPU")

from app.core.config import HEATMAP_DIR, SVM_MODEL_PATH, ONNX_MODEL_PATH, BASE_URL, get_settings
from app.core import model_registry
from app.api.routes import health, predict, predictions, auth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_models() -> None:
    """Load all ML models and register them in the model registry."""
    logger.info("Loading ML models...")

    # EfficientNet-B0 feature extractor
    base_model = EfficientNetB0(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("predictions").output)
    model_registry.register("feature_extractor", feature_extractor)

    # Grad-CAM gradient model (last conv layer → predictions)
    try:
        # EfficientNet models include a named last conv layer.
        last_conv_layer = base_model.get_layer("top_conv")
    except ValueError:
        # Fallback for model variants without the expected name.
        last_conv_layer = next(
            layer for layer in reversed(base_model.layers)
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
        )
    grad_model = Model(inputs=base_model.input, outputs=[last_conv_layer.output, base_model.output])
    model_registry.register("grad_model", grad_model)

    # SVM classifier
    if os.path.exists(SVM_MODEL_PATH):
        with open(SVM_MODEL_PATH, "rb") as f:
            svm_model = pickle.load(f)
        model_registry.register("svm", svm_model)
        logger.info("SVM model loaded from %s", SVM_MODEL_PATH)
    else:
        logger.warning("SVM model not found at %s", SVM_MODEL_PATH)

    # ONNX model (optional — for parameter-based prediction)
    try:
        import onnxruntime as ort
        if os.path.exists(ONNX_MODEL_PATH):
            onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
            model_registry.register("onnx", onnx_session)
            logger.info("ONNX model loaded from %s", ONNX_MODEL_PATH)
        else:
            logger.warning("ONNX model not found at %s", ONNX_MODEL_PATH)
    except ImportError:
        logger.warning("onnxruntime not installed — parameter-scan endpoint disabled")

    logger.info("Model loading complete")


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files for heatmap images
    application.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")

    # Routes
    application.include_router(health.router, tags=["Health"])
    application.include_router(predict.router, tags=["Prediction"])
    application.include_router(predictions.router, prefix="/predictions", tags=["Prediction History"])
    application.include_router(auth.router, prefix="/auth", tags=["Authentication"])

    return application


# Load models at module level (runs once on startup)
_load_models()

# Create the application instance
app = create_app()


if __name__ == "__main__":
    logger.info("Keratoconus Detection API — Docs: %s/docs", BASE_URL)
    uvicorn.run(app, host="0.0.0.0", port=8000)
