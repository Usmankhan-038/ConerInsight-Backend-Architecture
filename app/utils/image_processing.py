"""
Image Processing Utilities

Preprocessing, Grad-CAM generation, occlusion sensitivity,
heatmap overlay, and composite image splitting.
"""

import logging
from typing import Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

from app.core.config import IMG_SIZE, MAP_CODES

logger = logging.getLogger(__name__)


def preprocess_image(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode raw image bytes and prepare for model input.

    Returns:
        (original_resized_bgr, preprocessed_array)
    """
    raw = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    resized = cv2.resize(img, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    preprocessed = preprocess_input(img_to_array(rgb))
    return resized, preprocessed


def generate_gradcam_heatmap(
    img_preprocessed: np.ndarray,
    grad_model: tf.keras.Model,
) -> np.ndarray:
    """
    Fast Grad-CAM heatmap via single forward + backward pass.

    Args:
        img_preprocessed: Preprocessed array (224, 224, 3).
        grad_model: Model outputting [conv_features, predictions].

    Returns:
        Normalized heatmap (224, 224).
    """
    img_batch = np.expand_dims(img_preprocessed, 0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        top_class = tf.argmax(predictions[0])
        target_score = predictions[:, top_class]

    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return cv2.resize(heatmap, IMG_SIZE)


def generate_occlusion_heatmap(
    img_preprocessed: np.ndarray,
    all_features: np.ndarray,
    map_idx: int,
    feature_extractor: tf.keras.Model,
    svm_model,
    patch_size: int = 56,
    stride: int = 28,
) -> np.ndarray:
    """
    Occlusion sensitivity heatmap for SVM-based prediction.

    Slides a gray patch across the image and measures SVM decision
    score changes. Slow (~30-60s) but directly explains SVM behavior.

    Args:
        img_preprocessed: Preprocessed array (224, 224, 3).
        all_features: Full concatenated feature vector (7 × 1000).
        map_idx: Index of this map (0–6) in the feature vector.
        feature_extractor: CNN feature extraction model.
        svm_model: Trained SVM classifier.
        patch_size: Occlusion patch size in pixels.
        stride: Sliding step size.

    Returns:
        Normalized heatmap (224, 224).
    """
    from app.core.config import FEATURES_PER_MAP

    h, w = IMG_SIZE
    grid_h = (h - patch_size) // stride + 1
    grid_w = (w - patch_size) // stride + 1
    heatmap = np.zeros((grid_h, grid_w))

    base_score = svm_model.decision_function(all_features.reshape(1, -1))[0]
    gray_value = preprocess_input(np.ones((1, 1, 3), dtype=np.float32) * 128)[0, 0, :]

    for i, y in enumerate(range(0, h - patch_size + 1, stride)):
        for j, x in enumerate(range(0, w - patch_size + 1, stride)):
            occluded = img_preprocessed.copy()
            occluded[y:y + patch_size, x:x + patch_size, :] = gray_value

            occluded_features = feature_extractor.predict(
                np.expand_dims(occluded, 0), verbose=0,
            ).flatten()

            modified = all_features.copy()
            start = map_idx * FEATURES_PER_MAP
            end = start + FEATURES_PER_MAP
            modified[start:end] = occluded_features

            new_score = svm_model.decision_function(modified.reshape(1, -1))[0]
            heatmap[i, j] = abs(base_score - new_score)

    heatmap_full = cv2.resize(heatmap, IMG_SIZE)
    if heatmap_full.max() > heatmap_full.min():
        heatmap_full = (heatmap_full - heatmap_full.min()) / (heatmap_full.max() - heatmap_full.min())

    return heatmap_full


def overlay_heatmap_on_image(original_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a JET-colorized heatmap onto the original BGR image."""
    resized = cv2.resize(heatmap, IMG_SIZE)
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(original_img, 0.6, colored, 0.4, 0)


def split_belin_ambrosio_image(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split a Belin/Ambrósio Enhanced Ectasia composite into 7 corneal maps.

    Layout (standard Pentacam display):
        Left panel: 3×2 grid of maps (Elv, EC, Sag rows × Anterior/Posterior cols)
        Right panel: Corneal Thickness map + patient data
    """
    height, width = img.shape[:2]
    logger.debug("Composite image dimensions: %dx%d", width, height)

    # Left panel column boundaries
    col1_start = int(width * 0.05)
    col1_end = int(width * 0.28)
    col2_start = col1_end
    col2_end = int(width * 0.50)

    # Row boundaries (after header)
    row1_start = int(height * 0.12)
    row1_end = int(height * 0.41)
    row2_start = row1_end
    row2_end = int(height * 0.70)
    row3_start = row2_end
    row3_end_trimmed = int(height * 0.95)

    # Right panel — Corneal Thickness
    ct_left = int(width * 0.67)
    ct_top = int(height * 0.05)
    ct_bottom = int(height * 0.50)

    maps = {
        "Elv_A": img[row1_start:row1_end, col1_start:col1_end],
        "Elv_P": img[row1_start:row1_end, col2_start:col2_end],
        "CT_A":  img[ct_top:ct_bottom, ct_left:width],
        "EC_A":  img[row2_start:row2_end, col1_start:col1_end],
        "EC_P":  img[row2_start:row2_end, col2_start:col2_end],
        "Sag_A": img[row3_start:row3_end_trimmed, col1_start:col1_end],
        "Sag_P": img[row3_start:row3_end_trimmed, col2_start:col2_end],
    }

    for code, map_img in maps.items():
        logger.debug("  Extracted %s: %dx%d", code, map_img.shape[1], map_img.shape[0])

    return maps
