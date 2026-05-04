"""
Calculation Utilities

Map contribution analysis, explanation generation, and confidence classification.
"""

import logging
from typing import List

import numpy as np

from app.core.config import MAP_CODES, MAP_DISPLAY_NAMES, FEATURES_PER_MAP

logger = logging.getLogger(__name__)


def calculate_map_contributions(features: np.ndarray, svm_model) -> List[dict]:
    """
    Calculate each map's contribution to the prediction via feature ablation.

    Zeroes out each map's feature segment and measures the change in SVM
    decision score. Higher change = higher contribution.

    Args:
        features: Concatenated feature vector (FEATURES_PER_MAP * 7).
        svm_model: Trained SVM classifier.

    Returns:
        List of contribution dicts sorted descending by percentage.
    """
    base_score = svm_model.decision_function(features.reshape(1, -1))[0]
    contributions = []

    for idx, map_code in enumerate(MAP_CODES):
        ablated = features.copy()
        start = idx * FEATURES_PER_MAP
        end = start + FEATURES_PER_MAP
        ablated[start:end] = 0

        ablated_score = svm_model.decision_function(ablated.reshape(1, -1))[0]
        importance = abs(base_score - ablated_score)

        contributions.append({
            "map_code": map_code,
            "map_name": MAP_DISPLAY_NAMES[map_code],
            "importance": importance,
        })

    total = sum(c["importance"] for c in contributions)
    even_split = 100.0 / len(MAP_CODES)

    for c in contributions:
        c["percentage"] = (c["importance"] / total * 100) if total > 0 else even_split

    return sorted(contributions, key=lambda c: c["percentage"], reverse=True)


def generate_explanation(prediction: str, confidence: float, contributions: List[dict]) -> str:
    """
    Generate a concise human-readable explanation for a prediction.

    Args:
        prediction: Diagnosis label ("Normal" or "Keratoconus").
        confidence: Confidence percentage (0–100).
        contributions: Sorted map contributions.

    Returns:
        One-sentence clinical explanation.
    """
    if prediction == "Keratoconus":
        top_indicators = [
            f"{c['map_name'].split('(')[0].strip()} abnormalities ({c['percentage']:.1f}%)"
            for c in contributions[:3]
        ]
        return (
            f"KERATOCONUS DETECTED with {confidence:.1f}% confidence. "
            f"Key indicators: {'; '.join(top_indicators)}"
        )

    return (
        f"NORMAL cornea detected with {confidence:.1f}% confidence. "
        f"All corneal maps show values within normal parameters."
    )


def get_confidence_level(confidence: float) -> str:
    """
    Classify confidence into High / Medium / Low.

    Args:
        confidence: Confidence percentage (0–100).

    Returns:
        "High" (≥85), "Medium" (≥70), or "Low".
    """
    if confidence >= 85:
        return "High"
    elif confidence >= 70:
        return "Medium"
    else:
        return "Low"
