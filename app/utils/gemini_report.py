"""
Gemini AI Report Generator

Sends all prediction data to Gemini API and returns its response directly.
Retries with backoff on rate limits instead of silently falling back.
"""

import io
import logging
import os
import time
from typing import Dict, List, Optional
import PIL.Image

logger = logging.getLogger(__name__)

try:
    from google import genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
_RETRYABLE_ERRORS = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 5


def generate_ai_report(
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    contributions: List[dict],
    confidence_level: str,
    clinical_params: Optional[Dict[str, dict]] = None,
) -> str:
    """
    Send all prediction data to Gemini and return its analysis.
    Retries on rate limits. Falls back only if Gemini is completely unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not _GEMINI_AVAILABLE or api_key == "your-gemini-api-key-here":
        logger.warning("Gemini not configured, returning basic summary")
        return _basic_fallback(prediction, confidence, probabilities, contributions)

    result = _call_gemini(api_key, prediction, confidence, probabilities, contributions, confidence_level, clinical_params)
    if result:
        return result

    logger.warning("Gemini failed after retries, returning basic summary")
    return _basic_fallback(prediction, confidence, probabilities, contributions)


def _call_gemini(
    api_key: str,
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    contributions: List[dict],
    confidence_level: str,
    clinical_params: Optional[Dict[str, dict]] = None,
) -> Optional[str]:
    """Send one detailed prompt to Gemini with retries on rate limit."""
    try:
        client = genai.Client(api_key=api_key)
        prompt = _build_prompt(prediction, confidence, probabilities, contributions, confidence_level, clinical_params)

        for model_name in _GEMINI_MODELS:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    logger.info("Gemini %s — attempt %d/%d", model_name, attempt, _MAX_RETRIES)
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    if response and response.text:
                        logger.info("Gemini report received from %s", model_name)
                        return response.text.strip()
                except Exception as exc:
                    error_str = str(exc)
                    if any(code in error_str for code in _RETRYABLE_ERRORS):
                        if attempt < _MAX_RETRIES:
                            wait = _RETRY_DELAY_SECONDS * attempt
                            logger.warning(
                                "%s unavailable or rate limited (attempt %d/%d), retrying in %ds...",
                                model_name, attempt, _MAX_RETRIES, wait,
                            )
                            time.sleep(wait)
                        else:
                            logger.warning("%s exhausted all retries, trying next model", model_name)
                    else:
                        logger.error("%s error: %s", model_name, exc)
                        break  # Non-rate-limit error, skip to next model

        logger.error("All Gemini models failed")
    except Exception as exc:
        logger.error("Gemini client error: %s", exc)

    return None


def _build_prompt(
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    contributions: List[dict],
    confidence_level: str,
    clinical_params: Optional[Dict[str, dict]] = None,
) -> str:
    """Build one comprehensive prompt with ALL patient data."""
    prob_lines = "\n".join(f"  - {label}: {_fmt(val)}" for label, val in probabilities.items())

    if contributions:
        contrib_lines = "Map Contributions (sorted by importance):\n" + "\n".join(
            f"  {i+1}. {c.get('map_name', c.get('map_code', ''))} "
            f"({c.get('map_code', '')}): {c.get('percentage', c.get('contribution_percentage', 0)):.1f}%"
            for i, c in enumerate(contributions)
        )
    elif clinical_params:
        contrib_lines = "Clinical Parameters Used (Parameter Scan):\n" + "\n".join(
            f"  - {k}: {v.get('value')} (Risk: {v.get('risk')})"
            for k, v in clinical_params.items()
        )
    else:
        contrib_lines = "  No specific maps or parameters provided."

    return (
        f"You are an ophthalmology AI analyzing a Pentacam corneal topography scan for keratoconus.\n\n"
        f"=== SCAN RESULTS ===\n"
        f"Diagnosis: {prediction}\n"
        f"Confidence: {confidence:.1f}% ({confidence_level})\n\n"
        f"Class Probabilities:\n{prob_lines}\n\n"
        f"{contrib_lines}\n\n"
        f"=== TASK ===\n"
        f"Write a 50-60 word clinical explanation for this specific patient.\n"
        f"- Reference the actual map names and their exact contribution percentages\n"
        f"- Explain WHY those maps led to '{prediction}' classification\n"
        f"- Be specific to THIS scan data, not generic\n"
        f"- No emojis, no markdown, plain text only\n"
    )


def _fmt(value: float) -> str:
    """Format probability to percentage."""
    if isinstance(value, (int, float)) and value <= 1:
        return f"{value * 100:.1f}%"
    return f"{value:.1f}%"


def _basic_fallback(
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    contributions: List[dict],
) -> str:
    """Minimal fallback when Gemini is completely unavailable."""
    prob_str = ", ".join(f"{k}: {_fmt(v)}" for k, v in probabilities.items())

    top_maps = ""
    if contributions:
        top_maps = " Top maps: " + ", ".join(
            f"{c.get('map_name', c.get('map_code', ''))} ({c.get('percentage', c.get('contribution_percentage', 0)):.1f}%)"
            for c in contributions[:3]
        ) + "."

    return f"{prediction} detected with {confidence:.1f}% confidence. Probabilities: {prob_str}.{top_maps}"


def validate_pentacam_image(image_bytes: bytes) -> bool:
    """
    Check if the provided image appears to be a valid Pentacam corneal topography map.
    Returns True if valid (or if Gemini fails/unavailable), False if the image is explicitly rejected.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not _GEMINI_AVAILABLE or api_key == "your-gemini-api-key-here":
        # If Gemini is not configured, don't block the prediction
        return True

    try:
        client = genai.Client(api_key=api_key)
        image = PIL.Image.open(io.BytesIO(image_bytes))

        prompt = (
            "Analyze this image carefully. Is this image a genuine clinical Pentacam corneal topography map, scan, or report? "
            "Look for characteristics like color-coded medical heatmaps, keratometry values (K1, K2, Kmax), "
            "pachymetry maps, or Sagittal curvature maps. "
            "If the image is an architecture diagram, a flowchart, a UI wireframe, a photo of a person/animal, "
            "or ANY non-medical image, you MUST reply strictly with 'NO'."
            "Reply strictly with only 'YES' if it is a genuine Pentacam topography map/scan/report, "
            "or 'NO' if it is anything else."
        )

        for model_name in _GEMINI_MODELS:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[image, prompt],
                    )
                    if response and response.text:
                        result_text = response.text.strip().upper()
                        
                        # LOGGING FOR DEBUGGING
                        logger.info("Gemini validation response from %s: %r", model_name, response.text)
                        print("Gemini validation response:", repr(response.text))
                        
                        if "NO" in result_text and "YES" not in result_text:
                            return False
                        return True
                except Exception as e:
                    error_str = str(e)
                    if any(code in error_str for code in _RETRYABLE_ERRORS):
                        if attempt < _MAX_RETRIES:
                            time.sleep(_RETRY_DELAY_SECONDS * attempt)
                            continue
                        logger.warning("%s validation exhausted all retries, trying next model", model_name)
                    else:
                        logger.error("%s validation error: %s", model_name, getattr(e, "message", str(e)))
                        break
                    
        # If we failed after all retries or due to a non-rate-limit error,
        # fail open so we don't accidentally block valid predictions when
        # the API is down or rate-limited.
        return True
    except ValueError as e:
        if str(e) == "RATE_LIMIT":
            raise e
        logger.error("Gemini validation setup error: %s", e)
        return True
    except Exception as e:
        logger.error("Gemini context setup failed for validation: %s", getattr(e, "message", str(e)))
        return True
