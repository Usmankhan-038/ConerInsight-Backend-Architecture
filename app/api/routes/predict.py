"""
Prediction Routes

Endpoints for keratoconus detection from corneal topography images
and clinical parameter screening.
"""

import io
import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

from app.core.config import MAP_CODES, MAP_NAMES, IMG_SIZE, HEATMAP_DIR, BASE_URL, FEATURES_PER_MAP
from app.core.model_registry import get_svm_model, get_feature_extractor, get_grad_model
from app.core.database import save_prediction as db_save_prediction, upload_heatmap_to_storage, get_next_sequence_id
from app.schemas.prediction import (
    MapContribution,
    ParameterScanRequest,
    ParameterScanResponse,
    PredictionResponse,
)
from app.utils.image_processing import (
    preprocess_image,
    generate_gradcam_heatmap,
    generate_occlusion_heatmap,
    overlay_heatmap_on_image,
    split_belin_ambrosio_image,
)
from app.utils.calculations import calculate_map_contributions, generate_explanation, get_confidence_level
from app.utils.gemini_report import generate_ai_report, validate_pentacam_image
from app.utils.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Shared Helpers ───────────────────────────────────────────────────────────

def _extract_features_from_maps(
    images: Dict[str, Tuple[np.ndarray, np.ndarray]],
    feature_extractor,
) -> np.ndarray:
    """Extract and concatenate feature vectors from all 7 corneal maps."""
    feature_vectors = []
    for map_code in MAP_CODES:
        _, preprocessed = images[map_code]
        features = feature_extractor.predict(np.expand_dims(preprocessed, 0), verbose=0)
        feature_vectors.append(features.flatten())
    return np.concatenate(feature_vectors)


def _classify(
    features: np.ndarray,
    svm_model,
) -> Tuple[str, float, np.ndarray, List[dict], str]:
    """Run SVM classification and compute contributions / explanation."""
    predicted_class = svm_model.predict(features.reshape(1, -1))[0]
    probabilities = svm_model.predict_proba(features.reshape(1, -1))[0]

    diagnosis = "Normal" if predicted_class == 0 else "Keratoconus"
    confidence = float(probabilities[predicted_class] * 100)

    contributions = calculate_map_contributions(features, svm_model)
    confidence_level = get_confidence_level(confidence)
    return diagnosis, confidence, probabilities, contributions, confidence_level


def _save_heatmap_image(
    overlay: np.ndarray,
    filename: str,
    session_id: str,
    map_code: str,
    persist_to_db: bool,
) -> str:
    """Save a heatmap overlay — to Supabase storage if requested, else locally."""
    if persist_to_db:
        _, encoded = cv2.imencode(".png", overlay)
        remote_url = upload_heatmap_to_storage(encoded.tobytes(), session_id, map_code)
        if remote_url:
            return remote_url

    filepath = os.path.join(HEATMAP_DIR, filename)
    cv2.imwrite(filepath, overlay)
    return f"{BASE_URL}/heatmaps/{filename}"


def _generate_combined_visualization(
    heatmap_images: Dict[str, np.ndarray],
    contributions: List[dict],
    diagnosis: str,
    confidence: float,
    session_id: str,
    persist_to_db: bool,
    filename_prefix: str = "combined",
) -> Optional[str]:
    """Create a 7-panel combined heatmap figure and save it."""
    try:
        fig, axes = plt.subplots(1, 7, figsize=(21, 3))
        for i, map_code in enumerate(MAP_CODES):
            img = heatmap_images.get(map_code)
            if img is None:
                axes[i].axis("off")
                continue
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            contrib = next((c for c in contributions if c["map_code"] == map_code), {})
            pct = contrib.get("percentage", 0)
            axes[i].set_title(f"{map_code}\n{pct:.1f}%", fontsize=9)
            axes[i].axis("off")

        color = "green" if diagnosis == "Normal" else "red"
        plt.suptitle(f"{diagnosis} ({confidence:.1f}%)", fontweight="bold", color=color)
        plt.tight_layout()

        filename = f"{filename_prefix}_{session_id}.png"

        if persist_to_db:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            combined_bytes = buf.read()
            plt.close()
            remote_url = upload_heatmap_to_storage(combined_bytes, session_id, "combined")
            if remote_url:
                return remote_url
            filepath = os.path.join(HEATMAP_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(combined_bytes)
        else:
            filepath = os.path.join(HEATMAP_DIR, filename)
            plt.savefig(filepath, dpi=150)
            plt.close()

        return f"{BASE_URL}/heatmaps/{filename}"

    except Exception as exc:
        logger.error("Combined visualization failed: %s", exc)
        plt.close("all")
        return None


def _persist_prediction(
    diagnosis: str,
    confidence: float,
    probabilities: np.ndarray,
    reason: str,
    prediction_type: str,
    contributions: List[dict],
    session_id: str,
    patient_id: Optional[str],
    patient_name: Optional[str],
    eye: Optional[str],
    notes: Optional[str],
    heatmap_urls: Dict[str, str],
    user_id: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Save prediction to the database. Returns (saved, prediction_id)."""
    try:
        result = db_save_prediction(
            prediction=diagnosis,
            confidence=round(confidence, 2),
            prob_normal=round(float(probabilities[0]), 4),
            prob_keratoconus=round(float(probabilities[1]), 4),
            reason=reason,
            prediction_type=prediction_type,
            map_contributions=[
                {
                    "map_code": c["map_code"],
                    "map_name": c["map_name"],
                    "contribution_percentage": round(c["percentage"], 2),
                }
                for c in contributions
            ],
            session_id=session_id,
            patient_id=patient_id,
            patient_name=patient_name,
            eye=eye,
            notes=notes,
            heatmap_urls=heatmap_urls,
            user_id=user_id,
        )
        if result:
            return True, session_id
    except Exception as exc:
        logger.error("Failed to persist prediction: %s", exc)
    return False, None


def _build_response(
    diagnosis: str,
    confidence: float,
    probabilities: np.ndarray,
    contributions: List[dict],
    confidence_level: str,
    ai_analysis: str,
    prediction_type: str,
    heatmap_urls: Dict[str, str],
    combined_url: Optional[str],
    saved_to_db: bool,
    prediction_id: Optional[str],
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    eye: Optional[str] = None,
    user_id: Optional[str] = None,
) -> PredictionResponse:
    """Construct the unified prediction response."""

    return PredictionResponse(
        prediction=diagnosis,
        confidence=round(confidence, 2),
        probabilities={
            "Normal": round(float(probabilities[0]), 4),
            "Keratoconus": round(float(probabilities[1]), 4),
        },
        map_contributions=[
            MapContribution(
                map_code=c["map_code"],
                map_name=c["map_name"],
                contribution_percentage=round(c["percentage"], 2),
            )
            for c in contributions
        ],
        reason=ai_analysis,
        confidence_level=confidence_level,
        prediction_type=prediction_type,
        gradcam_heatmaps=heatmap_urls,
        combined_gradcam=combined_url,
        saved_to_db=saved_to_db,
        prediction_id=prediction_id,
        ai_analysis=ai_analysis,
        user_id=user_id,
    )


# ── Route: Predict from 7 Separate Maps ─────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict_from_maps(
    CT_A: UploadFile = File(..., description="Corneal Thickness (Anterior) map"),
    Elv_A: UploadFile = File(..., description="Elevation (Anterior) map"),
    Elv_P: UploadFile = File(..., description="Elevation (Posterior) map"),
    EC_A: UploadFile = File(..., description="Eccentricity (Anterior) map"),
    EC_P: UploadFile = File(..., description="Eccentricity (Posterior) map"),
    Sag_A: UploadFile = File(..., description="Sagittal Curvature (Anterior) map"),
    Sag_P: UploadFile = File(..., description="Sagittal Curvature (Posterior) map"),
    save_to_db: bool = Query(False, description="Persist prediction to database"),
    patient_id: Optional[str] = Query(None, description="Patient identifier"),
    patient_name: Optional[str] = Query(None, description="Patient name"),
    eye: Optional[str] = Query(None, description="Examined eye"),
    notes: Optional[str] = Query(None, description="Clinical notes"),
    svm_model=Depends(get_svm_model),
    feature_extractor=Depends(get_feature_extractor),
    grad_model=Depends(get_grad_model),
    current_user: dict = Depends(get_current_user),
):
    uploads = {
        "CT_A": CT_A, "Elv_A": Elv_A, "Elv_P": Elv_P,
        "EC_A": EC_A, "EC_P": EC_P, "Sag_A": Sag_A, "Sag_P": Sag_P,
    }

    session_id = get_next_sequence_id()
    images: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for map_code in MAP_CODES:
        content = await uploads[map_code].read()
        
        if map_code == MAP_CODES[0]:
            try:
                if not validate_pentacam_image(content):
                    raise HTTPException(status_code=400, detail="The uploaded image does not appear to be a valid Pentacam map. Please upload correct images.")
            except ValueError as e:
                if str(e) == "RATE_LIMIT":
                    raise HTTPException(status_code=429, detail="AI Validation service is experiencing high traffic. Please wait 1 minute and try your prediction again.")
                raise

        original, preprocessed = preprocess_image(content)
        images[map_code] = (original, preprocessed)

    all_features = _extract_features_from_maps(images, feature_extractor)
    diagnosis, confidence, probabilities, contributions, confidence_level = _classify(all_features, svm_model)

    ai_analysis = generate_ai_report(
        prediction=diagnosis,
        confidence=confidence,
        probabilities={"Normal": float(probabilities[0]), "Keratoconus": float(probabilities[1])},
        contributions=contributions,
        confidence_level=confidence_level,
    )

    # Generate occlusion sensitivity heatmaps
    logger.info("Generating occlusion sensitivity heatmaps...")
    heatmap_urls: Dict[str, str] = {}
    heatmap_images: Dict[str, np.ndarray] = {}

    for idx, map_code in enumerate(MAP_CODES):
        original, preprocessed = images[map_code]
        heatmap = generate_occlusion_heatmap(preprocessed, all_features, idx, feature_extractor, svm_model)
        overlay = overlay_heatmap_on_image(original, heatmap)
        heatmap_images[map_code] = overlay

        filename = f"occlusion_{session_id}_{map_code}.png"
        heatmap_urls[map_code] = _save_heatmap_image(overlay, filename, session_id, map_code, save_to_db)
        logger.info("  %s heatmap generated", map_code)

    combined_url = _generate_combined_visualization(
        heatmap_images, contributions, diagnosis, confidence, session_id, save_to_db,
    )

    saved, prediction_id = False, None
    if save_to_db:
        reason = generate_explanation(diagnosis, confidence, contributions)
        saved, prediction_id = _persist_prediction(
            diagnosis, confidence, probabilities, reason,
            "image_based", contributions, session_id, patient_id, patient_name, eye, notes, heatmap_urls,
            user_id=current_user["user_id"],
        )

    return _build_response(
        diagnosis, confidence, probabilities, contributions, confidence_level,
        ai_analysis, "image_based", heatmap_urls, combined_url, saved, prediction_id,
        patient_id=patient_id, patient_name=patient_name, eye=eye,
        user_id=current_user["user_id"],
    )


# ── Route: Predict from Composite Image ─────────────────────────────────────

@router.post("/predict-composite", response_model=PredictionResponse)
async def predict_from_composite(
    image: UploadFile = File(..., description="Belin/Ambrósio composite Pentacam image"),
    save_to_db: bool = Query(False, description="Persist prediction to database"),
    generate_heatmaps: bool = Query(True, description="Generate Grad-CAM heatmaps"),
    patient_id: Optional[str] = Query(None, description="Patient identifier"),
    patient_name: Optional[str] = Query(None, description="Patient name"),
    eye: Optional[str] = Query(None, description="Examined eye"),
    notes: Optional[str] = Query(None, description="Clinical notes"),
    svm_model=Depends(get_svm_model),
    feature_extractor=Depends(get_feature_extractor),
    grad_model=Depends(get_grad_model),
    current_user: dict = Depends(get_current_user),
):
    try:
        content = await image.read()
        
        try:
            is_valid = validate_pentacam_image(content)
        except ValueError as e:
            if str(e) == "RATE_LIMIT":
                raise HTTPException(status_code=429, detail="AI Validation service is experiencing high traffic. Please wait 1 minute and try your prediction again.")
            raise

        if not is_valid:
            raise HTTPException(status_code=400, detail="The uploaded image does not appear to be a valid Pentacam map. Please upload correct images.")
            
        composite = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if composite is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")

        logger.info("Processing composite image: %s", composite.shape)

        split_maps = split_belin_ambrosio_image(composite)
        session_id = get_next_sequence_id()

        images: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for map_code in MAP_CODES:
            if map_code not in split_maps:
                raise HTTPException(status_code=500, detail=f"Failed to extract {map_code} from composite image")
            resized = cv2.resize(split_maps[map_code], IMG_SIZE)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            preprocessed = preprocess_input(img_to_array(rgb))
            images[map_code] = (resized, preprocessed)

        all_features = _extract_features_from_maps(images, feature_extractor)
        diagnosis, confidence, probabilities, contributions, confidence_level = _classify(all_features, svm_model)

        ai_analysis = generate_ai_report(
            prediction=diagnosis,
            confidence=confidence,
            probabilities={"Normal": float(probabilities[0]), "Keratoconus": float(probabilities[1])},
            contributions=contributions,
            confidence_level=confidence_level,
        )

        heatmap_urls: Dict[str, str] = {}
        heatmap_images: Dict[str, np.ndarray] = {}
        combined_url = None

        if generate_heatmaps:
            logger.info("Generating Grad-CAM heatmaps...")
            for map_code in MAP_CODES:
                original, preprocessed = images[map_code]
                heatmap = generate_gradcam_heatmap(preprocessed, grad_model)
                overlay = overlay_heatmap_on_image(original, heatmap)
                heatmap_images[map_code] = overlay

                filename = f"composite_{session_id}_{map_code}.png"
                heatmap_urls[map_code] = _save_heatmap_image(overlay, filename, session_id, map_code, save_to_db)
                logger.info("  %s heatmap generated", map_code)

            combined_url = _generate_combined_visualization(
                heatmap_images, contributions, diagnosis, confidence,
                session_id, save_to_db, filename_prefix="composite_combined",
            )

        saved, prediction_id = False, None
        if save_to_db:
            reason = generate_explanation(diagnosis, confidence, contributions)
            saved, prediction_id = _persist_prediction(
                diagnosis, confidence, probabilities, reason,
                "composite_image", contributions, session_id, patient_id, patient_name, eye, notes, heatmap_urls,
                user_id=current_user["user_id"],
            )

        return _build_response(
            diagnosis, confidence, probabilities, contributions, confidence_level,
            ai_analysis, "composite_image", heatmap_urls, combined_url, saved, prediction_id,
            patient_id=patient_id, patient_name=patient_name, eye=eye,
            user_id=current_user["user_id"],
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        raise HTTPException(status_code=500, detail=err_msg)


# ── Route: Parameter-Based Screening ────────────────────────────────────────

CLINICAL_THRESHOLDS = {
    "k_max": {"normal": (0, 47.2), "suspect": (47.2, 49.0), "keratoconus": (49.0, 999), "weight": 0.20},
    "cct": {"normal": (490, 999), "suspect": (470, 490), "keratoconus": (0, 470), "weight": 0.15, "inverted": True},
    "posterior_elevation": {"normal": (0, 17), "suspect": (17, 29), "keratoconus": (29, 999), "weight": 0.20},
    "anterior_elevation": {"normal": (0, 12), "suspect": (12, 20), "keratoconus": (20, 999), "weight": 0.12},
    "asymmetry_index": {"normal": (0, 1.5), "suspect": (1.5, 3.0), "keratoconus": (3.0, 999), "weight": 0.13},
    "anterior_curvature": {"normal": (7.2, 8.4), "suspect_low": (6.8, 7.2), "keratoconus_low": (0, 6.8), "weight": 0.10, "range_based": True},
    "posterior_curvature": {"normal": (5.8, 6.8), "suspect_low": (5.2, 5.8), "keratoconus_low": (0, 5.2), "weight": 0.10, "range_based": True},
}


def _score_parameter(value: float, thresholds: dict) -> int:
    """Score a single parameter: 0=normal, 1=suspect, 2=keratoconus."""
    if thresholds.get("range_based"):
        lo, hi = thresholds["normal"]
        suspect_lo = thresholds.get("suspect_low", (0, 0))
        if lo <= value <= hi:
            return 0
        elif suspect_lo[0] <= value < suspect_lo[1] or value > hi:
            return 1
        else:
            return 2

    elif thresholds.get("inverted"):
        if value >= thresholds["normal"][0]:
            return 0
        elif value >= thresholds["suspect"][0]:
            return 1
        else:
            return 2

    else:
        if value < thresholds["normal"][1]:
            return 0
        elif value < thresholds["suspect"][1]:
            return 1
        else:
            return 2


@router.post("/parameter-scan", response_model=ParameterScanResponse)
async def parameter_scan(
    request: ParameterScanRequest,
    current_user: dict = Depends(get_current_user),
):
    """Screen for keratoconus using clinical corneal topography parameters."""
    params = {
        "k_max": request.k_max,
        "cct": request.cct,
        "posterior_elevation": request.posterior_elevation,
        "anterior_elevation": request.anterior_elevation,
        "asymmetry_index": request.asymmetry_index,
        "anterior_curvature": request.anterior_curvature,
        "posterior_curvature": request.posterior_curvature,
    }

    risk_scores: Dict[str, dict] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for param_name, value in params.items():
        thresholds = CLINICAL_THRESHOLDS[param_name]
        score = _score_parameter(value, thresholds)
        weight = thresholds["weight"]

        risk_scores[param_name] = {
            "value": value,
            "risk": ["Normal", "Suspect", "Keratoconus"][score],
            "score": score,
        }
        weighted_sum += score * weight
        total_weight += weight

    # Optional: K astigmatism
    if request.k_steep is not None and request.k_flat is not None:
        k_diff = request.k_steep - request.k_flat
        k_score = 2 if k_diff > 3.0 else (1 if k_diff > 2.0 else 0)
        weighted_sum += k_score * 0.05
        total_weight += 0.05
        risk_scores["k_astigmatism"] = {
            "value": round(k_diff, 2),
            "risk": ["Normal", "Suspect", "Keratoconus"][k_score],
            "score": k_score,
        }

    # Optional: Thinnest point
    if request.thinnest_point is not None:
        tp_score = _score_parameter(request.thinnest_point, CLINICAL_THRESHOLDS["cct"])
        weighted_sum += tp_score * 0.08
        total_weight += 0.08
        risk_scores["thinnest_point"] = {
            "value": request.thinnest_point,
            "risk": ["Normal", "Suspect", "Keratoconus"][tp_score],
            "score": tp_score,
        }

    final_score = weighted_sum / total_weight if total_weight > 0 else 0

    # Classification
    if final_score < 0.5:
        prediction_label = "Normal"
        prob_n = max(0.6, 1.0 - final_score)
        prob_s = min(0.3, final_score * 0.6)
        prob_k = min(0.1, final_score * 0.2)
    elif final_score < 1.2:
        prediction_label = "Suspect"
        prob_n = max(0.1, 0.5 - final_score * 0.3)
        prob_s = max(0.4, 0.6 - abs(final_score - 0.85) * 0.3)
        prob_k = min(0.4, final_score * 0.3)
    else:
        prediction_label = "Keratoconus"
        prob_n = max(0.02, 0.2 - final_score * 0.1)
        prob_s = max(0.08, 0.3 - final_score * 0.1)
        prob_k = min(0.95, final_score * 0.5)

    # Normalize probabilities
    total_prob = prob_n + prob_s + prob_k
    prob_n = round(prob_n / total_prob, 4)
    prob_s = round(prob_s / total_prob, 4)
    prob_k = round(1.0 - prob_n - prob_s, 4)

    probabilities = {"Normal": prob_n, "Suspect": prob_s, "Keratoconus": prob_k}
    confidence = round(max(probabilities.values()) * 100, 2)
    confidence_level = get_confidence_level(confidence)

    ai_analysis = generate_ai_report(
        prediction=prediction_label,
        confidence=confidence,
        probabilities=probabilities,
        contributions=[],
        confidence_level=confidence_level,
        clinical_params=risk_scores,
    )

    saved, prediction_id = False, None
    session_id = get_next_sequence_id()
    if request.save_to_db:
        param_titles = {
            "k_max": ("K_Max", "K-max (Max Keratometry)"),
            "cct": ("CCT", "CCT (Central Corneal Thickness)"),
            "thinnest_point": ("T_Point", "Thinnest Point"),
            "posterior_elevation": ("P_Elev", "Posterior Elevation"),
            "anterior_elevation": ("A_Elev", "Anterior Elevation"),
            "asymmetry_index": ("Asym_Idx", "Asymmetry Index"),
            "anterior_curvature": ("A_Curv", "Anterior Curvature"),
            "posterior_curvature": ("P_Curv", "Posterior Curvature"),
            "k_astigmatism": ("K_Astig", "K Astigmatism"),
        }
        param_contributions = [
            {
                "map_code": param_titles.get(k, (k[:10], k))[0],
                "map_name": param_titles.get(k, (k[:10], k.replace("_", " ").title()))[1],
                "contribution_percentage": float(v["value"]),
            }
            for k, v in risk_scores.items()
        ]

        try:
            result = db_save_prediction(
                prediction=prediction_label,
                confidence=confidence,
                prob_normal=prob_n,
                prob_keratoconus=prob_k,
                reason=ai_analysis,
                prediction_type="parameter_scan",
                map_contributions=param_contributions,
                session_id=session_id,
                patient_id=request.patient_id,
                patient_name=request.patient_name,
                eye=request.eye,
                notes=request.notes,
                user_id=current_user["user_id"],
            )
            if result:
                saved = True
                prediction_id = session_id
        except Exception as exc:
            logger.error("Failed to save parameter scan: %s", exc)

    return ParameterScanResponse(
        prediction=prediction_label,
        confidence=confidence,
        probabilities=probabilities,
        risk_scores=risk_scores,
        reason=ai_analysis,
        confidence_level=confidence_level,
        saved_to_db=saved,
        prediction_id=prediction_id,
        patient_id=request.patient_id,
        patient_name=request.patient_name,
        eye=request.eye,
        ai_analysis=ai_analysis,
    )
