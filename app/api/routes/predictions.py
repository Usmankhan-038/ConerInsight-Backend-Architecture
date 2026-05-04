"""
Prediction History Routes

CRUD endpoints for viewing and managing stored prediction records.
All endpoints require authentication.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.database import (
    get_prediction_by_id as db_get_prediction_by_id,
    delete_prediction as db_delete_prediction,
    get_all_predictions_detailed as db_get_all_predictions_detailed,
    update_doctor_feedback as db_update_doctor_feedback,
)
from app.schemas.prediction import DoctorFeedbackRequest
from app.utils.calculations import get_confidence_level
from app.utils.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def list_predictions(
    limit: int = Query(50, ge=1, le=100, description="Max predictions to return"),
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    current_user: dict = Depends(get_current_user),
):
    """List prediction history with map contributions (filtered by logged-in user)."""
    predictions = db_get_all_predictions_detailed(
        limit=limit,
        patient_id=patient_id,
        user_id=current_user["user_id"],
    )
    return {
        "status": "success",
        "count": len(predictions),
        "predictions": predictions,
    }


@router.get("/all")
async def list_all_predictions(
    limit: int = Query(100, ge=1, le=500, description="Max predictions to return"),
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    current_user: dict = Depends(get_current_user),
):
    """
    List all predictions formatted to match the /predict-composite response shape.

    Includes: prediction, confidence, probabilities, map contributions,
    heatmap URLs, patient info, and session metadata.
    Filtered to only show the authenticated user's predictions.
    """
    predictions = db_get_all_predictions_detailed(
        limit=limit,
        patient_id=patient_id,
        user_id=current_user["user_id"],
    )

    formatted = []
    for pred in predictions:
        confidence = pred.get("confidence", 0)

        formatted.append({
            "id": pred.get("session_id") or pred.get("id"),
            "original_id": pred.get("id"),
            "status": "success",
            "prediction": pred.get("prediction", "Unknown"),
            "confidence": confidence,
            "probabilities": {
                "Normal": pred.get("prob_normal", 0),
                "Keratoconus": pred.get("prob_keratoconus", 0),
            },
            "map_contributions": [
                {
                    "map_code": c.get("map_code", ""),
                    "map_name": c.get("map_name", ""),
                    "contribution_percentage": c.get("contribution_percentage", 0),
                }
                for c in pred.get("map_contributions", [])
            ],
            "top_contribution": pred.get("top_contribution", ""),
            "reason": pred.get("reason", ""),
            "confidence_level": get_confidence_level(confidence),
            "prediction_type": pred.get("prediction_type", "image_based"),
            "gradcam_heatmaps": pred.get("heatmap_urls") or {},
            "user_id": pred.get("user_id"),
            "patient_id": pred.get("patient_id"),
            "patient_name": pred.get("patient_name"),
            "eye": pred.get("eye"),
            "notes": pred.get("notes"),
            "doctor_feedback": pred.get("doctor_feedback"),
            "session_id": pred.get("session_id"),
            "created_at": pred.get("created_at"),
        })

    return {
        "status": "success",
        "count": len(formatted),
        "predictions": formatted,
    }


@router.get("/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Retrieve a single prediction with map contributions."""
    prediction = db_get_prediction_by_id(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Ensure the prediction belongs to the current user
    if prediction.get("user_id") and prediction["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="You do not have access to this prediction")
        
    prediction["original_id"] = prediction.get("id")
    if prediction.get("session_id"):
        prediction["id"] = prediction["session_id"]
        
    return {"status": "success", "prediction": prediction}


@router.patch("/{prediction_id}/doctor-feedback")
async def submit_doctor_feedback(
    prediction_id: str,
    request: DoctorFeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """Save the doctor's submitted assessment for an existing prediction."""
    prediction = db_get_prediction_by_id(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if prediction.get("user_id") and prediction["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="You do not have access to this prediction")

    updated = db_update_doctor_feedback(prediction_id, request.doctor_feedback)
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to save doctor feedback")

    return {
        "status": "success",
        "message": "Doctor feedback saved",
        "prediction": updated,
    }


@router.delete("/{prediction_id}")
async def remove_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a prediction by ID (only if it belongs to the current user)."""
    # Verify ownership before deleting
    prediction = db_get_prediction_by_id(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if prediction.get("user_id") and prediction["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="You do not have access to this prediction")

    success = db_delete_prediction(prediction_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prediction not found or delete failed")
    return {"status": "success", "message": f"Prediction {prediction_id} deleted"}
