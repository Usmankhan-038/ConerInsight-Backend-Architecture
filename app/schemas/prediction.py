"""
Prediction Schemas

Pydantic models for prediction request/response payloads.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MapContribution(BaseModel):
    """Contribution of a single corneal map to the overall prediction."""
    map_code: str
    map_name: str
    contribution_percentage: float


class PredictionResponse(BaseModel):
    """Unified response for image-based keratoconus predictions."""
    status: str = "success"
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    map_contributions: List[MapContribution] = []
    reason: str = ""
    confidence_level: str = ""
    prediction_type: str = "image_based"
    heatmaps: Dict[str, str] = Field(default_factory=dict, alias="gradcam_heatmaps")
    combined_heatmap_url: Optional[str] = Field(default=None, alias="combined_gradcam")
    saved_to_db: bool = False
    prediction_id: Optional[str] = None
    user_id: Optional[str] = None
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    eye: Optional[str] = None
    doctor_feedback: Optional[str] = None
    ai_analysis: Optional[str] = None

    class Config:
        populate_by_name = True


class DoctorFeedbackRequest(BaseModel):
    """Doctor's submitted assessment for an existing prediction."""
    doctor_feedback: str = Field(..., min_length=1, description="Doctor's clinical assessment or feedback")


class ParameterScanRequest(BaseModel):
    """Request body for parameter-based keratoconus screening."""
    # Required parameters
    k_max: float = Field(..., description="K-Max value in Diopters (normal: 42–46 D)")
    asymmetry_index: float = Field(..., description="Asymmetry Index (normal: 0–1.5)")
    cct: float = Field(..., description="Central Corneal Thickness in μm (normal: 500–570)")
    anterior_elevation: float = Field(..., description="Anterior Elevation in μm from BFS")
    posterior_elevation: float = Field(..., description="Posterior Elevation in μm from BFS")
    anterior_curvature: float = Field(..., description="Anterior Curvature radius in mm")
    posterior_curvature: float = Field(..., description="Posterior Curvature radius in mm")

    # Optional parameters
    k_flat: Optional[float] = None
    k_steep: Optional[float] = None
    k_mean: Optional[float] = None
    thinnest_point: Optional[float] = None
    isv: Optional[float] = None
    iva: Optional[float] = None
    ki: Optional[float] = None
    cki: Optional[float] = None
    iha: Optional[float] = None
    ihd: Optional[float] = None
    esi: Optional[float] = None

    # Meta parameters
    save_to_db: bool = Field(False, description="Persist prediction to database")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    patient_name: Optional[str] = Field(None, description="Patient name")
    eye: Optional[str] = Field(None, description="Examined eye")
    notes: Optional[str] = Field(None, description="Clinical notes")


class ParameterScanResponse(BaseModel):
    """Response for parameter-based keratoconus screening."""
    status: str = "success"
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    risk_scores: Dict[str, dict]
    reason: str = ""
    confidence_level: str = ""
    prediction_type: str = "parameter_scan"
    saved_to_db: bool = False
    prediction_id: Optional[str] = None
    user_id: Optional[str] = None
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    eye: Optional[str] = None
    doctor_feedback: Optional[str] = None
    ai_analysis: Optional[str] = None
