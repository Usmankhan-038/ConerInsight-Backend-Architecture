"""
Database — Supabase Integration

Client initialization and CRUD operations for predictions, map contributions,
user management, and heatmap storage.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_client: Optional[Client] = None


def get_supabase_client() -> Optional[Client]:
    """Return the Supabase client singleton (lazy-initialized)."""
    global _client

    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning("Supabase credentials not found in environment")
            return None
        try:
            _client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase connected: %s", SUPABASE_URL)
        except Exception as exc:
            logger.error("Supabase connection failed: %s", exc)
            return None

    return _client




def get_next_sequence_id() -> str:
    """Generate a sequential PAT-XXXX ID based on table count."""
    client = get_supabase_client()
    if client is None:
        import random
        return f"PAT-{random.randint(1000, 9999)}"

    try:
        res = client.table("predictions").select("id", count="exact").limit(1).execute()
        count = res.count if res.count is not None else 0
        return f"PAT-{(count + 1):04d}"
    except Exception as exc:
        logger.error("Error getting sequence count for ID generation: %s", exc)
        import random
        return f"PAT-{random.randint(1000, 9999)}"


def save_prediction(
    prediction: str,
    confidence: float,
    prob_normal: float,
    prob_keratoconus: float,
    reason: str,
    prediction_type: str = "image_based",
    map_contributions: Optional[List[Dict]] = None,
    session_id: Optional[str] = None,
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    eye: Optional[str] = None,
    notes: Optional[str] = None,
    heatmap_urls: Optional[Dict[str, str]] = None,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Insert a prediction record and its map contributions."""
    client = get_supabase_client()
    if client is None:
        return None

    try:
        data = {
            "prediction": prediction,
            "confidence": confidence,
            "prob_normal": prob_normal,
            "prob_keratoconus": prob_keratoconus,
            "reason": reason,
            "prediction_type": prediction_type,
            "session_id": session_id,
            "patient_id": patient_id,
            "notes": notes,
            "heatmap_urls": heatmap_urls,
        }
        if user_id is not None:
            data["user_id"] = user_id
        
        # Only cleanly append them so we don't strictly crash if null
        if patient_name is not None:
            data["patient_name"] = patient_name
        if eye is not None:
            data["eye"] = eye

        try:
            result = client.table("predictions").insert(data).execute()
        except Exception as e:
            # Fallback if the Supabase schema doesn't have the new eye/patient columns yet
            if "PGRST204" in str(e) or "column" in str(e).lower():
                data.pop("eye", None)
                data.pop("patient_name", None)
                result = client.table("predictions").insert(data).execute()
            else:
                raise e

        if not result.data:
            return None

        prediction_id = result.data[0]["id"]

        if map_contributions:
            rows = [
                {
                    "prediction_id": prediction_id,
                    "map_code": c["map_code"],
                    "map_name": c["map_name"],
                    "contribution_percentage": c["contribution_percentage"],
                }
                for c in map_contributions
            ]
            client.table("map_contributions").insert(rows).execute()

        logger.info("Prediction saved: %s", prediction_id)
        return result.data[0]

    except Exception as exc:
        logger.error("Error saving prediction: %s", exc)
        return None


def get_predictions(
    limit: int = 50,
    patient_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent predictions, optionally filtered by patient and/or user."""
    client = get_supabase_client()
    if client is None:
        return []

    try:
        query = client.table("predictions").select("*").order("created_at", desc=True).limit(limit)
        if patient_id:
            query = query.eq("patient_id", patient_id)
        if user_id:
            query = query.eq("user_id", user_id)

        result = query.execute()
        return result.data or []

    except Exception as exc:
        logger.error("Error fetching predictions: %s", exc)
        return []


def get_prediction_by_id(prediction_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single prediction with its map contributions. Supports UUID or PAT-XXXX ID."""
    client = get_supabase_client()
    if client is None:
        return None

    import uuid
    is_uuid = False
    try:
        uuid.UUID(prediction_id)
        is_uuid = True
    except ValueError:
        pass

    try:
        query = client.table("predictions").select("*")
        if is_uuid:
            result = query.eq("id", prediction_id).execute()
        else:
            result = query.eq("session_id", prediction_id).execute()

        if not result.data:
            return None

        prediction = result.data[0]
        actual_id = prediction["id"]

        contrib_result = client.table("map_contributions").select("*").eq("prediction_id", actual_id).execute()
        prediction["map_contributions"] = contrib_result.data or []

        return prediction

    except Exception as exc:
        logger.error("Error fetching prediction %s: %s", prediction_id, exc)
        return None


def update_doctor_feedback(
    prediction_id: str,
    doctor_feedback: str,
) -> Optional[Dict[str, Any]]:
    """Update the doctor's submitted assessment for a prediction."""
    client = get_supabase_client()
    if client is None:
        return None

    import uuid
    is_uuid = False
    try:
        uuid.UUID(prediction_id)
        is_uuid = True
    except ValueError:
        pass

    try:
        query = client.table("predictions").update({
            "doctor_feedback": doctor_feedback,
        })

        if is_uuid:
            result = query.eq("id", prediction_id).execute()
        else:
            result = query.eq("session_id", prediction_id).execute()

        if not result.data:
            return None

        logger.info("Doctor feedback updated for prediction: %s", prediction_id)
        return result.data[0]

    except Exception as exc:
        logger.error("Error updating doctor feedback for %s: %s", prediction_id, exc)
        return None


def delete_prediction(prediction_id: str) -> bool:
    """Delete a prediction and its cascade-linked contributions. Supports UUID or PAT-XXXX ID."""
    client = get_supabase_client()
    if client is None:
        return False

    import uuid
    is_uuid = False
    try:
        uuid.UUID(prediction_id)
        is_uuid = True
    except ValueError:
        pass

    try:
        if is_uuid:
            client.table("predictions").delete().eq("id", prediction_id).execute()
        else:
            client.table("predictions").delete().eq("session_id", prediction_id).execute()
        logger.info("Prediction deleted: %s", prediction_id)
        return True
    except Exception as exc:
        logger.error("Error deleting prediction %s: %s", prediction_id, exc)
        return False


def get_all_predictions_detailed(
    limit: int = 100,
    patient_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch predictions with full details including contributions and top contributor."""
    client = get_supabase_client()
    if client is None:
        return []

    try:
        query = client.table("predictions").select("*").order("created_at", desc=True).limit(limit)
        if patient_id:
            query = query.eq("patient_id", patient_id)
        if user_id:
            query = query.eq("user_id", user_id)

        result = query.execute()
        predictions = result.data or []

        for pred in predictions:
            try:
                contrib_result = (
                    client.table("map_contributions")
                    .select("*")
                    .eq("prediction_id", pred["id"])
                    .order("contribution_percentage", desc=True)
                    .execute()
                )
                contributions = contrib_result.data or []
                pred["map_contributions"] = contributions
                pred["top_contribution"] = contributions[0].get("map_name", "") if contributions else ""
            except Exception:
                pred["map_contributions"] = []
                pred["top_contribution"] = ""

        return predictions

    except Exception as exc:
        logger.error("Error fetching detailed predictions: %s", exc)
        return []


# ── Heatmap Storage ──────────────────────────────────────────────────────────

def upload_heatmap_to_storage(
    image_data: bytes,
    session_id: str,
    map_code: str,
) -> Optional[str]:
    """Upload a heatmap PNG to Supabase Storage. Returns the public URL."""
    client = get_supabase_client()
    if client is None:
        return None

    try:
        file_name = f"{session_id}/{map_code}.png"
        client.storage.from_("heatmaps").upload(
            file_name,
            image_data,
            {"content-type": "image/png", "upsert": "true"},
        )
        public_url = client.storage.from_("heatmaps").get_public_url(file_name)
        logger.debug("Uploaded heatmap %s → %s", map_code, public_url)
        return public_url

    except Exception as exc:
        logger.error("Error uploading heatmap: %s", exc)
        return None


# ── User Management ──────────────────────────────────────────────────────────

def create_user(
    name: str,
    email: str,
    hashed_password: str,
    phone_number: Optional[str] = None,
    specialization: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create a new user account."""
    client = get_supabase_client()
    if client is None:
        return None

    try:
        data: Dict[str, Any] = {"name": name, "email": email, "hashed_password": hashed_password}
        if phone_number:
            data["phone_number"] = phone_number
        if specialization:
            data["specialization"] = specialization

        result = client.table("users").insert(data).execute()
        if result.data:
            logger.info("User created: %s", email)
            return result.data[0]
        return None

    except Exception as exc:
        logger.error("Error creating user: %s", exc)
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Look up a user by email address."""
    client = get_supabase_client()
    if client is None:
        return None

    try:
        result = client.table("users").select("*").eq("email", email).execute()
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.error("Error fetching user by email: %s", exc)
        return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Look up a user by UUID."""
    client = get_supabase_client()
    if client is None:
        return None

    try:
        result = client.table("users").select("*").eq("id", user_id).execute()
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.error("Error fetching user by ID: %s", exc)
        return None


# Initialize client eagerly on module load
get_supabase_client()
