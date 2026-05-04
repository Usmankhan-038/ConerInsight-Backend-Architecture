"""
Authentication Utilities

Password hashing (bcrypt) and JWT token management.
Includes dual-token system: Access Token (15 min) + Refresh Token (7 days).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import JWT_SECRET, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS

logger = logging.getLogger(__name__)


_bearer_scheme = HTTPBearer(auto_error=False)




def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))




def create_access_token(user_id: str, email: str, name: str = "") -> str:
    """Create a short-lived JWT access token (15 minutes)."""
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "token_type": "access",
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str, email: str, name: str = "") -> str:
    """Create a long-lived JWT refresh token (7 days)."""
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "token_type": "refresh",
        "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)



def decode_token(token: str) -> Dict:
    """
    Decode and validate a JWT token. Returns the payload dict.
    Raises HTTPException on invalid or expired tokens.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )



async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict:
    """
    FastAPI dependency that extracts and validates the access token.

    Usage in routes:
        current_user = Depends(get_current_user)

    Returns a dict with: sub (user_id), email, name, token_type
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)

    # Only access tokens are valid for API calls
    if payload.get("token_type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Please use an access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": payload["sub"],
        "email": payload.get("email", ""),
        "name": payload.get("name", ""),
    }
