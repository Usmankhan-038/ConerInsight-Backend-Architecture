"""
Authentication Routes

Contains endpoints for user registration, login, and token refresh.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.user import (
    UserRegisterRequest,
    UserLoginRequest,
    RefreshTokenRequest,
    UserResponse,
    AuthMessageResponse
)
from app.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.core.database import (
    create_user as db_create_user,
    get_user_by_email as db_get_user_by_email
)


router = APIRouter()


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@router.post("/register", response_model=AuthMessageResponse)
async def register_user(request: UserRegisterRequest):
    """
    Register a new user.
    
    Creates a new user account with the provided name, email, phone, specialization, and password.
    Returns the created user data (without password) and both access + refresh tokens.
    """
    # Check if email already exists
    existing_user = db_get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="A user with this email already exists"
        )
    
    # Hash the password
    hashed_password = hash_password(request.password)
    
    # Create the user
    user_data = db_create_user(
        name=request.name,
        email=request.email,
        hashed_password=hashed_password,
        phone_number=request.phone_number,
        specialization=request.specialization
    )
    
    if not user_data:
        raise HTTPException(
            status_code=500,
            detail="Failed to create user. Please try again."
        )
    
    # Generate tokens
    access_token = create_access_token(user_data["id"], user_data["email"], user_data["name"])
    refresh_token = create_refresh_token(user_data["id"], user_data["email"], user_data["name"])
    
    return AuthMessageResponse(
        status="success",
        message="User registered successfully",
        user=UserResponse(
            id=user_data["id"],
            name=user_data["name"],
            email=user_data["email"],
            phone_number=user_data.get("phone_number"),
            specialization=user_data.get("specialization"),
            created_at=user_data.get("created_at"),
            updated_at=user_data.get("updated_at")
        ),
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/login", response_model=AuthMessageResponse)
async def login_user(request: UserLoginRequest):
    """
    Login a user.
    
    Validates the email and password and returns both access + refresh tokens.
    """
    # Find user by email
    user_data = db_get_user_by_email(request.email)
    if not user_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(request.password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    # Generate tokens
    access_token = create_access_token(user_data["id"], user_data["email"], user_data["name"])
    refresh_token = create_refresh_token(user_data["id"], user_data["email"], user_data["name"])
    
    return AuthMessageResponse(
        status="success",
        message="Login successful",
        user=UserResponse(
            id=user_data["id"],
            name=user_data["name"],
            email=user_data["email"],
            phone_number=user_data.get("phone_number"),
            specialization=user_data.get("specialization"),
            created_at=user_data.get("created_at"),
            updated_at=user_data.get("updated_at")
        ),
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/refresh", response_model=AuthMessageResponse)
async def refresh_tokens(request: RefreshTokenRequest):
    """
    Refresh tokens.
    
    Accepts a valid refresh token and returns a new pair of access + refresh tokens.
    The old refresh token becomes invalid once a new pair is issued.
    """
    # Decode and validate the refresh token
    payload = decode_token(request.refresh_token)
    
    # Ensure it's actually a refresh token
    if payload.get("token_type") != "refresh":
        raise HTTPException(
            status_code=401,
            detail="Invalid token type. Please provide a refresh token."
        )
    
    user_id = payload["sub"]
    email = payload.get("email", "")
    name = payload.get("name", "")
    
    # Generate new token pair
    new_access_token = create_access_token(user_id, email, name)
    new_refresh_token = create_refresh_token(user_id, email, name)
    
    return AuthMessageResponse(
        status="success",
        message="Tokens refreshed successfully",
        user=UserResponse(
            id=user_id,
            name=name,
            email=email,
        ),
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer"
    )
