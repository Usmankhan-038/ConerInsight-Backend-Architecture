"""
User Schemas for Authentication API

Pydantic models for user registration, login, and response data.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserRegisterRequest(BaseModel):
    """Request model for user registration."""
    name: str = Field(..., min_length=2, max_length=255, description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    phone_number: Optional[str] = Field(None, max_length=20, description="User's phone number")
    specialization: Optional[str] = Field(None, max_length=255, description="User's specialization (e.g., Ophthalmologist)")
    password: str = Field(..., min_length=6, max_length=128, description="User's password")


class UserLoginRequest(BaseModel):
    """Request model for user login."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


class RefreshTokenRequest(BaseModel):
    """Request model for refreshing tokens."""
    refresh_token: str = Field(..., description="Valid refresh token")


class UserResponse(BaseModel):
    """Response model for user data (excludes password)."""
    id: str
    name: str
    email: str
    phone_number: Optional[str] = None
    specialization: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Response model for authentication tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


class AuthMessageResponse(BaseModel):
    """Response model for auth operations with messages."""
    status: str
    message: str
    user: Optional[UserResponse] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: Optional[str] = None
