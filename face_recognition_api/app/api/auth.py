from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from ..schemas.auth import (
    UserCreate, UserResponse, TokenResponse, RefreshTokenReqeust,
    PasswordChange, PasswordReset, PasswordResetConfirm, UserUpdate
)
from ..services.auth import (
    authenticate_user, create_tokens, refresh_access_token,
    logout_user, change_password, request_password_reset, reset_password,
    verify_email, update_user
)
from ..config import settings
from ..utils import create_response
from ..dependencies.auth import get_current_user, get_current_active_user, is_admin

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get access token."""
    try:
        user, refresh_token_id = authenticate_user(form_data.username, form_data.password)
        
        user_groups = [group["name"] for group in user["groups"]]
        
        tokens = create_tokens(user["user_id"], user_groups, refresh_token_id, user["username"])
        
        headers = {
            "Jwt-Token": tokens["access_token"],
            "Refresh-Token": tokens["refresh_token"]
        }
        
        return create_response(body={"message":"Authentication successful"},
                               headers=headers
                               )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid credentials exception: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh(refresh_request: RefreshTokenReqeust):
    """Refresh access token."""
    try:
        tokens = refresh_access_token(refresh_request.refresh_token)
        
        headers = {
            "Jwt-Token": tokens["access_token"],
            "Refresh-Token": tokens["refresh_token"]
        }

        return create_response(body={"message": "Token refreshed successful"}, headers=headers)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    refresh_request: RefreshTokenReqeust,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Logout and invalidate refresh token."""
    try:
        logout_user(current_user["user_id"], refresh_request.refresh_token)
        return create_response(
            body={},
            status_code=status.HTTP_204_NO_CONTENT
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/password-change", status_code=status.HTTP_204_NO_CONTENT)
async def password_change(
    password_data: PasswordChange,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Change password for logged in user."""
    try:
        change_password(
            current_user["user_id"],
            password_data.current_password,
            password_data.new_password
        )
        return create_response(
            body={},
            status_code=status.HTTP_204_NO_CONTENT
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/password-reset-confirm", status_code=status.HTTP_204_NO_CONTENT)
async def password_reset_confirm(reset_data: PasswordResetConfirm):
    """Confirm password reset with token."""
    try:
        reset_password(reset_data.reset_token, reset_data.new_password)
        return create_response(
            body={},
            status_code=status.HTTP_204_NO_CONTENT
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_me(
    user_data: UserUpdate,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Update current user profile."""
    try:
        # Users can only update specific fields of their own profile
        limited_data = UserUpdate(
            username=None,  # Don't allow username change
            email=None,     # Don't allow email change
            phone=user_data.phone  # Allow phone number change
        )
        updated_user = update_user(current_user["user_id"], limited_data)
        return create_response(body=updated_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )