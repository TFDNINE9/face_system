import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm

from face_recognition_api.app.services.error_handling import ValidationError
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

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

# @router.post("/login", response_model=TokenResponse)
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     """Login to get access token."""
#     try:
#         user, refresh_token_id = authenticate_user(form_data.username, form_data.password)
        
#         user_groups = [group["name"] for group in user["groups"]]
        
#         tokens = create_tokens(user["user_id"], user_groups, refresh_token_id, user["username"])
        
#         headers = {
#             "Jwt-Token": tokens["access_token"],
#             "Refresh-Token": tokens["refresh_token"]
#         }
        
#         return create_response(body={"message":"Authentication successful"},
#                                headers=headers
#                                )
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail=f"Invalid credentials exception: {e}",
#             headers={"WWW-Authenticate": "Bearer"},
#         )




@router.post("/login", response_model=TokenResponse)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login to get access token.
    
    This endpoint is rate-limited and monitors for brute force attempts.
    Multiple failed attempts will trigger temporary IP blocking.
    """
    try:
        # Get client IP for security tracking
        client_ip = get_client_ip(request)
        
        # Authenticate user with IP for security tracking
        user, refresh_token_id = await authenticate_user(
            form_data.username, 
            form_data.password,
            client_ip
        )
        
        user_groups = [group["name"] for group in user["groups"]]
        
        # Create tokens
        tokens = create_tokens(user["user_id"], user_groups, refresh_token_id, user["username"])
        
        # Set response headers
        headers = {
            "Jwt-Token": tokens["access_token"],
            "Refresh-Token": tokens["refresh_token"]
        }
        
        # Log successful login
        logger.info(f"Successful login for user '{user['username']}' from IP {client_ip}")
        
        return create_response(
            body={"message": "Authentication successful"},
            headers=headers
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error with client IP
        logger.error(f"Login failed from IP {get_client_ip(request)}: {str(e)}")
        
        # Determine appropriate status code and message
        status_code = status.HTTP_401_UNAUTHORIZED
        detail = "Invalid credentials"
        
        # Extract more specific error message if available
        if isinstance(e, ValidationError) and "retry_after" in getattr(e, "details", {}):
            status_code = status.HTTP_403_FORBIDDEN
            detail = str(e)
        
        raise HTTPException(
            status_code=status_code,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        Client IP address
    """
    # Check for X-Forwarded-For header (when behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Use the first IP in the list (client IP)
        return forwarded_for.split(",")[0].strip()
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"


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