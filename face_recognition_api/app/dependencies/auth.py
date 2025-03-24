# from fastapi import HTTPException, status, Header
# from ..config import settings

# async def verify_api_key(x_api_key: str = Header(None)):
#     if not x_api_key:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="API key is required"
#         )
#     if x_api_key != settings.API_KEY:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid API key"
#         )
#     return x_api_key


from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from typing import Optional, List, Dict, Any
import logging
from ..util.auth import decode_token
from ..services.auth import get_user_by_id
from ..database import get_db_connection
from ..schemas.auth import UserResponse

logger = logging.getLogger(__name__)

# OAuth2 scheme for swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """
    Get the current authenticated user from a JWT token.
    
    Args:
        token: JWT token
        
    Returns:
        User data
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = decode_token(token)
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        token_type: str = payload.get("type")
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    
            
    except JWTError:
        raise credentials_exception
        
    try:
        # Get user from database
        user = get_user_by_id(user_id)
        
        if not user:
            raise credentials_exception
            
        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return user
        
    except Exception as e:
        logger.error(f"Error in get_current_user: {str(e)}")
        raise credentials_exception

async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """
    Get the current active user.
    
    Args:
        current_user: Current user
        
    Returns:
        User data
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def has_group(required_groups: List[str]):
    """
    Dependency to check if user has any of the required groups.
    
    Args:
        required_groups: List of required group names
        
    Returns:
        Dependency function
    """
    async def check_groups(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
        user_groups = [group["name"] for group in current_user["groups"]]
        
        for group in required_groups:
            if group in user_groups:
                return current_user
                
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
        
    return check_groups

# Specific role-based dependencies
async def is_admin(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """
    Check if user is an admin.
    
    Args:
        current_user: Current user
        
    Returns:
        User data
        
    Raises:
        HTTPException: If user is not an admin
    """
    user_groups = [group["name"] for group in current_user["groups"]]
    
    if "admin" not in user_groups:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
        
    return current_user