from fastapi import Depends, HTTPException, status
from jose import JWTError
from jwt.exceptions import ExpiredSignatureError
from typing import List, Dict, Any
import logging
from ..util.auth import decode_token, security_scheme
from ..services.auth import get_user_by_id
from ..schemas.auth import UserResponse

logger = logging.getLogger(__name__)

async def get_current_user(token: str = Depends(security_scheme)) ->  Dict[str, Any]:
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
    
    token_expired_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token has expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if token is None:
        raise credentials_exception
    
    try:
        token_str = token.credentials
        
        payload = decode_token(token_str)
        
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        user_id: str = payload.get("uid")
        if user_id is None:
            raise credentials_exception
            
        token_type: str = payload.get("type")
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except ExpiredSignatureError:
        
        logger.warning("Token has expired")
        raise token_expired_exception
    except JWTError as e:
        
        logger.error(f"JWT Error: {str(e)}")
        raise credentials_exception
        
    try:
        
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