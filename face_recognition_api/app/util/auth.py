from fastapi.security import APIKeyHeader
import jwt
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from passlib.context import CryptContext
from ..config import settings
import logging

logger = logging.getLogger(__name__)

jwt_token_header = APIKeyHeader(name="Jwt-Token", auto_error=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    issuance_time = datetime.now() - timedelta(seconds=30)
    
    to_encode.update({
        "exp": expire,
        "iat": issuance_time,
        "jti": str(uuid.uuid4()),
        "type": "access"
    })
    
    try:
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.JWT_SECRET_KEY, 
            algorithm="HS512"
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise

def decode_token(token: str) -> Dict[str, Any]:
    try:
        if isinstance(token, str):
            token_bytes = token.encode('utf-8')
        else:
            token_bytes = token
        
        payload = jwt.decode(
            token_bytes, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": False, 
                "verify_aud": True
            }
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error decoding token: {str(e)}")
        raise
    
def create_refresh_token_id() -> str:
    return str(uuid.uuid4())

def get_password_reset_token_expiry() -> datetime:
    return datetime.now() + timedelta(hours=24)

def get_email_verification_token_expiry() -> datetime:
    return datetime.now() + timedelta(days=7)

def get_refresh_token_expiry() -> datetime:
    return datetime.now() + timedelta(days=30)