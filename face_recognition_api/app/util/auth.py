import jwt
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from passlib.context import CryptContext
from ..config import settings
import logging

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(uuid.uuid4()),
        "type": "access"
    })
    
    try:
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.JWT_SECRET_KEY, 
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise

def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Expired token")
        raise
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error decoding token: {str(e)}")
        raise

def get_token_data(token: str) -> Dict[str, Any]:
    payload = decode_token(token)
    return payload

def create_refresh_token_id() -> str:
    return str(uuid.uuid4())

def get_password_reset_token_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=24)

def get_email_verification_token_expiry() -> datetime:
    return datetime.now(timezone.utc)+ timedelta(days=7)

def get_refresh_token_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=30)