from datetime import datetime
import re
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


class TokenBase(BaseModel):
    token_type: str = "bearer"

class TokenResponse(TokenBase):
    access_token: str
    refresh_token: str
    expires_date: int
    
class RefreshTokenReqeust(BaseModel):
    refresh_token: str
    
class UserBase(BaseModel):
    username: str   
    email: EmailStr
    phone: Optional[str] = None
    
    @field_validator('username')
    def username_must_be_valid(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
            raise ValueError("Username must be 3-50 characters and contain only letters, numbers, underscores, and hyphens")
    
    @field_validator('phone')
    def phone_must_be_valid(cls, v ):
        if v and not re.match(r'^\+?[0-9]{10,15}$', v):
            raise ValueError('Phone number must be 10-15 digits, optionally starting with a plus sign')
        return v
        
        
class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    
    @field_validator('password')
    def password_must_be_strong(cls, v):
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$', v):
            raise ValueError('Password must be at least 8 characters and contain uppercase, lowercase, and digits')
        return v    

class UserLogin(BaseModel):
    username_or_email: str
    password: str
    
class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    
    @field_validator('username')
    def username_must_be_valid(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
            raise ValueError('Username must be 3-50 characters and contain only letters, numbers, underscores, and hyphens')
        return v

    @field_validator('phone')
    def phone_must_be_valid(cls, v):
        if v and not re.match(r'^\+?[0-9]{10,15}$', v):
            raise ValueError('Phone number must be 10-15 digits, optionally starting with a plus sign')
        return v
        
class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @field_validator('new_password')
    def password_must_be_strong(cls, v):
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$', v):
            raise ValueError('Password must be at least 8 characters and contain uppercase, lowercase, and digits')
        return v
    
class PasswordReset(BaseModel):
    email: EmailStr
    
class PasswordResetConfirm(BaseModel):
    reset_token: str
    new_password: str = Field(..., min_length=8)
    
    @field_validator('new_password')
    def password_must_be_strong(cls, v):
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$', v):
            raise ValueError('Password must be at least 8 characters and contain uppercase, lowercase, and digits')
        return v

class GroupBase(BaseModel):
    name: str
    description: Optional[str] = None

class GroupCreate(GroupBase):
    pass

class GroupResponse(GroupBase):
    group_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class UserResponse(UserBase):
    user_id: str
    is_active: bool
    is_email_verified: bool
    groups: List[GroupResponse] = []
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True