import logging
import uuid
import random
import string
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Union

import jwt

from ..database import get_db_connection, get_db_transaction
from ..schemas.auth import UserCreate, UserUpdate, TokenResponse, UserResponse
from ..util.auth import (
    hash_password, verify_password, create_access_token, 
    create_refresh_token_id, get_refresh_token_expiry,
    get_password_reset_token_expiry, get_email_verification_token_expiry
)
from ..config import settings
from .error_handling import (
    handle_service_error,
    NotFoundError,
    ValidationError,
    DatabaseError,
    ConflictError
)

logger = logging.getLogger(__name__)

@handle_service_error
def register_user(user_data: UserCreate) -> UserResponse:
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        
    Returns:
        Created user data
        
    Raises:
        ConflictError: If username or email already exists
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT 1 FROM auth_users WHERE username = ?",
                (user_data.username,)
            )
            if cursor.fetchone():
                raise ConflictError(
                    f"Username '{user_data.username}' is already taken",
                    details={"field": "username"}
                )
                
            cursor.execute(
                "SELECT 1 FROM auth_users WHERE email = ?",
                (user_data.email,)
            )
            if cursor.fetchone():
                raise ConflictError(
                    f"Email '{user_data.email}' is already registered",
                    details={"field": "email"}
                )
                
            password_hash = hash_password(user_data.password)
            
            user_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO auth_users (
                    user_id, username, email, phone, password_hash
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, user_data.username, user_data.email, user_data.phone, password_hash)
            )
            
            cursor.execute("SELECT group_id FROM auth_groups WHERE name = 'user'")
            group_row = cursor.fetchone()
            
            if group_row:
                group_id = group_row[0]
                cursor.execute(
                    "INSERT INTO auth_user_groups (user_id, group_id) VALUES (?, ?)",
                    (user_id, group_id)
                )
            
            conn.commit()
            
            verification_id = str(uuid.uuid4())
            expires_at = get_email_verification_token_expiry()
            
            cursor.execute(
                """
                INSERT INTO auth_email_verifications (
                    verification_id, user_id, expires_at
                ) VALUES (?, ?, ?)
                """,
                (verification_id, user_id, expires_at)
            )
            
            conn.commit()
            
            # TODO: Send verification email
            
            user = get_user_by_id(user_id)
            
            return user
            
    except ConflictError:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to register user: {str(e)}", original_error=e)

@handle_service_error
def authenticate_user(username_or_email: str, password: str) -> Tuple[UserResponse, str]:
    """
    Authenticate a user by username/email and password.
    
    Args:
        username_or_email: Username or email
        password: Password
        
    Returns:
        Tuple of (user_data, refresh_token_id)
        
    Raises:
        ValidationError: If credentials are invalid
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            is_email = '@' in username_or_email
            
            if is_email:
                query = "SELECT user_id, password_hash, is_active FROM auth_users WHERE email = ?"
            else:
                query = "SELECT user_id, password_hash, is_active FROM auth_users WHERE username = ?"
                
            cursor.execute(query, (username_or_email,))
            user_row = cursor.fetchone()
            
            if not user_row:
                raise ValidationError("Invalid credentials")
                
            user_id, password_hash, is_active = user_row
            
            if not verify_password(password, password_hash):
                raise ValidationError("Invalid credentials")
                
            if not is_active:
                raise ValidationError("User account is disabled")
                
            try:
                token_id = create_refresh_token_id()
                expires_at = get_refresh_token_expiry()
                
                cursor.execute(
                    """
                    INSERT INTO auth_refresh_tokens (
                        token_id, user_id, expires_at
                    ) VALUES (?, ?, ?)
                    """,
                    (token_id, user_id, expires_at)
                )
                
                cursor.execute(
                    "UPDATE auth_users SET last_login = SYSUTCDATETIME() WHERE user_id = ?",
                    (user_id,)
                )
                
                cursor.execute(
                    """
                    INSERT INTO auth_audit_logs (
                        user_id, event_type, details
                    ) VALUES (?, ?, ?)
                    """,
                    (user_id, "login", "Successful login")
                )
                
                conn.commit()
                
                user = get_user_by_id(user_id)
                
                return user, token_id
            except jwt.PyJWTError as e:
                logger.error(f"JWT error in authentication: {str(e)}")
                raise ValidationError("Authentication failed due to token error")
            
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error authenticating user: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to authenticate user: {str(e)}", original_error=e)

@handle_service_error
def get_user_by_id(user_id: str) -> UserResponse:
    """
    Get user by ID with their groups.
    
    Args:
        user_id: User ID
        
    Returns:
        User data with groups
        
    Raises:
        NotFoundError: If user is not found
        DatabaseError: If there's a database error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT 
                    u.user_id, u.username, u.email, u.phone, 
                    u.is_active, u.is_email_verified, u.last_login,
                    u.created_at, u.updated_at
                FROM auth_users u
                WHERE u.user_id = ?
                """,
                (user_id,)
            )
            
            user_row = cursor.fetchone()
            if not user_row:
                raise NotFoundError("User", user_id)
                
            cursor.execute(
                """
                SELECT g.group_id, g.name, g.description, g.created_at, g.updated_at
                FROM auth_groups g
                JOIN auth_user_groups ug ON g.group_id = ug.group_id
                WHERE ug.user_id = ?
                """,
                (user_id,)
            )
            
            groups = []
            group_row = cursor.fetchone()
            while group_row:
                groups.append({
                    "group_id": str(group_row[0]),
                    "name": group_row[1],
                    "description": group_row[2],
                    "created_at": group_row[3],
                    "updated_at": group_row[4]
                })
                group_row = cursor.fetchone()
    
            user = {
                "user_id": str(user_row[0]),
                "username": user_row[1],
                "email": user_row[2],
                "phone": user_row[3],
                "is_active": bool(user_row[4]),
                "is_email_verified": bool(user_row[5]),
                "last_login": user_row[6],
                "created_at": user_row[7],
                "updated_at": user_row[8],
                "groups": groups
            }
            
            return user
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error retrieving user: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve user: {str(e)}", original_error=e)

@handle_service_error
def create_tokens(user_id: str, user_groups: List[str], refresh_token_id: str, username: str) -> TokenResponse:
    """
    Create access and refresh tokens for a user.
    
    Args:
        user_id: User ID
        user_groups: List of user group names
        refresh_token_id: Refresh token ID
        
    Returns:
        Token response with access and refresh tokens
    """
    try:
        token_data = {
            "sub": username,
            "uid": user_id,
            "groups": user_groups
        }

        access_token = create_access_token(token_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token_id,
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
        }
    except Exception as e:
        logger.error(f"Error creating tokens: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to create tokens: {str(e)}", original_error=e)

@handle_service_error
def refresh_access_token(refresh_token_id: str) -> TokenResponse:
    """
    Refresh access token using refresh token.
    
    Args:
        refresh_token_id: Refresh token ID
        
    Returns:
        New token response
        
    Raises:
        ValidationError: If refresh token is invalid
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT t.user_id, t.is_revoked, t.expires_at
                FROM auth_refresh_tokens t
                WHERE t.token_id = ?
                """,
                (refresh_token_id,)
            )
            
            token_row = cursor.fetchone()
            if not token_row:
                raise ValidationError("Invalid refresh token")
                
            user_id, is_revoked, expires_at = token_row
            if is_revoked:
                raise ValidationError("Refresh token has been revoked")

            if expires_at < datetime.now():
                raise ValidationError("Refresh token has expired")
                
            cursor.execute(
                """
                SELECT g.name
                FROM auth_groups g
                JOIN auth_user_groups ug ON g.group_id = ug.group_id
                WHERE ug.user_id = ?
                """,
                (user_id,)
            )
            
            groups = [row[0] for row in cursor.fetchall()]
            
            cursor.execute(
                """SELECT username from auth_users where user_id = ?""", user_id
            )
            
            username_row = cursor.fetchone()
            if not  username_row:
                raise
            
            username = username_row[0] if username_row else None

            new_token_id = create_refresh_token_id()
            new_expires_at = get_refresh_token_expiry()
            
            cursor.execute(
                """
                INSERT INTO auth_refresh_tokens (
                    token_id, user_id, expires_at
                ) VALUES (?, ?, ?)
                """,
                (new_token_id, user_id, new_expires_at)
            )

            cursor.execute(
                """
                UPDATE auth_refresh_tokens
                SET is_revoked = 1, updated_at = SYSUTCDATETIME()
                WHERE token_id = ?
                """,
                (refresh_token_id,)
            )

            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "token_refresh", "Access token refreshed")
            )
            
            conn.commit()
            
            return create_tokens(user_id, groups, new_token_id, username)
            
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to refresh token: {str(e)}", original_error=e)

@handle_service_error
def logout_user(user_id: str, refresh_token_id: str) -> None:
    """
    Log out a user by revoking their refresh token.
    
    Args:
        user_id: User ID
        refresh_token_id: Refresh token ID
        
    Raises:
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                UPDATE auth_refresh_tokens
                SET is_revoked = 1, updated_at = SYSUTCDATETIME()
                WHERE token_id = ? AND user_id = ?
                """,
                (refresh_token_id, user_id)
            )
            
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "logout", "User logged out")
            )
            
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error logging out user: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to log out user: {str(e)}", original_error=e)

@handle_service_error
def change_password(user_id: str, current_password: str, new_password: str) -> None:
    """
    Change a user's password.
    
    Args:
        user_id: User ID
        current_password: Current password
        new_password: New password
        
    Raises:
        ValidationError: If current password is incorrect
        NotFoundError: If user is not found
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT password_hash FROM auth_users WHERE user_id = ?",
                (user_id,)
            )
            
            user_row = cursor.fetchone()
            if not user_row:
                raise NotFoundError("User", user_id)
                
            current_hash = user_row[0]
            
            if not verify_password(current_password, current_hash):
                raise ValidationError("Current password is incorrect")

            new_hash = hash_password(new_password)
            
            cursor.execute(
                """
                UPDATE auth_users
                SET password_hash = ?, updated_at = SYSUTCDATETIME()
                WHERE user_id = ?
                """,
                (new_hash, user_id)
            )
            
            cursor.execute(
                """
                UPDATE auth_refresh_tokens
                SET is_revoked = 1, updated_at = SYSUTCDATETIME()
                WHERE user_id = ? AND is_revoked = 0
                """,
                (user_id,)
            )
            
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "password_change", "Password changed")
            )
            
            conn.commit()
            
    except (ValidationError, NotFoundError):
        raise
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to change password: {str(e)}", original_error=e)

@handle_service_error
def request_password_reset(email: str) -> None:
    """
    Request a password reset for a user.
    
    Args:
        email: User email
        
    Raises:
        NotFoundError: If user is not found
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT user_id FROM auth_users WHERE email = ?",
                (email,)
            )
            
            user_row = cursor.fetchone()
            if not user_row:
                raise NotFoundError("User with email", email)
                
            user_id = user_row[0]
            
            reset_id = str(uuid.uuid4())
            expires_at = get_password_reset_token_expiry()
            
            cursor.execute(
                """
                INSERT INTO auth_password_resets (
                    reset_id, user_id, expires_at
                ) VALUES (?, ?, ?)
                """,
                (reset_id, user_id, expires_at)
            )
            
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "password_reset_request", "Password reset requested")
            )
            
            conn.commit()
            
            # TODO: Send password reset email
            
    except NotFoundError:
        logger.info(f"Password reset requested for non-existent email: {email}")
        return
    except Exception as e:
        logger.error(f"Error requesting password reset: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to request password reset: {str(e)}", original_error=e)

@handle_service_error
def reset_password(reset_token: str, new_password: str) -> None:
    """
    Reset a user's password using a reset token.
    
    Args:
        reset_token: Password reset token
        new_password: New password
        
    Raises:
        ValidationError: If reset token is invalid
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT r.user_id, r.is_used, r.expires_at
                FROM auth_password_resets r
                WHERE r.reset_id = ?
                """,
                (reset_token,)
            )
            
            token_row = cursor.fetchone()
            if not token_row:
                raise ValidationError("Invalid password reset token")
                
            user_id, is_used, expires_at = token_row
            
            if is_used:
                raise ValidationError("Password reset token has already been used")
                
            if expires_at < datetime.now():
                raise ValidationError("Password reset token has expired")
                
            new_hash = hash_password(new_password)
            
            cursor.execute(
                """
                UPDATE auth_users
                SET password_hash = ?, updated_at = SYSUTCDATETIME()
                WHERE user_id = ?
                """,
                (new_hash, user_id)
            )

            cursor.execute(
                """
                UPDATE auth_password_resets
                SET is_used = 1
                WHERE reset_id = ?
                """,
                (reset_token,)
            )

            cursor.execute(
                """
                UPDATE auth_refresh_tokens
                SET is_revoked = 1, updated_at = SYSUTCDATETIME()
                WHERE user_id = ? AND is_revoked = 0
                """,
                (user_id,)
            )
            
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "password_reset", "Password reset completed")
            )
            
            conn.commit()
            
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to reset password: {str(e)}", original_error=e)

@handle_service_error
def activate_user(user_id: str) -> None:
    """
    Activate a user account.
    
    Args:
        user_id: User ID
        
    Raises:
        NotFoundError: If user is not found
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT 1 FROM auth_users WHERE user_id = ?",
                (user_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("User", user_id)
                
            cursor.execute(
                """
                UPDATE auth_users
                SET is_active = 1, updated_at = SYSUTCDATETIME()
                WHERE user_id = ?
                """,
                (user_id,)
            )

            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "account_activated", "User account activated by admin")
            )
            
            conn.commit()
            
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error activating user: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to activate user: {str(e)}", original_error=e)

@handle_service_error
def reset_user_password(user_id: str) -> str:
    """
    Reset a user's password to a random temporary password.
    
    Args:
        user_id: User ID
        
    Returns:
        Temporary password
        
    Raises:
        NotFoundError: If user is not found
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT 1 FROM auth_users WHERE user_id = ?",
                (user_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("User", user_id)
                
            temp_password = ''.join(
                random.choices(
                    string.ascii_uppercase + 
                    string.ascii_lowercase + 
                    string.digits,
                    k=12
                )
            )
            
            password_hash = hash_password(temp_password)
            
            cursor.execute(
                """
                UPDATE auth_users
                SET password_hash = ?, updated_at = SYSUTCDATETIME()
                WHERE user_id = ?
                """,
                (password_hash, user_id)
            )
            
            cursor.execute(
                """
                UPDATE auth_refresh_tokens
                SET is_revoked = 1, updated_at = SYSUTCDATETIME()
                WHERE user_id = ? AND is_revoked = 0
                """,
                (user_id,)
            )
            
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "password_reset", "Password reset by admin")
            )
            
            conn.commit()
            
            return temp_password
            
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to reset password: {str(e)}", original_error=e)

@handle_service_error
def add_user_to_group(user_id: str, group_name: str) -> None:
    """
    Add a user to a group.
    
    Args:
        user_id: User ID
        group_name: Group name
        
    Raises:
        NotFoundError: If user or group is not found
        ConflictError: If user is already in the group
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
     
            cursor.execute(
                "SELECT 1 FROM auth_users WHERE user_id = ?",
                (user_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("User", user_id)
    
            cursor.execute(
                "SELECT group_id FROM auth_groups WHERE name = ?",
                (group_name,)
            )
            
            group_row = cursor.fetchone()
            if not group_row:
                raise NotFoundError("Group", group_name)
                
            group_id = group_row[0]
            
            cursor.execute(
                """
                SELECT 1 FROM auth_user_groups
                WHERE user_id = ? AND group_id = ?
                """,
                (user_id, group_id)
            )
            
            if cursor.fetchone():
                raise ConflictError(
                    f"User is already in group '{group_name}'",
                    details={"user_id": user_id, "group_name": group_name}
                )

            cursor.execute(
                """
                INSERT INTO auth_user_groups (user_id, group_id)
                VALUES (?, ?)
                """,
                (user_id, group_id)
            )
            
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "added_to_group", f"User added to group '{group_name}'")
            )
            
            conn.commit()
            
    except (NotFoundError, ConflictError):
        raise
    except Exception as e:
        logger.error(f"Error adding user to group: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to add user to group: {str(e)}", original_error=e)

@handle_service_error
def remove_user_from_group(user_id: str, group_name: str) -> None:
    """
    Remove a user from a group.
    
    Args:
        user_id: User ID
        group_name: Group name
        
    Raises:
        NotFoundError: If user or group is not found
        ValidationError: If user is not in the group
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT 1 FROM auth_users WHERE user_id = ?",
                (user_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("User", user_id)
   
            cursor.execute(
                "SELECT group_id FROM auth_groups WHERE name = ?",
                (group_name,)
            )
            
            group_row = cursor.fetchone()
            if not group_row:
                raise NotFoundError("Group", group_name)
                
            group_id = group_row[0]
   
            cursor.execute(
                """
                SELECT 1 FROM auth_user_groups
                WHERE user_id = ? AND group_id = ?
                """,
                (user_id, group_id)
            )
            
            if not cursor.fetchone():
                raise ValidationError(
                    f"User is not in group '{group_name}'",
                    details={"user_id": user_id, "group_name": group_name}
                )

            cursor.execute(
                """
                DELETE FROM auth_user_groups
                WHERE user_id = ? AND group_id = ?
                """,
                (user_id, group_id)
            )
            
        
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "removed_from_group", f"User removed from group '{group_name}'")
            )
            
            conn.commit()
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error removing user from group: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to remove user from group: {str(e)}", original_error=e)

@handle_service_error
def verify_email(verification_token: str) -> None:
    """
    Verify a user's email using a verification token.
    
    Args:
        verification_token: Email verification token
        
    Raises:
        ValidationError: If verification token is invalid
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT v.user_id, v.is_used, v.expires_at
                FROM auth_email_verifications v
                WHERE v.verification_id = ?
                """,
                (verification_token,)
            )
            
            token_row = cursor.fetchone()
            if not token_row:
                raise ValidationError("Invalid email verification token")
                
            user_id, is_used, expires_at = token_row
            
    
            if is_used:
                raise ValidationError("Email verification token has already been used")

            if expires_at < datetime.now():
                raise ValidationError("Email verification token has expired")

            cursor.execute(
                """
                UPDATE auth_users
                SET is_email_verified = 1, updated_at = SYSUTCDATETIME()
                WHERE user_id = ?
                """,
                (user_id,)
            )
            

            cursor.execute(
                """
                UPDATE auth_email_verifications
                SET is_used = 1
                WHERE verification_id = ?
                """,
                (verification_token,)
            )
            

            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "email_verified", "Email verified")
            )
            
            conn.commit()
            
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error verifying email: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to verify email: {str(e)}", original_error=e)

@handle_service_error
def update_user(user_id: str, user_data: UserUpdate) -> UserResponse:
    """
    Update a user's information.
    
    Args:
        user_id: User ID
        user_data: Updated user data
        
    Returns:
        Updated user data
        
    Raises:
        NotFoundError: If user is not found
        ConflictError: If username or email already exists
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            

            cursor.execute(
                "SELECT 1 FROM auth_users WHERE user_id = ?",
                (user_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("User", user_id)

            cursor.execute(
                """
                SELECT COUNT(*) FROM auth_user_groups ug
                JOIN auth_groups g ON ug.group_id = g.group_id
                WHERE ug.user_id = ? AND g.name = 'admin'
                """,
                (user_id,)
            )
            
            is_admin = cursor.fetchone()[0] > 0
            
        
            if is_admin:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM auth_users u
                    JOIN auth_user_groups ug ON u.user_id = ug.user_id
                    JOIN auth_groups g ON ug.group_id = g.group_id
                    WHERE g.name = 'admin' AND u.is_active = 1 AND u.user_id != ?
                    """,
                    (user_id,)
                )
                
                other_admins = cursor.fetchone()[0]
                
                if other_admins == 0:
                    raise ValidationError(
                        "Cannot deactivate the last active admin user",
                        details={"user_id": user_id}
                    )
                
            update_fields = []
            update_values = []
            
           
            if user_data.username is not None:
                
                cursor.execute(
                    "SELECT 1 FROM auth_users WHERE username = ? AND user_id != ?",
                    (user_data.username, user_id)
                )
                
                if cursor.fetchone():
                    raise ConflictError(
                        f"Username '{user_data.username}' is already taken",
                        details={"field": "username"}
                    )
                    
                update_fields.append("username = ?")
                update_values.append(user_data.username)
                
       
            if user_data.email is not None:
             
                cursor.execute(
                    "SELECT 1 FROM auth_users WHERE email = ? AND user_id != ?",
                    (user_data.email, user_id)
                )
                
                if cursor.fetchone():
                    raise ConflictError(
                        f"Email '{user_data.email}' is already registered",
                        details={"field": "email"}
                    )
                    
                update_fields.append("email = ?")
                update_values.append(user_data.email)
              
                update_fields.append("is_email_verified = 0")
                
           
                verification_id = str(uuid.uuid4())
                expires_at = get_email_verification_token_expiry()
                
                cursor.execute(
                    """
                    INSERT INTO auth_email_verifications (
                        verification_id, user_id, expires_at
                    ) VALUES (?, ?, ?)
                    """,
                    (verification_id, user_id, expires_at)
                )
                
                # TODO: Send verification email
                
  
            if user_data.phone is not None:
                update_fields.append("phone = ?")
                update_values.append(user_data.phone)
                
        
            if not update_fields:
                return get_user_by_id(user_id)
                

            update_fields.append("updated_at = SYSUTCDATETIME()")
            
   
            update_values.append(user_id)
    
            query = f"UPDATE auth_users SET {', '.join(update_fields)} WHERE user_id = ?"
            cursor.execute(query, tuple(update_values))
            
        
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "profile_update", "User profile updated")
            )
            
            conn.commit()
            
      
            return get_user_by_id(user_id)
            
    except (NotFoundError, ConflictError):
        raise
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to update user: {str(e)}", original_error=e)

@handle_service_error
def get_all_users(skip: int = 0, limit: int = 100) -> List[UserResponse]:
    """
    Get all users with pagination.
    
    Args:
        skip: Number of users to skip
        limit: Maximum number of users to return
        
    Returns:
        List of users
        
    Raises:
        DatabaseError: If there's a database error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
     
            cursor.execute("SELECT COUNT(*) FROM auth_users")
            total_count = cursor.fetchone()[0]
            
 
            cursor.execute(
                """
                SELECT 
                    u.user_id, u.username, u.email, u.phone, 
                    u.is_active, u.is_email_verified, u.last_login,
                    u.created_at, u.updated_at
                FROM auth_users u
                ORDER BY u.username
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                (skip, limit)
            )
            
            users = []
            while row := cursor.fetchone():
                user_id = row[0]
                
           
                cursor.execute(
                    """
                    SELECT g.group_id, g.name, g.description, g.created_at, g.updated_at
                    FROM auth_groups g
                    JOIN auth_user_groups ug ON g.group_id = ug.group_id
                    WHERE ug.user_id = ?
                    """,
                    (user_id,)
                )
                
                groups = []
                group_row = cursor.fetchone()
                while group_row:
                    groups.append({
                        "group_id": str(group_row[0]),
                        "name": group_row[1],
                        "description": group_row[2],
                        "created_at": group_row[3].isoformat() if group_row[3] else None,
                        "updated_at": group_row[4].isoformat() if group_row[4] else None
                    })
                    group_row = cursor.fetchone()
                
            
                user = {
                    "user_id": str(user_id),
                    "username": row[1],
                    "email": row[2],
                    "phone": row[3],
                    "is_active": bool(row[4]),
                    "is_email_verified": bool(row[5]),
                    "last_login": row[6].isoformat() if row[6] else None,
                    "created_at": row[7].isoformat() if row[7] else None,
                    "updated_at": row[8].isoformat() if row[8] else None,
                    "groups": groups
                }
                
                users.append(user)
            
            return users
            
    except Exception as e:
        logger.error(f"Error getting users: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to get users: {str(e)}", original_error=e)

@handle_service_error
def deactivate_user(user_id: str) -> None:
    """
    Deactivate a user account.
    
    Args:
        user_id: User ID
        
    Raises:
        NotFoundError: If user is not found
        DatabaseError: If there's a database error
    """
    try:
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
       
            cursor.execute(
                "SELECT 1 FROM auth_users WHERE user_id = ?",
                (user_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("User", user_id)
                
      
            cursor.execute(
                """
                UPDATE auth_users
                SET is_active = 0, updated_at = SYSUTCDATETIME()
                WHERE user_id = ?
                """,
                (user_id,)
            )
    
            cursor.execute(
                """
                UPDATE auth_refresh_tokens
                SET is_revoked = 1, updated_at = SYSUTCDATETIME()
                WHERE user_id = ? AND is_revoked = 0
                """,
                (user_id,)
            )
            
        
            cursor.execute(
                """
                INSERT INTO auth_audit_logs (
                    user_id, event_type, details
                ) VALUES (?, ?, ?)
                """,
                (user_id, "account_deactivated", "User account deactivated by admin")
            )
            
            conn.commit()
            
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error deactivating user: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to deactivate user: {str(e)}", original_error=e)