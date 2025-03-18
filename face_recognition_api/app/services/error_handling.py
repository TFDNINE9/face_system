import logging
from fastapi import HTTPException, status
from typing import Optional, Type, Dict, Any, Union, Callable
import traceback
import inspect

logger = logging.getLogger(__name__)

class ServiceError(Exception):
    
    def __init__(
        self, 
        message: str, 
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details
        super().__init__(message)

class NotFoundError(ServiceError):
    
    def __init__(self, resource_type: str, identifier: str, details: Optional[Dict[str, Any]] = None):
        message = f"{resource_type} not found: {identifier}"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=f"{resource_type.upper()}_NOT_FOUND",
            details=details
        )

class ValidationError(ServiceError):
    """Validation error for input data."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="VALIDATION_ERROR",
            details=details
        )

class DatabaseError(ServiceError):
    """Database-related errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None):
        error_details = details or {}
        if original_error:
            error_details["error_type"] = type(original_error).__name__
            
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="DATABASE_ERROR",
            details=error_details
        )

class ConflictError(ServiceError):
    """Conflict error for resource state conflicts."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT_ERROR",
            details=details
        )

def handle_service_error(func: Callable) -> Callable:
    is_async = inspect.iscoroutinefunction(func)
    
    if is_async:
        # Async wrapper for async functions
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except ServiceError as e:
                logger.error(f"Service error: {e.message}", exc_info=True)
                raise HTTPException(
                    status_code=e.status_code,
                    detail={"message": e.message}
                )
            except Exception as e:
                logger.error(f"Unhandled error in service function: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"message": "An unexpected error occurred"}
                )
        return async_wrapper
    else:
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HTTPException:
                raise
            except ServiceError as e:
                logger.error(f"Service error: {e.message}", exc_info=True)
                raise HTTPException(
                    status_code=e.status_code,
                    detail={"message": e.message}
                )
            except Exception as e:
                logger.error(f"Unhandled error in service function: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"message": "An unexpected error occurred"}
                )
        return sync_wrapper