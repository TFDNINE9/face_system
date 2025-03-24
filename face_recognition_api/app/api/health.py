import logging
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from ..dependencies.auth import get_current_active_user, is_admin
from ..schemas.auth import UserResponse
from ..services.health import check_system_health

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check():
    """
    Public health check endpoint.
    
    No authentication required.
    """
    try:
        health_data = check_system_health()
        return {
            "status": "healthy",
            **health_data
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "detail": str(e)
            }
        )

@router.get("/system-info")
async def system_info(current_user: UserResponse = Depends(is_admin)):
    """
    Get detailed system information.
    
    Requires admin role.
    """
    try:
        health_data = check_system_health()
        
        # Add more detailed system info for admins
        health_data["environment"] = {
            "app_version": "2.0.0",
            "auth_system": "JWT-based authentication",
            "database_connections": "Active"
        }
        
        return {
            "status": "healthy",
            **health_data
        }
    except Exception as e:
        logger.error(f"System info check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "detail": str(e)
            }
        )