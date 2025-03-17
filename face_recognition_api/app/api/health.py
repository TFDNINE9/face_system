import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from ..services.health import check_system_health

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check():
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