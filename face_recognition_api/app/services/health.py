import logging
import os
from fastapi import HTTPException, status
from ..database import get_db_connection
from ..config import settings

logger = logging.getLogger(__name__)

def check_system_health():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
        
        storage_ok = os.path.exists(settings.STORAGE_DIR) and os.access(settings.STORAGE_DIR, os.W_OK)
        temp_ok = os.path.exists(settings.TEMP_DIR) and os.access(settings.TEMP_DIR, os.W_OK)
        
        return {
            "database": {
                "status": "connected",
                "version": version
            },
            "storage": {
                "status": "accessible" if storage_ok else "inaccessible",
                "path": settings.STORAGE_DIR
            },
            "temp_storage": {
                "status": "accessible" if temp_ok else "inaccessible",
                "path": settings.TEMP_DIR
            },
            "face_system": {
                "status": "initialized"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )