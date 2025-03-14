import logging
import pyodbc
from contextlib import contextmanager
from fastapi import HTTPException, status
from .face_system_imports import DatabaseFaceSystem, FaceSystemConfig
from .config import settings

logger = logging.getLogger(__name__)

db_face_system = DatabaseFaceSystem(
    connection_string=settings.DB_CONNECTION_STRING,
    config=settings.FACE_SYSTEM_CONFIG
)

@contextmanager
def get_db_connection():
  
    conn = None
    try:
        conn = pyodbc.connect(settings.DB_CONNECTION_STRING)
        yield conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection error"
        )
    finally:
        if conn is not None:
            conn.close()