import logging
import pyodbc
from contextlib import contextmanager
from .services.error_handling import DatabaseError, ServiceError
from .config import settings

logger = logging.getLogger(__name__)

def get_db_config():
    """
    Get database configuration from settings.
    
    Returns:
        Dictionary with database configuration
    """
    return {
        'server': settings.DB_SERVER,
        'database': settings.DB_NAME,
        'username': settings.DB_USER,
        'password': settings.DB_PASSWORD,
        'driver': settings.DB_DRIVER
    }

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    
    Yields:
        A database connection object
        
    Raises:
        DatabaseError: If a database connection error occurs
        Other exceptions are passed through unchanged
    """
    conn = None
    try:
        conn = pyodbc.connect(settings.DB_CONNECTION_STRING)
        yield conn
    except pyodbc.Error as e:
        logger.error(f"Database connection error: {str(e)}", exc_info=True)
        raise DatabaseError(f"Database connection error: {str(e)}", original_error=e)
    except ServiceError:
        # Pass through service errors like NotFoundError without wrapping them
        raise
    except Exception as e:
        logger.error(f"Unexpected database error: {str(e)}", exc_info=True)
        raise DatabaseError(f"Unexpected database error: {str(e)}", original_error=e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {str(e)}")

@contextmanager
def get_db_transaction():
    """
    Context manager for database transactions.
    Automatically commits on success or rolls back on exception.
    
    Yields:
        A database connection object
        
    Raises:
        DatabaseError: If a database error occurs
        Other service exceptions are passed through unchanged
    """
    conn = None
    try:
        conn = pyodbc.connect(settings.DB_CONNECTION_STRING)
        yield conn
        conn.commit()
    except ServiceError:
        # Pass through service errors like NotFoundError without wrapping them
        if conn:
            conn.rollback()
        raise
    except pyodbc.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database transaction error: {str(e)}", exc_info=True)
        raise DatabaseError(f"Database transaction error: {str(e)}", original_error=e)
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Unexpected database error: {str(e)}", exc_info=True)
        raise DatabaseError(f"Unexpected database error: {str(e)}", original_error=e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {str(e)}")
                
# Initialize the face system from database configuration
from .face_system_imports import DatabaseFaceSystem, FaceSystemConfig

face_system_config = FaceSystemConfig()
face_system_config.dirs['base_storage'] = settings.STORAGE_DIR
face_system_config.dirs['temp'] = settings.TEMP_DIR
face_system_config.db = get_db_config()

db_face_system = DatabaseFaceSystem(
    connection_string=settings.DB_CONNECTION_STRING,
    config=face_system_config
)