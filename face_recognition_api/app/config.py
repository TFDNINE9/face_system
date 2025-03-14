import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Face Recognition System API V2"
    API_DESCRIPTION: str = "API for detection, clustering, and search with database integration"
    API_VERSION: str = "2.0.0"
    BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8000")
    API_KEY: str = os.getenv("API_KEY", "")
    
    # Path Settings
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "storage")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    
    # Database Settings
    DB_SERVER: str = os.getenv("DB_SERVER", "")
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_DRIVER: str = os.getenv("DB_DRIVER", "{ODBC Driver 17 for SQL Server}")
    
    # SSL Settings
    SSL_KEYFILE: str = os.getenv("SSL_KEYFILE", "")
    SSL_CERTFILE: str = os.getenv("SSL_CERTFILE", "")
    
    @property
    def DB_CONNECTION_STRING(self) -> str:
   
        return (
            f"DRIVER={self.DB_DRIVER};"
            f"SERVER={self.DB_SERVER};"
            f"DATABASE={self.DB_NAME};"
            f"UID={self.DB_USER};"
            f"PWD={self.DB_PASSWORD}"
        )
    
    @property
    def FACE_SYSTEM_CONFIG(self):
        from face_system_db import FaceSystemConfig
        
        config = FaceSystemConfig()
        config.dirs['base_storage'] = self.STORAGE_DIR
        config.dirs['temp'] = self.TEMP_DIR
        config.db = {
            'server': self.DB_SERVER,
            'database': self.DB_NAME,
            'username': self.DB_USER,
            'password': self.DB_PASSWORD,
            'driver': self.DB_DRIVER
        }
        
        return config


settings = Settings()

os.makedirs(settings.STORAGE_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)