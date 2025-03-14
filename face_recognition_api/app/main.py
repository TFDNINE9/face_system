import logging
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .api.routes import api_router
from .utils import create_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
 
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static directories
    app.mount("/storage", StaticFiles(directory=settings.STORAGE_DIR), name="storage")
    app.mount("/temp", StaticFiles(directory=settings.TEMP_DIR), name="temp")
    
    # Include API routes
    app.include_router(api_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return create_response({
            "version": settings.API_VERSION,
            "status": "operational"
        })
    
  
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred"}
        )
    
    return app


app = create_app()

if __name__ == "__main__":
   
    for dir_path in [settings.STORAGE_DIR, settings.TEMP_DIR]:
        os.makedirs(os.path.abspath(dir_path), exist_ok=True)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=settings.SSL_KEYFILE,
        ssl_certfile=settings.SSL_CERTFILE,
        reload=False
    )