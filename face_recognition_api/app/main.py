import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import ValidationError as PydanticValidationError
from .config import settings
from .api.routes import api_router
from .utils import create_response
from .services.error_handling import ServiceError
from fastapi.openapi.utils import get_openapi

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
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        openapi_schema["components"]["securitySchemes"] = {
            "JwtToken": {
                "type": "apiKey",
                "in": "header",
                "name": "Jwt-Token",
                "description": "Enter your JWT token"
            }
        }

        openapi_schema.pop("security", None)
        
        for path, path_item in openapi_schema['paths'].items():
            for method, operation in path_item.items():
                if not (path == "/auth/login" or path == "/health" or path == "/password-reset-confirm"):
                    operation['security'] = [{"JwtToken": []}]
                
                if path.startswith("/auth/") and path not in ["/auth/login", "/auth/password-reset-confirm"]:
                    operation['security'] = [{"JwtToken": []}]
                
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
        
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.mount("/storage", StaticFiles(directory=settings.STORAGE_DIR), name="storage")
    app.mount("/temp", StaticFiles(directory=settings.TEMP_DIR), name="temp")
    

    app.include_router(api_router)
    
    app.openapi = custom_openapi
    

    @app.get("/")
    async def root():
        return create_response({
            "version": settings.API_VERSION,
            "status": "operational"
        })
    

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        """Handle service-specific errors with appropriate status codes and formatting."""
        logger.error(f"Service error: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                 "message": exc.message,
            }
        )
    
    @app.exception_handler(PydanticValidationError)
    async def validation_error_handler(request: Request, exc: PydanticValidationError):
        first_error = exc.errors()[0] if exc.errors() else {"msg": "Validation error"}
        return JSONResponse(
            status_code=422,
            content={
                "message": first_error["msg"]
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "message" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "message": exc.detail["message"]
                }
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "message": str(exc.detail)
            }
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions with a simplified 500 error response."""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "message": "An unexpected error occurred"
            }
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