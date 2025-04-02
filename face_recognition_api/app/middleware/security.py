import logging
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import JSONResponse

from ..util.ip_security import IPSecurity
from ..config import settings

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for applying security measures like IP blacklisting.
    Blocks requests from blacklisted IPs.
    """
    
    def __init__(self, app):
        """
        Initialize security middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.ip_security = IPSecurity()
        logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process the request through security checks.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        # Skip security for certain paths
        if not settings.IP_BLACKLIST_ENABLED or self._should_skip_security(request.url.path):
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blacklisted
        is_blacklisted = await self.ip_security.is_ip_blacklisted(client_ip)
        
        if is_blacklisted:
            # Get blacklist details for the response
            blacklist_details = await self.ip_security.get_blacklist_details(client_ip)
            
            logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
            
            # Return 403 Forbidden
            return JSONResponse(
                status_code=HTTP_403_FORBIDDEN,
                content={
                    "message": "Your IP address has been temporarily blocked due to multiple failed login attempts.",
                    "retry_after_seconds": blacklist_details["remaining_seconds"] if blacklist_details else None
                }
            )
        
        # Process the request normally
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request headers or connection info.
        
        Args:
            request: FastAPI request
            
        Returns:
            Client IP address as string
        """
        # Check for X-Forwarded-For header (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Use the first IP in the list (client IP)
            return forwarded_for.split(",")[0].strip()
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _should_skip_security(self, path: str) -> bool:
        """
        Check if security checks should be skipped for this path.
        
        Args:
            path: Request URL path
            
        Returns:
            Boolean indicating if security should be skipped
        """
        # Always allow health check and docs
        skip_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static/",
            "/storage/",
            "/temp/"
        ]
        
        for skip_path in skip_paths:
            if path.startswith(skip_path):
                return True
        
        return False