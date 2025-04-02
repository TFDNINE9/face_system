import logging
import time
from typing import Optional, Callable, Dict, Union, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from fastapi.responses import JSONResponse

from ..util.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

class RateLimitConfig:
    """Configuration for different rate limit types."""
    
    def __init__(self):
        # Default rate limits
        self.default = {"max_requests": 100, "window_seconds": 60}
        
        # Path-specific rate limits (override defaults)
        self.paths = {
            # Authentication endpoints - more strict
            "/auth/login": {"max_requests": 10, "window_seconds": 120},
            "/auth/refresh": {"max_requests": 10, "window_seconds": 60},
            "/auth/password-reset": {"max_requests": 5, "window_seconds": 300},
            "/auth/password-reset-confirm": {"max_requests": 5, "window_seconds": 300},
            
            # Health check - very lenient
            "/health": {"max_requests": 300, "window_seconds": 60},
            
            # API endpoints with authentication - moderate
            "/api/": {"max_requests": 60, "window_seconds": 60},
            
            # Processing endpoints - more restrictive due to resource usage
            "/events/{event_id}/process": {"max_requests": 10, "window_seconds": 60},
            "/events/{event_id}/search": {"max_requests": 20, "window_seconds": 60}
        }

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests based on client IP and path.
    Uses Redis for distributed rate limiting.
    
    Features:
    - Different rate limits for different endpoints
    - IP-based rate limiting
    - Token bucket algorithm via Redis
    - Fallback to in-memory if Redis is unavailable
    - Proper rate limit headers
    """
    
    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            config: Rate limit configuration
        """
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.limiter = get_rate_limiter()
        logger.info("Rate limiting middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process the request through rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        # Skip rate limiting for certain paths
        if self._should_skip_rate_limiting(request.url.path):
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)
        
        # Determine which rate limit to apply
        rate_limit = self._get_rate_limit_for_path(request.url.path)
        max_requests = rate_limit["max_requests"]
        window_seconds = rate_limit["window_seconds"]
        
        # Create rate limit key (separate limits for different paths)
        path_key = self._normalize_path(request.url.path)
        rate_limit_key = f"ip:{client_ip}:{path_key}"
        
        # Check if request should be limited
        is_limited, rate_info = await self.limiter.is_rate_limited(
            rate_limit_key, 
            max_requests,
            window_seconds
        )
        
        # Prepare rate limit headers
        rate_limit_headers = {
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset"])
        }
        
        if is_limited:
            logger.warning(f"Rate limit exceeded for {client_ip} on {path_key}")
            
            # Add Retry-After header if retry time is available
            if rate_info["retry_after"] > 0:
                rate_limit_headers["Retry-After"] = str(rate_info["retry_after"])
            
            # Return 429 Too Many Requests
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "message": "Too many requests. Please try again later."
                },
                headers=rate_limit_headers
            )
        
        # Process the request normally
        response = await call_next(request)
        
        # Add rate limit headers to response
        for header, value in rate_limit_headers.items():
            response.headers[header] = value
        
        return response
    
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
    
    def _get_rate_limit_for_path(self, path: str) -> Dict[str, int]:
        """
        Get rate limit configuration for the request path.
        
        Args:
            path: Request URL path
            
        Returns:
            Dict with rate limit configuration
        """
        # Check for exact path match
        if path in self.config.paths:
            return self.config.paths[path]
        
        # Check for path prefix matches
        for prefix, limit in self.config.paths.items():
            if prefix.endswith("/"):
                if path.startswith(prefix):
                    return limit
            elif prefix.endswith("/{"):
                # Handle parameterized paths
                base_prefix = prefix.split("/{")[0]
                if path.startswith(base_prefix):
                    return limit
        
        # Default rate limit
        return self.config.default
    
    def _should_skip_rate_limiting(self, path: str) -> bool:
        """
        Check if rate limiting should be skipped for this path.
        
        Args:
            path: Request URL path
            
        Returns:
            Boolean indicating if rate limiting should be skipped
        """
        # Skip for static files or other non-API paths
        skip_prefixes = [
            "/static/",
            "/storage/",
            "/temp/",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        
        for prefix in skip_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for rate limiting by replacing path parameters with placeholders.
        
        Args:
            path: Request URL path
            
        Returns:
            Normalized path string
        """
        segments = path.strip("/").split("/")
        normalized = []
        
        # Replace UUIDs and numeric IDs with placeholders
        for segment in segments:
            # Check for UUID-like segments
            if len(segment) > 30 and "-" in segment:
                normalized.append("{id}")
            # Check for numeric IDs
            elif segment.isdigit():
                normalized.append("{num}")
            else:
                normalized.append(segment)
        
        return "/".join(normalized)