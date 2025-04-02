import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Union
import asyncio
import redis.asyncio as redis
from ..config import settings

logger = logging.getLogger(__name__)

class RedisRateLimiter:
    """
    Rate limiter implementation using Redis.
    Uses a sliding window approach to limit requests within a time frame.
    """
    
    def __init__(self, redis_url: str = None):
        """
        Initialize the rate limiter with Redis connection.
        
        Args:
            redis_url: Redis connection URL (defaults to settings value)
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis = None
        
    async def get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
                logger.info("Connected to Redis successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                # Fall back to an in-memory limiter
                self._redis = InMemoryFallback()
                logger.warning("Using in-memory rate limiter (not suitable for production)")
        return self._redis
    
    async def is_rate_limited(
        self, 
        key: str, 
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, Union[int, float]]]:
        """
        Check if a key is rate limited.
        
        Args:
            key: Unique identifier for the rate limit (e.g., "ip:127.0.0.1:login")
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple containing:
            - Boolean indicating if the request should be rate limited
            - Dict with rate limit information for headers
        """
        redis_client = await self.get_redis()
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Create the full key with namespace
        full_key = f"rate_limit:{key}"
        
        try:
            # Add the current timestamp to the sorted set
            await redis_client.zadd(full_key, {str(current_time): current_time})
            
            # Remove timestamps outside the current window
            await redis_client.zremrangebyscore(full_key, 0, window_start)
            
            # Set expiry on the key to auto-cleanup
            await redis_client.expire(full_key, window_seconds * 2)
            
            # Count the number of requests in the current window
            request_count = await redis_client.zcard(full_key)
            
            # Calculate remaining requests and reset time
            remaining = max(0, max_requests - request_count)
            reset_at = current_time + window_seconds
            
            # Get retry-after time in seconds (if rate limited)
            retry_after = 0
            if request_count > max_requests:
                # Find the oldest timestamp in the window
                oldest = await redis_client.zrange(full_key, 0, 0, withscores=True)
                if oldest:
                    timestamp = oldest[0][1]
                    retry_after = max(0, int(timestamp + window_seconds - current_time))
            
            rate_info = {
                "limit": max_requests,
                "remaining": remaining,
                "reset": int(reset_at),
                "retry_after": retry_after
            }
            
            # Return True if rate limited, False otherwise
            return request_count > max_requests, rate_info
            
        except Exception as e:
            logger.error(f"Rate limiting error: {str(e)}")
            # On error, allow the request to proceed
            return False, {"limit": max_requests, "remaining": 1, "reset": int(current_time + window_seconds)}
    
    async def close(self):
        """Close Redis connection."""
        if self._redis and not isinstance(self._redis, InMemoryFallback):
            await self._redis.close()
            self._redis = None


class InMemoryFallback:
    """
    Fallback in-memory rate limiter to use when Redis is not available.
    Note: This won't work properly in a distributed environment.
    """
    
    def __init__(self):
        # Store request timestamps by key
        self.requests = {}
        # Run cleanup periodically
        asyncio.create_task(self._cleanup_task())
    
    async def _cleanup_task(self):
        """Background task to clean up expired entries."""
        while True:
            await asyncio.sleep(60)  # Clean up every minute
            self._cleanup()
    
    def _cleanup(self):
        """Remove expired entries from memory."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, timestamps in self.requests.items():
            # Keep track of which timestamps to remove
            to_remove = []
            for ts in timestamps:
                # Remove timestamps older than 1 hour (arbitrary cleanup window)
                if current_time - ts > 3600:
                    to_remove.append(ts)
            
            # Remove old timestamps
            for ts in to_remove:
                timestamps.remove(ts)
            
            # If no timestamps left, mark key for removal
            if not timestamps:
                keys_to_remove.append(key)
        
        # Remove empty keys
        for key in keys_to_remove:
            del self.requests[key]
    
    async def zadd(self, key: str, mapping: Dict[str, float]):
        """Simulate zadd command."""
        if key not in self.requests:
            self.requests[key] = []
        
        for _, score in mapping.items():
            self.requests[key].append(score)
        
        return 1
    
    async def zremrangebyscore(self, key: str, min_score: float, max_score: float):
        """Simulate zremrangebyscore command."""
        if key not in self.requests:
            return 0
        
        before_count = len(self.requests[key])
        self.requests[key] = [ts for ts in self.requests[key] if ts > max_score]
        after_count = len(self.requests[key])
        
        return before_count - after_count
    
    async def expire(self, key: str, seconds: int):
        """Simulate expire command (no-op in memory implementation)."""
        return 1
    
    async def zcard(self, key: str):
        """Simulate zcard command."""
        return len(self.requests.get(key, []))
    
    async def zrange(self, key: str, start: int, end: int, withscores=False):
        """Simulate zrange command."""
        if key not in self.requests or not self.requests[key]:
            return []
        
        # Sort timestamps
        sorted_timestamps = sorted(self.requests[key])
        result_range = sorted_timestamps[start:end+1]
        
        if withscores:
            return [(str(ts), ts) for ts in result_range]
        else:
            return [str(ts) for ts in result_range]
    
    async def ping(self):
        """Simulate ping command."""
        return True
    
    async def close(self):
        """Simulate close (no-op)."""
        pass


# Singleton pattern - single instance of rate limiter
_rate_limiter = None

def get_rate_limiter() -> RedisRateLimiter:
    """Get the rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RedisRateLimiter()
    return _rate_limiter