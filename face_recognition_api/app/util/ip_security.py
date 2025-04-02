import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as redis
from ..config import settings

logger = logging.getLogger(__name__)

class IPSecurity:
    """
    IP security module for tracking failed attempts and blacklisting IPs.
    Uses Redis to store data, with fallback to in-memory storage.
    """
    
    def __init__(self, redis_url: str = None):
        """
        Initialize the IP security module with Redis connection.
        
        Args:
            redis_url: Redis connection URL (defaults to settings value)
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis = None
        self.max_failed_attempts = settings.MAX_FAILED_ATTEMPTS
        self.blacklist_duration = settings.BLACKLIST_DURATION_MINUTES * 60  # Convert to seconds
        
    async def get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
                logger.info("Connected to Redis successfully for IP security")
            except Exception as e:
                logger.error(f"Failed to connect to Redis for IP security: {str(e)}")
                # Fall back to an in-memory store
                self._redis = InMemoryStore()
                logger.warning("Using in-memory IP security store (not suitable for production)")
        return self._redis
    
    async def is_ip_blacklisted(self, ip_address: str) -> bool:
        """
        Check if an IP address is blacklisted.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            Boolean indicating if IP is blacklisted
        """
        if not settings.IP_BLACKLIST_ENABLED:
            return False
            
        redis_client = await self.get_redis()
        key = f"ip_blacklist:{ip_address}"
        
        try:
            # Check if key exists in Redis
            exists = await redis_client.exists(key)
            return exists > 0
        except Exception as e:
            logger.error(f"Error checking blacklisted IP: {str(e)}")
            return False
    
    async def record_failed_attempt(self, ip_address: str) -> Tuple[bool, int]:
        """
        Record a failed login attempt and possibly blacklist the IP.
        
        Args:
            ip_address: IP address to record
            
        Returns:
            Tuple containing:
            - Boolean indicating if IP is now blacklisted
            - Integer with attempts count
        """
        if not settings.IP_BLACKLIST_ENABLED:
            return False, 0
            
        redis_client = await self.get_redis()
        attempts_key = f"ip_failed_attempts:{ip_address}"
        blacklist_key = f"ip_blacklist:{ip_address}"
        
        try:
            # Add failed attempt and set expiry (rolling 24 hour window)
            pipe = redis_client.pipeline()
            await pipe.incr(attempts_key)
            await pipe.expire(attempts_key, 86400)  # 24 hours
            results = await pipe.execute()
            
            attempts_count = results[0]
            
            # Check if attempts exceed threshold
            if attempts_count >= self.max_failed_attempts:
                # Blacklist the IP
                await redis_client.setex(blacklist_key, self.blacklist_duration, "1")
                logger.warning(f"IP {ip_address} has been blacklisted for {self.blacklist_duration} seconds")
                return True, attempts_count
                
            return False, attempts_count
            
        except Exception as e:
            logger.error(f"Error recording failed attempt: {str(e)}")
            return False, 0
    
    async def clear_failed_attempts(self, ip_address: str) -> bool:
        """
        Clear failed attempts for an IP after successful login.
        
        Args:
            ip_address: IP address to clear
            
        Returns:
            Boolean indicating success
        """
        if not settings.IP_BLACKLIST_ENABLED:
            return True
            
        redis_client = await self.get_redis()
        attempts_key = f"ip_failed_attempts:{ip_address}"
        
        try:
            # Remove the attempts counter
            await redis_client.delete(attempts_key)
            return True
        except Exception as e:
            logger.error(f"Error clearing failed attempts: {str(e)}")
            return False
    
    async def get_failed_attempts(self, ip_address: str) -> int:
        """
        Get the current count of failed attempts for an IP.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            Integer count of failed attempts
        """
        if not settings.IP_BLACKLIST_ENABLED:
            return 0
            
        redis_client = await self.get_redis()
        attempts_key = f"ip_failed_attempts:{ip_address}"
        
        try:
            # Get the current count
            count = await redis_client.get(attempts_key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Error getting failed attempts: {str(e)}")
            return 0
    
    async def get_blacklist_details(self, ip_address: str) -> Optional[Dict]:
        """
        Get details about a blacklisted IP.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            Dictionary with blacklist details or None
        """
        if not settings.IP_BLACKLIST_ENABLED:
            return None
            
        redis_client = await self.get_redis()
        blacklist_key = f"ip_blacklist:{ip_address}"
        
        try:
            # Check if blacklisted
            ttl = await redis_client.ttl(blacklist_key)
            if ttl > 0:
                return {
                    "ip_address": ip_address,
                    "blacklisted": True,
                    "remaining_seconds": ttl,
                    "expires_at": datetime.now() + timedelta(seconds=ttl)
                }
            return None
        except Exception as e:
            logger.error(f"Error getting blacklist details: {str(e)}")
            return None
            
    async def unblacklist_ip(self, ip_address: str) -> bool:
        """
        Remove an IP from the blacklist.
        
        Args:
            ip_address: IP address to unblacklist
            
        Returns:
            Boolean indicating success
        """
        if not settings.IP_BLACKLIST_ENABLED:
            return True
            
        redis_client = await self.get_redis()
        blacklist_key = f"ip_blacklist:{ip_address}"
        attempts_key = f"ip_failed_attempts:{ip_address}"
        
        try:
            # Remove from blacklist and clear attempts
            pipe = redis_client.pipeline()
            await pipe.delete(blacklist_key)
            await pipe.delete(attempts_key)
            await pipe.execute()
            
            logger.info(f"IP {ip_address} has been removed from blacklist")
            return True
        except Exception as e:
            logger.error(f"Error unblacklisting IP: {str(e)}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self._redis and not isinstance(self._redis, InMemoryStore):
            await self._redis.close()
            self._redis = None


class InMemoryStore:
    """Fallback in-memory storage for when Redis is not available."""
    
    def __init__(self):
        self.data = {}
        self.expirations = {}
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
    
    async def _cleanup_task(self):
        """Task to clean up expired keys."""
        while True:
            await asyncio.sleep(60)  # Run every minute
            self._cleanup()
    
    def _cleanup(self):
        """Remove expired keys from storage."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, expiry in self.expirations.items():
            if current_time > expiry:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.data:
                del self.data[key]
            if key in self.expirations:
                del self.expirations[key]
    
    async def exists(self, key: str) -> int:
        """Check if key exists."""
        self._cleanup()
        return 1 if key in self.data else 0
    
    async def incr(self, key: str) -> int:
        """Increment a key's value."""
        if key not in self.data:
            self.data[key] = 0
        self.data[key] += 1
        return self.data[key]
    
    async def expire(self, key: str, seconds: int) -> int:
        """Set expiration time for a key."""
        self.expirations[key] = time.time() + seconds
        return 1
    
    async def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set key with expiration."""
        self.data[key] = value
        self.expirations[key] = time.time() + seconds
        return True
    
    async def get(self, key: str) -> Optional[str]:
        """Get a key's value."""
        self._cleanup()
        return self.data.get(key)
    
    async def ttl(self, key: str) -> int:
        """Get time-to-live for a key."""
        if key not in self.expirations:
            return -2  # Key does not exist
        remaining = self.expirations[key] - time.time()
        return max(0, int(remaining))
    
    async def delete(self, key: str) -> int:
        """Delete a key."""
        if key in self.data:
            del self.data[key]
            if key in self.expirations:
                del self.expirations[key]
            return 1
        return 0
    
    async def ping(self) -> bool:
        """Simulate ping command."""
        return True
    
    async def pipeline(self):
        """Simulate Redis pipeline."""
        return self
    
    async def execute(self):
        """Execute pipeline commands."""
        return self.pipeline_results
    
    def __init__(self):
        self.data = {}
        self.expirations = {}
        self.pipeline_commands = []
        self.pipeline_results = []
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())