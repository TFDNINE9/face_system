from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, Request
from typing import List, Dict, Any
from ..dependencies.auth import is_admin
from ..schemas.auth import UserResponse
from ..util.ip_security import IPSecurity
from ..utils import create_response

router = APIRouter(
    prefix="/admin/security",
    tags=["Security Administration"],
)

logger = logging.getLogger(__name__)



@router.get("/blacklist", response_model=List[Dict[str, Any]])
async def list_blacklisted_ips(
    admin_user: UserResponse = Depends(is_admin)
):
    """
    List all currently blacklisted IP addresses.
    
    Requires admin privileges.
    """
    try:
        ip_security = IPSecurity()
        blacklisted_ips = await ip_security.get_all_blacklisted_ips()
        
        if not blacklisted_ips:
            return create_response(
                body=[{
                    "message": "No IP addresses are currently blacklisted",
                    "count": 0
                }]
            )
        
        # Format the response with additional details
        response_data = []
        for ip_data in blacklisted_ips:
            response_data.append({
                "ip_address": ip_data["ip_address"],
                "remaining_seconds": ip_data["remaining_seconds"],
                "expires_at": ip_data["expires_at"].isoformat() if isinstance(ip_data["expires_at"], datetime) else ip_data["expires_at"],
                "blacklisted": True
            })
        
        return create_response(
            body=response_data
        )
    except Exception as e:
        logger.error(f"Error listing blacklisted IPs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/blacklist/{ip_address}", status_code=status.HTTP_204_NO_CONTENT)
async def unblacklist_ip(
    ip_address: str = Path(..., description="IP address to unblacklist"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Remove an IP address from the blacklist.
    
    Requires admin privileges.
    """
    try:
        ip_security = IPSecurity()
        result = await ip_security.unblacklist_ip(ip_address)
        
        if result:
            return create_response(
                body={},
                status_code=status.HTTP_204_NO_CONTENT
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to unblacklist IP address"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/failed-attempts", response_model=List[Dict[str, Any]])
async def list_all_failed_attempts(
    admin_user: UserResponse = Depends(is_admin)
):
    """
    List all IP addresses with failed login attempts.
    
    Requires admin privileges.
    """
    try:
        ip_security = IPSecurity()
        all_attempts = await ip_security.get_all_failed_attempts()
        
        if not all_attempts:
            return create_response(
                body=[{
                    "message": "No failed login attempts recorded",
                    "count": 0
                }]
            )
        
        # Format the response
        return create_response(
            body=all_attempts
        )
    except Exception as e:
        logger.error(f"Error listing failed attempts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
        
@router.post("/blacklist/{ip_address}", status_code=status.HTTP_201_CREATED)
async def manually_blacklist_ip(
    ip_address: str = Path(..., description="IP address to blacklist"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Manually blacklist an IP address.
    
    Requires admin privileges.
    """
    try:
        ip_security = IPSecurity()
        
        # Artificially create enough failed attempts to trigger blacklisting
        for i in range(5):  # Assuming threshold is 5 or less
            await ip_security.record_failed_attempt(ip_address)
        
        # Check if blacklisting was successful
        is_blacklisted = await ip_security.is_ip_blacklisted(ip_address)
        blacklist_info = await ip_security.get_blacklist_details(ip_address)
        
        if is_blacklisted:
            return create_response(
                body={
                    "message": f"IP address {ip_address} has been blacklisted",
                    "blacklist_details": blacklist_info
                },
                status_code=status.HTTP_201_CREATED
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to blacklist IP address"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
        
        
@router.delete("/blacklist", status_code=status.HTTP_200_OK)
async def clear_all_blacklisted_ips(
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Clear all blacklisted IP addresses.
    
    Requires admin privileges.
    """
    try:
        ip_security = IPSecurity()
        result = await ip_security.clear_all_blacklisted_ips()
        
        if result["success"]:
            return create_response(
                body={
                    "message": f"Successfully cleared {result['cleared_count']} blacklisted IP addresses",
                    "cleared_count": result["cleared_count"],
                    "total_count": result["total_count"]
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear blacklisted IPs: {result.get('error', 'Unknown error')}"
            )
    except Exception as e:
        logger.error(f"Error clearing blacklisted IPs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )