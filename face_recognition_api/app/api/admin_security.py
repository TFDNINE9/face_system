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

@router.get("/blacklist", response_model=List[Dict[str, Any]])
async def list_blacklisted_ips(
    admin_user: UserResponse = Depends(is_admin)
):
    """
    List all currently blacklisted IP addresses.
    
    Requires admin privileges.
    """
    try:
        # This requires implementation of a method to list all blacklisted IPs,
        # which would need to scan Redis keys or query a database.
        # For now, return a placeholder response
        return create_response(
            body=[
                {
                    "message": "This endpoint requires a custom implementation to list all blacklisted IPs from your storage.",
                    "note": "Consider implementing a persistent storage solution for blacklisted IPs."
                }
            ]
        )
    except Exception as e:
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

@router.get("/failed-attempts/{ip_address}")
async def get_failed_attempts(
    ip_address: str = Path(..., description="IP address to check"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Get the number of failed login attempts for an IP address.
    
    Requires admin privileges.
    """
    try:
        ip_security = IPSecurity()
        attempts = await ip_security.get_failed_attempts(ip_address)
        blacklist_info = await ip_security.get_blacklist_details(ip_address)
        
        return create_response(
            body={
                "ip_address": ip_address,
                "failed_attempts": attempts,
                "blacklisted": blacklist_info is not None,
                "blacklist_details": blacklist_info
            }
        )
    except Exception as e:
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