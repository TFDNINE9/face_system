from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from ..dependencies.auth import is_admin
from ..schemas.auth import (
    UserCreate, UserResponse, UserUpdate
)
from ..services.auth import (
    register_user, get_user_by_id, update_user, get_all_users,
    deactivate_user, activate_user, reset_user_password,
    add_user_to_group, remove_user_from_group
)
from ..utils import create_response

router = APIRouter(
    prefix="/users",
    tags=["User Management"],
)

@router.get("/", response_model=list[UserResponse])
async def list_users(
    admin_user: UserResponse = Depends(is_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    List all users.
    
    Requires admin privileges.
    """
    try:
        users = get_all_users(skip, limit)
        return create_response(body=users)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Create a new user.
    
    Requires admin privileges.
    """
    try:
        new_user = register_user(user_data)
        return create_response(
            body=new_user,
            status_code=status.HTTP_201_CREATED
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str = Path(..., description="User ID"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Get user by ID.
    
    Requires admin privileges.
    """
    try:
        user = get_user_by_id(user_id)
        return create_response(body=user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.put("/{user_id}", response_model=UserResponse)
async def update_user_details(
    user_id: str = Path(..., description="User ID"),
    user_data: UserUpdate = None,
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Update user details.
    
    Requires admin privileges.
    """
    try:
        updated_user = update_user(user_id, user_data)
        return create_response(body=updated_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/{user_id}/deactivate", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_user_account(
    user_id: str = Path(..., description="User ID"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Deactivate a user account.
    
    Requires admin privileges.
    """
    try:
        deactivate_user(user_id)
        return create_response(
            body={},
            status_code=status.HTTP_204_NO_CONTENT
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/{user_id}/activate", status_code=status.HTTP_204_NO_CONTENT)
async def activate_user_account(
    user_id: str = Path(..., description="User ID"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Activate a user account.
    
    Requires admin privileges.
    """
    try:
        activate_user(user_id)
        return create_response(
            body={},
            status_code=status.HTTP_204_NO_CONTENT
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/{user_id}/reset-password", status_code=status.HTTP_204_NO_CONTENT)
async def admin_reset_password(
    user_id: str = Path(..., description="User ID"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Reset user password and generate a temporary password.
    
    Requires admin privileges.
    """
    try:
        temp_password = reset_user_password(user_id)
        return create_response(
            body={"temporary_password": temp_password},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/{user_id}/groups/{group_name}", status_code=status.HTTP_204_NO_CONTENT)
async def add_to_group(
    user_id: str = Path(..., description="User ID"),
    group_name: str = Path(..., description="Group name"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Add user to a group.
    
    Requires admin privileges.
    """
    try:
        add_user_to_group(user_id, group_name)
        return create_response(
            body={},
            status_code=status.HTTP_204_NO_CONTENT
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/{user_id}/groups/{group_name}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_from_group(
    user_id: str = Path(..., description="User ID"),
    group_name: str = Path(..., description="Group name"),
    admin_user: UserResponse = Depends(is_admin)
):
    """
    Remove user from a group.
    
    Requires admin privileges.
    """
    try:
        remove_user_from_group(user_id, group_name)
        return create_response(
            body={},
            status_code=status.HTTP_204_NO_CONTENT
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )