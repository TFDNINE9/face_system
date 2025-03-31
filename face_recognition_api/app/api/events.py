from fastapi import APIRouter, Depends, status, Path, Response, Query
from ..dependencies.auth import get_current_active_user, has_group, is_admin
from ..schemas.event import EventCreate, EventUpdate, EventResponse
from ..schemas.auth import UserResponse
from ..services.event import (
    get_all_events,
    get_events_by_customer,
    get_event,
    create_event as create_event_service,
    update_event as update_event_service,
    delete_event
)
from ..utils import create_response
from ..config import settings


router = APIRouter(
    prefix="/events",
    tags=["Events"],
)

@router.get("/", response_model=list[EventResponse])
async def list_events(current_user: UserResponse = Depends(get_current_active_user)):
    """
    List events based on user permissions.
    
    For admins: Lists all events
    For users: Lists only events for their assigned customer
    
    Requires authentication.
    """
    user_groups = [group["name"] for group in current_user["groups"]]
    is_admin_user = "admin" in user_groups
    
    if is_admin_user:
        events = await get_all_events()
    else:
        if not current_user["customer_id"]:
            return create_response(body=[])
            
        events = await get_events_by_customer(current_user["customer_id"])
    
    return create_response(body=events)

    
@router.get("/{event_id}", response_model=EventResponse)
async def get_event_by_id(
    event_id : str = Path(..., description="The ID of the event"),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get event by ID.
    
    Requires authentication.
    """
    event_data = await get_event(event_id)
    return create_response(body=event_data)

@router.post("/", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def create_event(
    event_data: EventCreate,
    current_user: UserResponse = Depends(has_group(['admin', 'user']))
):
    """
    Create a new event.
    
    Requires user or admin role.
    """
    new_event = await create_event_service(event_data)
    headers = {"Location": f"{settings.BASE_URL}/events/{new_event['event_id']}"}
    return create_response(
        headers=headers,
        body=new_event,
        status_code=status.HTTP_201_CREATED
    )
    
@router.put("/{event_id}", response_model=EventResponse)
async def update_event(
    event_data: EventUpdate,
    event_id: str = Path(..., description="The ID of the event"),
    current_user: UserResponse = Depends(has_group(['admin', 'user']))
):
    """
    Update an existing event.
    
    Requires user or admin role.
    """
    updated_event = await update_event_service(event_id, event_data)
    return create_response(
        body=updated_event
    )
    
@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_event(
    event_id: str = Path(..., description="The ID of the event"),
    current_user: UserResponse = Depends(is_admin)
):
    """
    Delete an existing event.
    
    Requires admin role.
    """
    await delete_event(event_id)
    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )