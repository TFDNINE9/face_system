from fastapi import APIRouter, Depends, status, Path, Response, Query
from ..dependencies.auth import verify_api_key
from ..schemas.event import EventCreate, EventUpdate, EventResponse
from ..services.event import (
    get_all_events,
    get_events_by_customer,
    get_event,
    create_event,
    update_event,
    delete_event
)
from ..utils import create_response
from ..config import settings


router = APIRouter(
    prefix="/events",
    tags=["Events"],
)

@router.get("/", response_model=list[EventResponse],  dependencies=[Depends(verify_api_key)])
async def list_events(
    customer_id: str = Query(None, description="Filter events by customer ID")
):
    if customer_id:
        events = get_events_by_customer(customer_id)
    else:
        events = get_all_events()
    
    return create_response(body=events)

@router.get("/{event_id}", response_model=EventResponse)
async def get_event_by_id(
    event_id : str = Path(..., description="The ID of the event")
):
    event_data = get_event(event_id)
    return create_response(body=event_data)

@router.post("/", response_model=EventResponse, status_code=status.HTTP_201_CREATED,  dependencies=[Depends(verify_api_key)])
async def create_event(
    event_data: EventCreate
):
    new_event = create_event(event_data)
    
    headers = {"Location" :f"{settings.BASE_URL}/events/{new_event['event_id']}"}
    
    return create_response(
        headers=headers,
        body=new_event,
        status_code=status.HTTP_201_CREATED
    )
    
@router.put("/{event_id}", response_model=EventResponse,  dependencies=[Depends(verify_api_key)])
async def update_event(
      event_data: EventUpdate,
    event_id: str = Path(..., description="The ID of the event")
):
    updated_event = update_event(event_id, event_data)
    return create_response(
        body=updated_event
    )
    
    
@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT,  dependencies=[Depends(verify_api_key)])
async def delete_existing_event(
    event_id: str = Path(..., description="The ID of the event")
):
    delete_event(event_id)
    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )
    