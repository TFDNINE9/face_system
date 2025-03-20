from fastapi import APIRouter, Depends, Path, Query, Response, status
from ..dependencies.auth import verify_api_key
from ..schemas.person import PaginatedPersonListResponse, PersonDetailResponse, PersonCreate, PersonUpdate
from ..services.person import (
    get_person_by_event,
    get_person_detail,
    # create_person,
    update_person,
    # delete_person
)
from ..utils import create_response

router = APIRouter(
    tags=["Persons"],
)

@router.get("/events/{event_id}/persons", response_model=PaginatedPersonListResponse,  dependencies=[Depends(verify_api_key)])
async def list_event_persons(
    event_id: str = Path(..., description="The ID of the event"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of persons per page")

):

    persons_data = get_person_by_event(event_id, page, page_size)
    return create_response(body=persons_data)

@router.get("/events/{event_id}/persons/{person_id}", response_model=PersonDetailResponse,  dependencies=[Depends(verify_api_key)])
async def get_person_details(
    event_id: str = Path(..., description="The ID of the event"),
    person_id: str = Path(..., description="The ID of the person"),
    appearances_page: int = Query(1, ge=1, description="Page number for appearances"),
    appearances_page_size: int = Query(20, ge=1, le=100, description="Number of appearances per page")
):
    person_data = get_person_detail(event_id, person_id, appearances_page, appearances_page_size)
    return create_response(body=person_data)


@router.put("/events/{event_id}/persons/{person_id}", response_model=PersonDetailResponse)
async def update_existing_person(
    event_id: str = Path(..., description="The ID of the event"),
    person_id: str = Path(..., description="The ID of the person"),
    person_data: PersonUpdate = None
):
    updated_person = await update_person(event_id, person_id, person_data)
    return create_response(body=updated_person)

# @router.post("/events/{event_id}/persons", response_model=PersonDetailResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_api_key)])
# async def create_new_person(
#     event_id: str = Path(..., description="The ID of the event"),
#     person_data: PersonCreate = None
# ):
#     if person_data:
#         person_data.event_id = event_id
    
#     new_person = await create_person(person_data)
    
#     headers = {"Location": f"/events/{event_id}/persons/{new_person['person_id']}"}
    
#     return create_response(
#         body=new_person,
#         status_code=status.HTTP_201_CREATED,
#         headers=headers
#     )

# @router.delete("/events/{event_id}/persons/{person_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_api_key)])
# async def delete_existing_person(
#     event_id: str = Path(..., description="The ID of the event"),
#     person_id: str = Path(..., description="The ID of the person")
# ):
#     """
#     Delete a person.
    
#     Returns no content on success.
#     """
#     await delete_person(event_id, person_id)
#     return Response(status_code=status.HTTP_204_NO_CONTENT)