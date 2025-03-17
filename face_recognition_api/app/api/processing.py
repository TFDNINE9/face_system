from fastapi import APIRouter, Depends, Path, Query, BackgroundTasks, status, File, UploadFile
from ..dependencies.auth import verify_api_key
from ..schemas.processing import EventStatusResponse, ProcessingResponse, SearchResponse
from ..services.processing import (
    check_event_status,
    start_event_processing,
    search_face_in_event
)
from ..utils import create_response

router = APIRouter(
    tags=["Processing"],
)

@router.get("/events/{event_id}/status", response_model=EventStatusResponse,  dependencies=[Depends(verify_api_key)])
async def get_event_status(
    event_id: str = Path(..., description="The ID of the event")
):
    status_data = check_event_status(event_id)
    return create_response(body=status_data)

@router.post("/events/{event_id}/process", response_model=ProcessingResponse, status_code=status.HTTP_202_ACCEPTED,  dependencies=[Depends(verify_api_key)])
async def process_event(
    background_tasks: BackgroundTasks,
    event_id: str = Path(..., description="The ID of the event")
):
    processing_data = start_event_processing(event_id, background_tasks)
    return create_response(
        body=processing_data,
        status_code=status.HTTP_202_ACCEPTED
    )

@router.post("/events/{event_id}/search", response_model=SearchResponse)
async def search_face(
    event_id: str = Path(..., description="The ID of the event"),
    file: UploadFile = File(..., description="Image file with the face to search for"),
    threshold: float = Query(0.83, ge=0.5, le=1.0, description="Matching threshold (0.5 to 1.0)")
):
    search_results = await search_face_in_event(event_id, file, threshold)
    return create_response(body=search_results)