import os
from typing import List
from fastapi import APIRouter, Depends, File, Path, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from ..dependencies.auth import get_current_active_user, has_group
from ..schemas.photo import PhotoResponse, PaginatedPhotoResponse
from ..schemas.auth import UserResponse
from ..services.photo import (
    get_event_photos,
    get_photo,
    get_photo_file_path,
    get_resized_photo_path,
    upload_event_images,
    get_face_photo_file_path
)
from ..utils import create_response

router = APIRouter(
    tags=["Photos"],
)

@router.get("/events/{event_id}/photos", response_model=PaginatedPhotoResponse,tags=["public"])
async def list_event_photos(
    event_id: str = Path(..., description="The ID of the event"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of photos per page"),
):
    """
    List photos for an event.
    
    Requires authentication.
    """
    photos_data = get_event_photos(event_id, page, page_size)
    return create_response(body=photos_data)

@router.post("/events/{event_id}/images", response_model=List[PhotoResponse])
async def upload_images_to_event(
    event_id: str = Path(..., description="The ID of the event"),
    files: List[UploadFile] = File(..., description="Image files to upload"),
    current_user: UserResponse = Depends(has_group(['admin', 'user']))
):
    """
    Upload images to an event.
    
    Requires user or admin role.
    """
    upload_result = await upload_event_images(event_id, files)
    return create_response(
        body=upload_result["uploaded_files"],
        status_code=status.HTTP_201_CREATED
    )

@router.get("/events/{event_id}/photos/{photo_id}", response_model=PhotoResponse, tags=["public"])
async def get_photo_by_id(
    event_id: str = Path(..., description="The ID of the event"),
    photo_id: str = Path(..., description="The ID of the photo"),
):
    """
    Get photo by ID.
    
    Requires authentication.
    """
    photo_data = get_photo(event_id, photo_id)
    return create_response(body=photo_data)

@router.get("/events/{event_id}/photos/{photo_id}/file", tags=["public"])
async def get_photo_file(
    event_id: str = Path(..., description="The ID of the event"),
    photo_id: str = Path(..., description="The ID of the photo"),
    size: int = Query(None, ge=16, le=2048, description="Width of the resized image in pixels"),
):
    """
    Get photo file or resized version.
    
    Requires authentication.
    """
    photo_data = get_photo(event_id, photo_id)
    
    if size is None:
        file_path = get_photo_file_path(event_id, photo_id)
        filename = photo_data["original_filename"]
    else:
        file_path = get_resized_photo_path(event_id, photo_id, size)
        filename = f"{os.path.splitext(photo_data['original_filename'])[0]}_{size}{os.path.splitext(photo_data['original_filename'])[1]}"
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        media_type = 'image/jpeg'
    elif ext == '.png':
        media_type = 'image/png'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )
    
@router.get("/events/{event_id}/representatives/{face_id}/file", tags=["public"])
async def get_face_photo_file(
    event_id: str = Path(..., description="The ID of the event"),
    face_id: str = Path(..., description="The ID of face photo"),
):
    """
    Get representative face photo file.
    
    Requires authentication.
    """
    file_path = get_face_photo_file_path(event_id, face_id)
      
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        media_type = 'image/jpeg'
    elif ext == '.png':
        media_type = 'image/png'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        path=file_path,
        media_type=media_type
    )