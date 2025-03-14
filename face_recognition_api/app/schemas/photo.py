from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class PhotoBase(BaseModel):
    event_id: str

class PhotoResponse(BaseModel):
    photo_id: str
    event_id: str
    original_filename: str
    upload_date: Optional[datetime] = None
    file_size: int
    processed: bool = False
    
    class Config:
        from_attributes = True

class PaginatedPhotoResponse(BaseModel):
    photos: List[PhotoResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int