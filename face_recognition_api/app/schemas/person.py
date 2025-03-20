from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr


class PersonBase(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
    
class PersonCreate(PersonBase):
    event_id : str
    
class PersonUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
    is_identified: Optional[bool] = None
    representative_face_id: Optional[str] = None
    
class FaceAppearance(BaseModel):
    face_id: str
    image_id: str
    confidence: float
    original_filename: str
    
class PersonListResponse(BaseModel):
    person_id: str
    name: str
    is_identified: bool
    face_count: int
    representative_face_id: Optional[str] = None
    
class PaginatedPersonListResponse(BaseModel):
    persons: List[PersonListResponse]
    total_count: int
    page: int
    page_size: int 
    total_pages: int
    
class PaginatedFaceAppearanceResponse(BaseModel):
    appearances: List[FaceAppearance]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    
class PersonDetailResponse(PersonBase):
    person_id: str
    representative_face_id: Optional[str] = None
    face_count: int
    appearances: PaginatedFaceAppearanceResponse
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
