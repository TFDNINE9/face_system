from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
class EventStatusResponse(BaseModel):
    event_id: str
    event_name: str
    event_status: str
    processing_active: bool
    total_images: int
    processed_images: int
    total_clusters: int
    total_faces: int
    recent_jobs: List[JobResponse]
    
class ProcessingResponse(BaseModel):
    event_id: str
    unprocessed_images: int
    
class SearchMatch(BaseModel):
    cluster_id: str
    confidence: float
    face_count: int
    sample_face_id: str
    
class SearchResponse(BaseModel):
    matches: List[SearchMatch]
    job_id: Optional[str] = None
        
    