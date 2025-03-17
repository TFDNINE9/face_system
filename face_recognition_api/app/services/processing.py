import logging
import os
import uuid
from datetime import datetime
from fastapi import HTTPException, status, UploadFile, BackgroundTasks
from ..database import get_db_connection
from ..config import settings

logger = logging.getLogger(__name__)

from ..database import db_face_system

def check_event_status(event_id: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get event status
            cursor.execute(
                """
                SELECT e.event_id, e.name, est.status_code as event_status,
                       (SELECT COUNT(*) FROM images WHERE event_id = e.event_id) as total_images,
                       (SELECT COUNT(*) FROM images WHERE event_id = e.event_id AND processed = 1) as processed_images,
                       (SELECT COUNT(*) FROM clusters WHERE event_id = e.event_id) as total_clusters,
                       (SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.image_id WHERE i.event_id = e.event_id) as total_faces
                FROM events e
                JOIN event_status_types est ON e.status_id = est.status_id
                WHERE e.event_id = ?
                """,
                (event_id,)
            )
            
            event_info = cursor.fetchone()
            if not event_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Event {event_id} not found"
                )
            
            cursor.execute(
                """
                SELECT j.job_id, jt.job_code, st.status_code, j.created_at, j.started_at, j.completed_at
                FROM processing_jobs j
                JOIN job_types jt ON j.job_type_id = jt.job_type_id
                JOIN status_types st ON j.status_id = st.status_id
                WHERE j.event_id = ?
                ORDER BY j.created_at DESC
                """,
                (event_id,)
            )
            
            jobs = []
            row = cursor.fetchone()
            while row and len(jobs) < 10:
                jobs.append({
                    "job_id": str(row[0]),
                    "job_type": row[1],
                    "status": row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                    "started_at": row[4].isoformat() if row[4] else None,
                    "completed_at": row[5].isoformat() if row[5] else None
                })
                row = cursor.fetchone()
            
            processing_active = any(job["status"] == "processing" for job in jobs)
            
            return {
                "event_id": str(event_info[0]),
                "event_name": event_info[1],
                "event_status": event_info[2],
                "processing_active": processing_active,
                "total_images": event_info[3],
                "processed_images": event_info[4],
                "total_clusters": event_info[5],
                "total_faces": event_info[6],
                "recent_jobs": jobs
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking event status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check event status: {str(e)}"
        )

def start_event_processing(event_id: str, background_tasks: BackgroundTasks):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT e.customer_id, e.status_id, st.status_code, 
                       (SELECT COUNT(*) FROM images WHERE event_id = e.event_id AND processed = 0) as unprocessed_images
                FROM events e
                JOIN event_status_types st ON e.status_id = st.status_id
                WHERE e.event_id = ?
                """,
                (event_id,)
            )
            
            event_info = cursor.fetchone()
            if not event_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Event {event_id} not found"
                )
            
            event_status = event_info[2]
            unprocessed_images = event_info[3]
            
            cursor.execute(
                """
                SELECT COUNT(*) FROM processing_jobs j
                JOIN status_types s ON j.status_id = s.status_id
                WHERE j.event_id = ? AND s.status_code IN ('queued', 'processing')
                """,
                (event_id,)
            )
            
            active_jobs = cursor.fetchone()[0]
            
            if active_jobs > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Processing is already in progress for this event"
                )
                
            if unprocessed_images == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No unprocessed images found for this event"
                )
        
        def process_images_background():
            try:
                result = db_face_system.process_event_images(event_id)
                logger.info(f"Background processing completed for event {event_id}: {result}")
            except Exception as e:
                logger.error(f"Error in background processing for event {event_id}: {e}")
        
        background_tasks.add_task(process_images_background)
        
        return {
            "event_id": event_id,
            "unprocessed_images": unprocessed_images
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )

async def search_face_in_event(event_id: str, file: UploadFile, threshold: float = 0.83):
    temp_file = None
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM clusters 
                WHERE event_id = ?
                """,
                (event_id,)
            )
            
            cluster_count = cursor.fetchone()[0]
            if cluster_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face clusters found for this event. Process images first."
                )
        
        file_content = await file.read()
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        temp_filename = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.jpg"
        temp_file = os.path.join(settings.TEMP_DIR, temp_filename)
        
        with open(temp_file, "wb") as f:
            f.write(file_content)
        
        results = db_face_system.search_face(temp_file, event_id, threshold)
        
        if results['status'] == 'success':
            return {
                "matches": results['matches'],
                "job_id": results.get('job_id')
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=results['message']
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in face search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search face: {str(e)}"
        )
    
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass