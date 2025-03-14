import logging
import uuid
import os
import math
from PIL import Image
from fastapi import HTTPException, UploadFile, status
from ..database import get_db_connection
from ..config import settings

logger = logging.getLogger(__name__)

def get_event_photos(event_id: str, page: int = 1, page_size: int = 20):

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
    
            cursor.execute(
                "SELECT 1 FROM events WHERE event_id = ?",
                (event_id,)
            )
            
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Event {event_id} not found"
                )
            
   
            cursor.execute(
                "SELECT COUNT(*) FROM images WHERE event_id = ?",
                (event_id,)
            )
            
            total_count = cursor.fetchone()[0]
            total_pages = math.ceil(total_count / page_size)
            
     
            if page < 1:
                page = 1
            if page > total_pages and total_pages > 0:
                page = total_pages
            
      
            offset = (page - 1) * page_size
            
           
            cursor.execute(
                """
                SELECT image_id, event_id, original_filename, storage_path, upload_date, file_size, processed
                FROM images
                WHERE event_id = ?
                ORDER BY upload_date DESC
                OFFSET ? ROWS
                FETCH NEXT ? ROWS ONLY
                """,
                (event_id, offset, page_size)
            )
            
            photos = []
            columns = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                photo_dict = dict(zip(columns, row))
              
                photos.append({
                    "photo_id": str(photo_dict["image_id"]),
                    "event_id": str(photo_dict["event_id"]),
                    "original_filename": photo_dict["original_filename"],
                    "upload_date": photo_dict["upload_date"].isoformat() if photo_dict["upload_date"] else None,
                    "file_size": photo_dict["file_size"],
                    "processed": bool(photo_dict["processed"])
                })
            
            return {
                "photos": photos,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving event photos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve event photos: {str(e)}"
        )

async def upload_event_images(event_id: str, files: list[UploadFile]):

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
        
            cursor.execute(
                """
                SELECT e.customer_id, e.status_id, st.status_code
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
            
            customer_id = event_info[0]
            event_status = event_info[2]
            
    
            if event_status in ["archived"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot upload to event with status: {event_status}"
                )
            
     
            event_dir = os.path.join(
                settings.STORAGE_DIR,
                "customers",
                str(customer_id),
                "events",
                str(event_id)
            )
            
            original_dir = os.path.join(event_dir, "original")
            os.makedirs(original_dir, exist_ok=True)
            
            uploaded_files = []
            
            for file in files:

                file_uuid = str(uuid.uuid4())
                original_filename = file.filename
                extension = os.path.splitext(original_filename)[1].lower()
                
                if extension not in ['.jpg', '.jpeg', '.png']:
                    continue  
                
                storage_filename = f"{file_uuid}{extension}"
                storage_path = os.path.join(original_dir, storage_filename)
                
    
                with open(storage_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
            
                image_id = str(uuid.uuid4())
                
                cursor.execute(
                    """
                    INSERT INTO images (image_id, event_id, original_filename, storage_path, file_size)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (image_id, event_id, original_filename, storage_path, len(content))
                )
                
                uploaded_files.append({
                    "image_id": image_id,
                    "original_filename": original_filename,
                    "file_size": len(content),
                    "processed": False
                })
            
            conn.commit()
            
            return {
                "uploaded_count": len(uploaded_files),
                "uploaded_files": uploaded_files
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload images: {str(e)}"
        )


def get_photo(event_id: str, photo_id: str, include_storage_path: bool = False):

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT image_id, event_id, original_filename, storage_path, upload_date, file_size, processed
                FROM images
                WHERE image_id = ? AND event_id = ?
                """,
                (photo_id, event_id)
            )
            
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Photo {photo_id} not found in event {event_id}"
                )
                
            photo = {
                "photo_id": str(row[0]),
                "event_id": str(row[1]),
                "original_filename": row[2],
                "upload_date": row[4].isoformat() if row[4] else None,
                "file_size": row[5],
                "processed": bool(row[6])
            }
            
    
            if include_storage_path:
                photo["storage_path"] = row[3]
            
            return photo
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving photo: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve photo: {str(e)}"
        )

def get_photo_file_path(event_id: str, photo_id: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT storage_path FROM images WHERE image_id = ? AND event_id = ?",
                (photo_id, event_id)
            )
            
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Photo {photo_id} not found in event {event_id}"
                )
                
            storage_path = row[0]
            
            if not os.path.exists(storage_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Photo file not found on storage"
                )
                
            return storage_path
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving photo file path: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve photo file path: {str(e)}"
        )

def get_resized_photo_path(event_id: str, photo_id: str, size: int):

    try:

        original_path = get_photo_file_path(event_id, photo_id)
        

        filename, ext = os.path.splitext(original_path)
        resized_dir = os.path.join(os.path.dirname(os.path.dirname(original_path)), "resized")
        os.makedirs(resized_dir, exist_ok=True)
        
        resized_path = os.path.join(resized_dir, f"{os.path.basename(filename)}_{size}{ext}")
        

        if os.path.exists(resized_path):
            return resized_path
        
   
        try:
            with Image.open(original_path) as img:
 
                width, height = img.size
                new_height = int((size / width) * height)
                

                resized_img = img.resize((size, new_height), Image.Resampling.LANCZOS)
                
         
                resized_img.save(resized_path, quality=90)
                
                return resized_path
                
        except Exception as e:
            logger.error(f"Error resizing photo: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to resize photo: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing photo resize: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process photo resize: {str(e)}"
        )