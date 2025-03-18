import logging
import uuid
import os
import math
from PIL import Image
from fastapi import UploadFile, status
from ..database import get_db_connection, get_db_transaction
from ..config import settings
from .error_handling import (
    handle_service_error,
    NotFoundError,
    ValidationError,
    DatabaseError
)

logger = logging.getLogger(__name__)

@handle_service_error
def get_event_photos(event_id: str, page: int = 1, page_size: int = 20):
    try:
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid event_id format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )
        if page < 1:
            raise ValidationError(
                "Page number must be greater than or equal to 1",
                details={"field": "page", "value": page}
            )
        
        if page_size < 1 or page_size > 100:
            raise ValidationError(
                "Page size must be between 1 and 100",
                details={"field": "page_size", "value": page_size}
            )
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
    
            cursor.execute(
                "SELECT 1 FROM events WHERE event_id = ?",
                (event_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("Event", event_id)
            
            cursor.execute(
                "SELECT COUNT(*) FROM images WHERE event_id = ?",
                (event_id,)
            )
            
            total_count = cursor.fetchone()[0]
            total_pages = math.ceil(total_count / page_size)
            
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
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error retrieving event photos: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve event photos: {str(e)}", original_error=e)

@handle_service_error
async def upload_event_images(event_id: str, files: list[UploadFile]):
    try:
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid event ID format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )
        with get_db_transaction() as conn:
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
                raise NotFoundError("Event", event_id)
            
            customer_id = event_info[0]
            event_status = event_info[2]
            
            if event_status in ["archived"]:
                raise ValidationError(
                    f"Cannot upload to event with status: {event_status}",
                    details={"event_status": event_status}
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
                try:
                    if not file.filename:
                        logger.warning("Skipping file with no filename")
                        continue
                        
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
                except Exception as file_error:
                    logger.error(f"Error processing file {file.filename}: {str(file_error)}")
            
            return {
                "uploaded_count": len(uploaded_files),
                "uploaded_files": uploaded_files
            }
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to upload images: {str(e)}", original_error=e)

@handle_service_error
def get_photo(event_id: str, photo_id: str, include_storage_path: bool = False):
    try:
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid event ID format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )
            
        try:
            uuid.UUID(photo_id)
        except ValueError:
            raise ValidationError(
                f"Invalid photo ID format: {photo_id}. Must be a valid UUID.",
                details={"field": "photo_id", "value": photo_id}
            )
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
                raise NotFoundError("Photo", photo_id, {"event_id": event_id})
                
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
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error retrieving photo: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve photo: {str(e)}", original_error=e)

@handle_service_error
def get_photo_file_path(event_id: str, photo_id: str):
    try:
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid event ID format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )
            
        try:
            uuid.UUID(photo_id)
        except ValueError:
            raise ValidationError(
                f"Invalid photo ID format: {photo_id}. Must be a valid UUID.",
                details={"field": "photo_id", "value": photo_id}
            )
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT storage_path FROM images WHERE image_id = ? AND event_id = ?",
                (photo_id, event_id)
            )
            
            row = cursor.fetchone()
            
            if not row:
                raise NotFoundError("Photo", photo_id, {"event_id": event_id})
                
            storage_path = row[0]
            
            if not os.path.exists(storage_path):
                raise NotFoundError(
                    "Photo file", storage_path, 
                    {"event_id": event_id, "photo_id": photo_id}
                )
                
            return storage_path
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error retrieving photo file path: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve photo file path: {str(e)}", original_error=e)
    
    
    
@handle_service_error
def get_face_photo_file_path(event_id: str, face_id: str):
    try:
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid event ID format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )
            
        try:
            uuid.UUID(face_id)
        except ValueError:
            raise ValidationError(
                f"Invalid face ID format: {face_id}. Must be a valid UUID.",
                details={"field": "face_id", "value": face_id}
            )
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT f.face_path FROM faces f JOIN images i ON f.image_id = i.image_id WHERE f.face_id = ? AND i.event_id = ?
                """, (face_id, event_id)
            )
            
            row = cursor.fetchone()
            
            if not row:
                raise NotFoundError("Face Photo", face_id, {"event_id": event_id})
                
            storage_path = row[0]
            
            if not os.path.exists(storage_path):
                raise NotFoundError(
                    "Face file", storage_path, 
                    {"event_id": event_id, "face_id": face_id, "message": "Face file not found on server"}
                )
                
            return storage_path
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error retrieving photo file path: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve photo file path: {str(e)}", original_error=e)
    
    


@handle_service_error
def get_resized_photo_path(event_id: str, photo_id: str, size: int):
    try:
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid event ID format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )
            
        try:
            uuid.UUID(photo_id)
        except ValueError:
            raise ValidationError(
                f"Invalid photo ID format: {photo_id}. Must be a valid UUID.",
                details={"field": "photo_id", "value": photo_id}
            )
        if size < 16 or size > 2048:
            raise ValidationError(
                "Size must be between 16 and 2048 pixels",
                details={"field": "size", "value": size}
            )

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
            raise ValidationError(
                f"Failed to resize photo: {str(e)}",
                details={"original_path": original_path, "size": size}
            )
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error processing photo resize: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to process photo resize: {str(e)}", original_error=e)