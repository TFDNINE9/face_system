import logging
import uuid

from face_recognition_api.app.schemas.person import PersonUpdate
from face_recognition_api.app.services.error_handling import handle_service_error
from main import get_db_connection
from .error_handling import (
    handle_service_error,
    NotFoundError,
    ValidationError,
    DatabaseError
)


logger = logging.getLogger(__name__)

@handle_service_error
def get_person_by_event(event_id : str, page: int = 1, page_size: int = 20):
    try:
        try:
            uuid.UUID(event_id)
        except ValueError:
           raise ValidationError(
                f"invalid event id format: {event_id}. Must be a valid UUID",
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
                details={"field":"page_size", "value":page_size}
            )
            
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT 1 from events WHERE event_id = ?",
                (event_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("Event", event_id)
            
            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM persons
                WHERE event_id = ?
                """,
                (event_id,)
            )
            
            total_count = cursor.fetchone()[0]
            
            total_pages = (total_count + page_size -1) // page_size
            
            if page > total_pages and total_pages > 0:
                page = total_pages
                
            offset = (page - 1) * page_size
            
            cursor.execute(
                """SELECT p.person_id, p.name, p.is_identified, p.representative_face_id,
                       (SELECT COUNT(*) FROM person_appearances pa WHERE pa.person_id = p.person_id) as face_count
                FROM persons p
                WHERE p.event_id = ?
                ORDER BY face_count DESC, p.name
                OFFSET ? ROWS
                FETCH NEXT ? ROWS ONLY""",
                (event_id, offset, page_size)
            )
            
            # cursor.execute(
            #     """
            #     SELECT p.person_id, p.name, p.is_identified, p.representative_face_id,
            #            COUNT(pa.face_id) as face_count
            #     FROM persons p
            #     LEFT JOIN person_appearances pa ON p.person_id = pa.person_id
            #     WHERE p.event_id = ?
            #     GROUP BY p.person_id, p.name, p.is_identified, p.representative_face_id
            #     ORDER BY face_count DESC, p.name
            #     """,
            #     (event_id,)
            # )
            
            persons = []
            columns = [column[0] for column in cursor.description]
            
            for row in cursor.fetchall():
                person_dict = dict(zip(columns, row))
                
                person_data = {
                    "person_id" : str(person_dict["person_id"]),
                    "name" : person_dict["name"],
                    "is_identified": bool(person_dict["is_identified"]),
                    "face_count": person_dict["face_count"],
                    "representative_face_id": str(person_dict["representative_face_id"]) if person_dict["representative_face_id"] else None
                }
                
                persons.append(person_data)
                
            return {
                "persons": persons,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error getting persons by event: {str(e)}")
        raise DatabaseError(f"Failed to get persons : {str(e)}", original_error=e)
    
@handle_service_error
def get_person_detail(event_id : str, person_id: str, appearances_page: int = 1, appearances_page_size: int = 20):
    try:
        try:
            uuid.UUID(person_id)
        except ValueError:
            raise ValidationError(
                  f"Invalid person_id format",
                details={"person_id": person_id}
            )

        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                  f"Invalid event_id format",
                details={"event_id": event_id}
            )
        
        if appearances_page < 1:
            raise ValidationError(
                "Appearances page number must be greater than or equal to 1",
                details={"field": "appearances_page", "value": appearances_page}
            )
            
        if appearances_page_size < 1 or appearances_page_size > 100:
            raise ValidationError(
                "Appearances page size must be between 1 and 100",
                details={"field": "appearances_page_size", "value": appearances_page_size}
            )
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # cursor.execute(
            #     """ SELECT p.person_id, p.name, p.is_identified, p.representative_face_id,
            #            p.email, p.phone, p.notes, p.created_at, p.updated_at
            #     FROM persons p
            #     WHERE p.event_id = ? AND p.person_id = ?
            #     """,
            #     (event_id, person_id)
            # )
            
            cursor.execute(
                """
                SELECT p.person_id, p.name, p.is_identified, p.representative_face_id,
                       p.email, p.phone, p.notes, p.created_at, p.updated_at,
                       (SELECT COUNT(*) FROM person_appearances pa WHERE pa.person_id = p.person_id) as face_count
                FROM persons p
                WHERE p.event_id = ? AND p.person_id = ?
                """,
                (event_id, person_id)
            )
            
            row = cursor.fetchone()
            if not row:
                raise NotFoundError("Person", person_id, {"event_id": event_id})
            
            columns = [column[0] for column in cursor.description]
            
            person_dict = dict(zip(columns, row))
            
            total_appearances = person_dict["face_count"]
            
            total_pages = (total_appearances + appearances_page_size - 1) // appearances_page_size
            
            if appearances_page > total_pages and total_pages > 0:
                appearances_page = total_pages
                
            appearances_offset = (appearances_page - 1) * appearances_page_size
            
            cursor.execute(
                """
                SELECT pa.face_id, pa.image_id, pa.confidence,
                       i.original_filename
                FROM person_appearances pa
                JOIN images i ON pa.image_id = i.image_id
                WHERE pa.person_id = ?
                ORDER BY pa.confidence DESC
                OFFSET ? ROWS
                FETCH NEXT ? ROWS ONLY
                """,
                (person_id, appearances_offset, appearances_page_size)
            )
            
            
            # cursor.execute(
            #   """  SELECT pa.face_id, pa.image_id, pa.confidence,
            #         i.original_filename
            #     FROM person_appearances pa
            #     JOIN images i ON pa.image_id = i.image_id
            #     WHERE pa.person_id = ?
            #     ORDER BY pa.confidence DESC
            #     """,
            #     (person_id,)
            # )
            
            appearances = []
            appearance_columns = [column[0] for column in cursor.description]
            
            for appearance_row in cursor.fetchall():
                appearance_dict = dict(zip(appearance_columns, appearance_row))
                
                appearances.append({
                    "face_id": str(appearance_dict["face_id"]),
                    "image_id": str(appearance_dict["image_id"]),
                    "confidence": appearance_dict["confidence"],
                    "original_filename": appearance_dict["original_filename"]
                })
                
            person_data = {
                "person_id": str(person_dict["person_id"]),
                "name": person_dict["name"],
                "is_identified": bool(person_dict["is_identified"]),
                "email": person_dict["email"],
                "phone": person_dict["phone"],
                "notes": person_dict["notes"],
                "representative_face_id": str(person_dict["representative_face_id"]) if person_dict["representative_face_id"] else None,
                "face_count": total_appearances,
                "appearances": {
                    "appearances": appearances,
                    "total_count": total_appearances,
                    "page": appearances_page,
                    "page_size": appearances_page_size,
                    "total_pages": total_pages
                },
                "created_at": person_dict["created_at"].isoformat() if person_dict["created_at"] else None,
                "updated_at": person_dict["updated_at"].isoformat() if person_dict["updated_at"] else None
            }
            
            return person_data
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error getting person details: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to get person details : {str(e)}", original_error=e)
    
@handle_service_error
def update_person(event_id : str, person_id: str, person_data: PersonUpdate):
    try:
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid format event_id {event_id}",
                details={"event_id": event_id}
            )
        try:
            uuid.UUID(person_id)
        except ValueError:
            raise ValidationError(
                f"Invalid format person_id {person_id}",
                details={"person_id": person_id}
            )
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT 1 FROM persons WHERE person_id = ? AND event_id = ?", 
                (person_id, event_id)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("Person", person_id, {"event_id": event_id})

            update_fields = []
            update_values = []            

            if person_data.name is not None:
                update_fields.append("name = ?")
                update_values.append(person_data.name)
                
            if person_data.email is not None:
                update_fields.append("email = ?")
                update_values.append(person_data.email)
                
            if person_data.phone is not None:
                update_fields.append("phone = ?")
                update_values.append(person_data.phone)
                
            if person_data.notes is not None:
                update_fields.append("notes = ?")
                update_values.append(person_data.notes)
                
            if person_data.is_identified is not None:
                update_fields.append("is_identified = ?")
                update_values.append(person_data.is_identified)
                
            if person_data.representative_face_id is not None:
                update_fields.append("representative_face_id = ?")
                update_values.append(person_data.representative_face_id)
                
            if not update_fields:
                raise ValidationError("No fields provided for update")
                
            update_fields.append("updated_at = SYSUTCDATETIME()")
            update_values.append(person_id)
            
            query = f"UPDATE persons SET {', '.join(update_fields)} WHERE person_id = ?"
            cursor.execute(query, update_values)
            
        return get_person_detail(event_id, person_id)

    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error updating person: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to update person: {str(e)}", original_error=e)