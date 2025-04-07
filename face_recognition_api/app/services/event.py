import logging
import uuid
import os
import pyodbc
from fastapi import HTTPException, status
from ..database import get_db_connection, get_db_transaction
from ..schemas.event import EventCreate, EventUpdate
from ..config import settings
from .error_handling import (
    handle_service_error,
    NotFoundError,
    ValidationError,
    DatabaseError
)

logger = logging.getLogger(__name__)

@handle_service_error
async def get_all_events():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.event_id, e.customer_id, e.name, e.description, e.event_date, st.status_code,
                       e.created_at, e.updated_at
                FROM events e
                JOIN event_status_types st ON e.status_id = st.status_id
                ORDER BY e.event_date DESC, e.name
                """
            )
            events = []
            columns = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                event_dict = dict(zip(columns, row))
                events.append({
                    "event_id": str(event_dict["event_id"]),
                    "customer_id": str(event_dict["customer_id"]),
                    "name": event_dict["name"],
                    "description": event_dict["description"],
                    "event_date": event_dict["event_date"].isoformat() if event_dict["event_date"] else None,
                    "status": event_dict["status_code"],
                    "created_at": event_dict["created_at"].isoformat() if event_dict["created_at"] else None,
                    "updated_at": event_dict["updated_at"].isoformat() if event_dict["updated_at"] else None
                })
            return events
    except Exception as e:
        logger.error(f"Database error retrieving events: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve events: {str(e)}", original_error=e)
        
@handle_service_error
async def get_events_by_customer(customer_id: str):
    try:
    
        try:
            uuid.UUID(customer_id)
        except ValueError:
            raise ValidationError(
                f"Invalid customer_id format: {customer_id}. Must be a valid UUID.",
                details={"field": "customer_id", "value": customer_id}
            )

        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            
            cursor.execute(
                "SELECT 1 FROM customers WHERE customer_id = ?",
                (customer_id,)
            )
            if not cursor.fetchone():
                raise NotFoundError("Customer", customer_id)
                
            cursor.execute(
                """
                SELECT e.event_id, e.customer_id, e.name, e.description, e.event_date, st.status_code,
                       e.created_at, e.updated_at
                FROM events e
                JOIN event_status_types st ON e.status_id = st.status_id
                WHERE e.customer_id = ?
                ORDER BY e.event_date DESC, e.name
                """,
                (customer_id,)
            )
            events = []
            columns = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                event_dict = dict(zip(columns, row))
                events.append({
                    "event_id": str(event_dict["event_id"]),
                    "customer_id": str(event_dict["customer_id"]),
                    "name": event_dict["name"],
                    "description": event_dict["description"],
                    "event_date": event_dict["event_date"].isoformat() if event_dict["event_date"] else None,
                    "status": event_dict["status_code"],
                    "created_at": event_dict["created_at"].isoformat() if event_dict["created_at"] else None,
                    "updated_at": event_dict["updated_at"].isoformat() if event_dict["updated_at"] else None
                })
            return events
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Database error retrieving customer events: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve customer events: {str(e)}", original_error=e)

@handle_service_error
async def get_event(event_id: str):
    try:
        try:
            uuid.UUID(event_id) 
        except ValueError:
            raise ValidationError(
                f"Invalid event_id format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )

        with get_db_connection() as conn:
            cursor = conn.cursor()
            logger.info(f"Executing query for event_id: {event_id}")
            cursor.execute(
                """
                SELECT e.event_id, e.customer_id, e.name, e.description, e.event_date, st.status_code,
                       e.created_at, e.updated_at
                FROM events e
                JOIN event_status_types st ON e.status_id = st.status_id
                WHERE e.event_id = ?
                """,
                (event_id,)
            )
            row = cursor.fetchone()
            logger.info(f"Query result: {row}")
            if not row:
                logger.info(f"No event found for event_id: {event_id}")
                raise NotFoundError("Event", event_id)
                
            event = {
                "event_id": str(row[0]),
                "customer_id": str(row[1]),
                "name": row[2],
                "description": row[3],
                "event_date": row[4].isoformat() if row[4] else None,
                "status": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
                "updated_at": row[7].isoformat() if row[7] else None
            }
            return event
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error retrieving event: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve event: {str(e)}", original_error=e)

@handle_service_error
async def create_event(event: EventCreate): 
    try:
        # Validate customer_id format
        try:
            uuid.UUID(event.customer_id)
        except ValueError:
            raise ValidationError(
                f"Invalid customer_id format: {event.customer_id}. Must be a valid UUID.",
                details={"field": "customer_id", "value": event.customer_id}
            )
            
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT 1 FROM customers WHERE customer_id = ?", 
                (event.customer_id,)
            )
            if not cursor.fetchone():
                raise NotFoundError("Customer", event.customer_id)

            cursor.execute(
                "SELECT status_id FROM event_status_types WHERE status_code = 'pending'"
            )
            status_id_row = cursor.fetchone()
            if not status_id_row:
                raise DatabaseError("Event status 'pending' not found in the database")

            status_id = status_id_row[0]
            event_id = str(uuid.uuid4())
            
            
            cursor.execute(
                """
                INSERT INTO events (event_id, customer_id, name, description, event_date, status_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, event.customer_id, event.name, event.description, event.event_date, status_id)
            )
        
            # Create storage directories
            customer_dir = os.path.join(settings.STORAGE_DIR, "customers", event.customer_id)
            event_dir = os.path.join(customer_dir, "events", event_id.lower())
            os.makedirs(os.path.join(event_dir, "original"), exist_ok=True)
            os.makedirs(os.path.join(event_dir, "faces"), exist_ok=True)
            os.makedirs(os.path.join(event_dir, "embeddings"), exist_ok=True)

            # Get the created event data
            cursor.execute(
                """
                SELECT e.event_id, e.customer_id, e.name, e.description, e.event_date, st.status_code,
                       e.created_at, e.updated_at
                FROM events e JOIN event_status_types st ON e.status_id = st.status_id
                WHERE e.event_id = ?
                """,
                (event_id,)
            )
            row = cursor.fetchone()
            
            event_data = {
                "event_id": str(row[0]),
                "customer_id": str(row[1]),
                "name": row[2],
                "description": row[3],
                "event_date": row[4].isoformat() if row[4] else None,
                "status": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
                "updated_at": row[7].isoformat() if row[7] else None
            }
            return event_data
    except (NotFoundError, ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Error creating event: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to create event: {str(e)}", original_error=e)

@handle_service_error
async def update_event(event_id: str, event: EventUpdate):
    try:
        # Validate event_id format
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid event_id format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )

        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            # Check if event exists
            cursor.execute(
                "SELECT customer_id FROM events WHERE event_id = ?", 
                (event_id,)
            )
            existing_event = cursor.fetchone()
            if not existing_event:
                raise NotFoundError("Event", event_id)

            existing_customer_id = existing_event[0]
            
            # If customer_id is being changed, verify the new customer exists
            if event.customer_id is not None and event.customer_id != existing_customer_id:
                try:
                    uuid.UUID(event.customer_id)
                except ValueError:
                    raise ValidationError(
                        f"Invalid customer_id format: {event.customer_id}. Must be a valid UUID.",
                        details={"field": "customer_id", "value": event.customer_id}
                    )
                    
                cursor.execute(
                    "SELECT 1 FROM customers WHERE customer_id = ?", 
                    (event.customer_id,)
                )
                if not cursor.fetchone():
                    raise NotFoundError("Customer", event.customer_id)

            # Prepare update fields
            update_fields = []
            update_values = []
            if event.name is not None:
                update_fields.append("name = ?")
                update_values.append(event.name)
            if event.description is not None:
                update_fields.append("description = ?")
                update_values.append(event.description)
            if event.event_date is not None:
                update_fields.append("event_date = ?")
                update_values.append(event.event_date)
            if event.customer_id is not None:
                update_fields.append("customer_id = ?")
                update_values.append(event.customer_id)
                
            if not update_fields:
                raise ValidationError("No valid fields provided for update")

            # Add updated_at field and event_id for WHERE clause
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            update_values.append(event_id)
            
            # Execute update
            query = f"UPDATE events SET {', '.join(update_fields)} WHERE event_id = ?"
            cursor.execute(query, tuple(update_values))
            
            # Get updated event
            cursor.execute(
                """
                SELECT e.event_id, e.customer_id, e.name, e.description, e.event_date, st.status_code,
                       e.created_at, e.updated_at
                FROM events e JOIN event_status_types st ON e.status_id = st.status_id
                WHERE e.event_id = ?
                """,
                (event_id,)
            )
            row = cursor.fetchone()
            updated_event = {
                "event_id": str(row[0]),
                "customer_id": str(row[1]),
                "name": row[2],
                "description": row[3],
                "event_date": row[4].isoformat() if row[4] else None,
                "status": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
                "updated_at": row[7].isoformat() if row[7] else None
            }
            return updated_event
    except (NotFoundError, ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Error updating event: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to update event: {str(e)}", original_error=e)

@handle_service_error
async def delete_event(event_id: str):
    """
    Delete an event.
    
    Args:
        event_id: ID of the event to delete
        
    Raises:
        ValidationError: If the event ID format is invalid
        NotFoundError: If the event is not found
        DatabaseError: If there's a database error
    """
    try:
        # Validate event_id format
        try:
            uuid.UUID(event_id)
        except ValueError:
            raise ValidationError(
                f"Invalid event_id format: {event_id}. Must be a valid UUID.",
                details={"field": "event_id", "value": event_id}
            )

        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            # Check if event exists
            cursor.execute(
                "SELECT 1 FROM events WHERE event_id = ?", 
                (event_id,)
            )
            if not cursor.fetchone():
                raise NotFoundError("Event", event_id)
            
            # Check if event has related photos, clusters, etc.
            cursor.execute(
                "SELECT COUNT(*) FROM images WHERE event_id = ?",
                (event_id,)
            )
            image_count = cursor.fetchone()[0]
            if image_count > 0:
                raise ValidationError(
                    f"Cannot delete event with {image_count} associated images",
                    details={"related_images": image_count}
                )
            
            # Delete the event
            cursor.execute(
                "DELETE FROM events WHERE event_id = ?", 
                (event_id,)
            )
    except (NotFoundError, ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Error deleting event: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to delete event: {str(e)}", original_error=e)