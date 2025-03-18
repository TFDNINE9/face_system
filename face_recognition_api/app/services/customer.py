import logging
import uuid
from fastapi import HTTPException, status
from ..database import get_db_connection
from ..schemas.customer import CustomerCreate, CustomerUpdate
from .error_handling import (
    handle_service_error, 
    NotFoundError, 
    DatabaseError,
    ValidationError
)

logger = logging.getLogger(__name__)

@handle_service_error
async def get_all_customers():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT customer_id, name, email, phone, created_at FROM customers ORDER BY name"
            )
            customers = []
            columns = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                customer_dict = dict(zip(columns, row))
                customers.append({
                    "id": str(customer_dict["customer_id"]).lower(),
                    "name": customer_dict["name"],
                    "email": customer_dict["email"],
                    "phone": customer_dict["phone"],
                    "created_at": customer_dict["created_at"].isoformat() if customer_dict["created_at"] else None
                })
            
            return customers
    
    except Exception as e:
        logger.error(f"Error retrieving customers: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve customers: {str(e)}", original_error=e)

@handle_service_error
async def get_customer(customer_id: str):
    try:
        try:
            uuid.UUID(customer_id)
        except ValueError:
            raise ValidationError(
                f"Invalid customer ID format: {customer_id}. Must be a valid UUID",
                details={"field": "customer_id", "value": customer_id}
            )
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT customer_id, name, email, phone, created_at FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            row = cursor.fetchone()
            
            if not row:
                raise NotFoundError("Customer", customer_id)
                
            customer = {
                "id": str(row[0]).lower(),
                "name": row[1] if row[1] else None,
                "email": row[2] if row[2] else None,
                "phone": row[3] if row[3] else None,
                "created_at": row[4].isoformat() if row[4] else None
            }
            
            return customer
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error retrieving customer: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve customer: {str(e)}", original_error=e)

@handle_service_error
async def create_customer(customer: CustomerCreate):
    try:
        customer_id = str(uuid.uuid4())
        
        with get_db_connection() as conn:
            cursor = conn.cursor()

            if customer.email:
                cursor.execute(
                    "SELECT 1 FROM customers WHERE email = ?",
                    (customer.email,)
                )
                if cursor.fetchone():
                    raise ValidationError(
                        f"Customer with email {customer.email} already exists",
                        details={"field": "email", "value": customer.email}
                    )
            
            cursor.execute(
                """INSERT INTO customers (customer_id, name, email, phone) 
                   VALUES (?, ?, ?, ?)""", 
                (customer_id, customer.name, customer.email, customer.phone)
            )
            
            conn.commit()
        
        return {
            "id": customer_id,
            "name": customer.name,
            "email": customer.email,
            "phone": customer.phone,
            "created_at": None 
        }
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error creating customer: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to create customer: {str(e)}", original_error=e)

@handle_service_error
async def update_customer(customer_id: str, customer: CustomerUpdate):
    try:
        try:
            uuid.UUID(customer_id)
        except ValueError:
            raise ValidationError(
                f"Invalid customer ID format: {customer_id}. Must be a valid UUID",
                details={"field": "customer_id", "value": customer_id}
            )
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT customer_id FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("Customer", customer_id)
            
            if customer.email:
                cursor.execute(
                    "SELECT customer_id FROM customers WHERE email = ? AND customer_id != ?",
                    (customer.email, customer_id)
                )
                if cursor.fetchone():
                    raise ValidationError(
                        f"Customer with email {customer.email} already exists",
                        details={"field": "email", "value": customer.email}
                    )
            
            
            cursor.execute(
                """UPDATE customers 
                   SET name = ?, email = ?, phone = ? 
                   WHERE customer_id = ?""", 
                (customer.name, customer.email, customer.phone, customer_id)
            )
            
            if cursor.rowcount == 0:
                raise DatabaseError("Failed to update customer - no rows affected")
                
            conn.commit()
            
            
            cursor.execute(
                "SELECT customer_id, name, email, phone, created_at FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            row = cursor.fetchone()
            
            updated_customer = {
                "id": str(row[0]).lower(),
                "name": row[1],
                "email": row[2],
                "phone": row[3],
                "created_at": row[4].isoformat() if row[4] else None
            }
            
            return updated_customer
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error updating customer: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to update customer: {str(e)}", original_error=e)

@handle_service_error
async def delete_customer(customer_id: str):
    try:
        
        try:
            uuid.UUID(customer_id)
        except ValueError:
            raise ValidationError(
                f"Invalid customer ID format: {customer_id}. Must be a valid UUID",
                details={"field": "customer_id", "value": customer_id}
            )
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT customer_id FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("Customer", customer_id)

            cursor.execute(
                "SELECT COUNT(*) FROM events WHERE customer_id = ?",
                (customer_id,)
            )
            
            event_count = cursor.fetchone()[0]
            if event_count > 0:
                raise ValidationError(
                    f"Cannot delete customer with {event_count} associated events",
                    details={"related_events": event_count}
                )

            cursor.execute(
                "DELETE FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            conn.commit()
            
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error deleting customer: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to delete customer: {str(e)}", original_error=e)