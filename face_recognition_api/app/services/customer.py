import logging
import uuid
from fastapi import HTTPException, status

from face_recognition_api.app.services.auth import get_user_by_id
from ..database import get_db_connection, get_db_transaction
from ..schemas.customer import CustomerCreate, CustomerUpdate
from .error_handling import (
    ConflictError,
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
    
    
@handle_service_error
async def assign_user_to_customer(user_id:str, customer_id: str):
    try:
        try:
            uuid.UUID(user_id)
            uuid.UUID(customer_id)
        except ValueError:
            raise ValidationError(
            "Invalid UUID format for user_id or customer_id",
            details={"user_id": user_id, "customer_id": customer_id}
            )
        
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT 1 FROM auth_users WHERE user_id = ?", (user_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("User", user_id)
            
            cursor.execute(
                "SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)
            )
            
            if not cursor.fetchone():
                raise NotFoundError("Customer", customer_id)
            
            cursor.execute(
                "SELECT customer_id FROM auth_users WHERE user_id = ?", (user_id,)
            )
            
            current_customer = cursor.fetchone()
            if current_customer and current_customer[0] and current_customer[0] != customer_id:
                raise ConflictError(
                    "User is already assigned to a different customer",
                    details={"user_id": user_id, "current_customer_id": current_customer[0]}
                )
                
            cursor.execute(
                """UPDATE auth_users SET customer_id = ?, updated_at = SYSUTCDATETIME() WHERE user_id = ? """, (customer_id, user_id, )
            )
            
            cursor.execute(
                """SELECT g.group_id FROM auth_groups g WHERE g.name = 'user' """
            )
            
            user_group_row = cursor.fetchone()
            
            if user_group_row:
                user_group_id = user_group_row[0]
                
                cursor.execute(
                    """SELECT 1 FROM auth_user_groups WHERE user_id = ? AND group_id = ? """, (user_id, user_group_id)
                )
                
                if not cursor.fetchone():
                    cursor.execute(
                        """INSERT INTO auth_user_groups (user_id, group_id) VALUES (?,?)""", (user_id, user_group_id)
                    )
                    
            cursor.execute(
                """INSERT INTO auth_audit_logs(user_id, event_type,details) VALUES (?,?,?)""", (user_id, "customer_assignment", f"User asssigned to customer {customer_id}")
            )
            
            cursor.commit()
            
            user = get_user_by_id(user_id)
            
            return user
    except (NotFoundError, ValidationError, ConflictError):
        raise
    except Exception as e:
        logger.error(f"Error assigning user to customer: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to assign user to customer: {str(e)}", original_error=e)
    
    
@handle_service_error
def remove_user_from_customer(user_id: str):
    try:
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise ValidationError(
                "Invalid UUID format for user_id",
                details={"user_id": user_id}
            )
            
        # First, get user and validate they have a customer
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT customer_id FROM auth_users WHERE user_id = ?", (user_id,)
            )
            
            user_row = cursor.fetchone()
            if not user_row:
                raise NotFoundError("User", user_id)
            
            if not user_row[0]:
                raise ValidationError(
                    "User is not assigned to any customer",
                    details={"user_id": user_id}
                )
                
            previous_customer_id = user_row[0]
        
        # Then, perform the update in a separate transaction
        with get_db_transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """UPDATE auth_users SET customer_id = NULL, updated_at = SYSUTCDATETIME() WHERE user_id = ?""", 
                (user_id,)
            )
            
            cursor.execute(
                """INSERT INTO auth_audit_logs (user_id, event_type, details) VALUES (?, ?, ?)""", 
                (user_id, "customer_unassignment", f"User removed from customer {previous_customer_id}")
            )
        
        # Finally, get the updated user after the transaction is complete
        user = get_user_by_id(user_id)
        return user
            
    except (NotFoundError, ValidationError, ConflictError):
        raise  
    except Exception as e:
        logger.error(f"Error removing user from customer: {str(e)}", exc_info=True)
        raise DatabaseError(f"Failed to remove user from customer: {str(e)}", original_error=e)