import logging
import uuid
from fastapi import HTTPException, status
from ..database import get_db_connection
from ..schemas.customer import CustomerCreate, CustomerUpdate

logger = logging.getLogger(__name__)

def get_all_customers():
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Can't get customers: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve customers: {str(e)}"
        )

def get_customer(customer_id: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT customer_id, name, email, phone, created_at FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer not found"
                )
                
            customer = {
                "id": str(row[0]).lower(),
                "name": row[1] if row[1] else None,
                "email": row[2] if row[2] else None,
                "phone": row[3] if row[3] else None,
                "created_at": row[4].isoformat() if row[4] else None
            }
            
            return customer
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving customer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve customer: {str(e)}"
        )

def create_customer(customer: CustomerCreate):
    try:
        customer_id = str(uuid.uuid4())
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating customer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create customer: {str(e)}"
        )

def update_customer(customer_id: str, customer: CustomerUpdate):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # First check if the customer exists
            cursor.execute(
                "SELECT customer_id FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer not found"
                )
            
            # Update the customer
            cursor.execute(
                """UPDATE customers 
                   SET name = ?, email = ?, phone = ? 
                   WHERE customer_id = ?""", 
                (customer.name, customer.email, customer.phone, customer_id)
            )
            
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update customer"
                )
                
            conn.commit()
            
            # Get the updated customer to return
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
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating customer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update customer: {str(e)}"
        )

def delete_customer(customer_id: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # First check if the customer exists
            cursor.execute(
                "SELECT customer_id FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer not found"
                )
            
            # Delete the customer
            cursor.execute(
                "DELETE FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            conn.commit()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting customer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete customer: {str(e)}"
        )