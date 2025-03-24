from fastapi import APIRouter, Depends, status, Path, Response
from ..schemas.auth import UserResponse
from ..dependencies.auth import get_current_active_user, is_admin
from ..schemas.customer import CustomerCreate, CustomerUpdate, CustomerResponse
from ..services.customer import (
    get_all_customers, 
    get_customer,
    create_customer,
    update_customer,
    delete_customer
)
from ..utils import create_response
from ..config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/customers",
    tags=["Customers"],
)

@router.get("/", response_model=list[CustomerResponse])
async def get_customers(current_user: UserResponse = Depends(is_admin)):
    customers = await get_all_customers()
    return create_response(body=customers)

@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer_by_id(customer_id: str = Path(..., description="The ID of the customer to get"), current_user: UserResponse = Depends(get_current_active_user)):
    customer = await get_customer(customer_id)
    return create_response(body=customer)

@router.post("/", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_new_customer(customer_data: CustomerCreate, current_user: UserResponse = Depends(is_admin)):
    new_customer = await create_customer(customer_data)
    headers = {"Location": f"{settings.BASE_URL}/customers/{new_customer['id']}"}
    
    return create_response(
        body=new_customer,
        status_code=status.HTTP_201_CREATED,
        headers=headers
    )

@router.put("/{customer_id}", response_model=CustomerResponse)
async def update_existing_customer(
    customer_id: str, 
    customer_data: CustomerUpdate,
    current_user: UserResponse = Depends(is_admin)
):
    updated_customer = await update_customer(customer_id, customer_data)
    return create_response(body=updated_customer)

@router.delete("/{customer_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_customer(customer_id: str, current_user: UserResponse = Depends(is_admin)):
    await delete_customer(customer_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)