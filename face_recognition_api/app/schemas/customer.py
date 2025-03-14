from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class CustomerBase(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(CustomerBase):
    pass

class CustomerResponse(CustomerBase):
    id: str
    created_at: Optional[str] = None

    class Config:
        orm_mode = True