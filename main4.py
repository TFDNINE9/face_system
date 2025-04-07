import logging
import uuid
from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from fastapi import FastAPI, Response, UploadFile, File, HTTPException, Query, BackgroundTasks, status, Depends, Header, Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import os
from face_system_db import FaceSystemConfig, DatabaseFaceSystem
import pyodbc
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
TEMP_DIR = os.getenv("TEMP_DIR", "temp")

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

DB_CONFIG = {
    'server': os.getenv("DB_SERVER"),
    'database': os.getenv("DB_NAME"),
    'username': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'driver': os.getenv("DB_DRIVER", '{ODBC Driver 18 for SQL Server}'),
    'trust_server_certificate': 'yes',
    'encrypt': 'yes'
}

DB_CONNECTION_STRING = (
    f"DRIVER={DB_CONFIG['driver']};"
    f"SERVER={DB_CONFIG['server']};"
    f"DATABASE={DB_CONFIG['database']};"
    f"UID={DB_CONFIG['username']};"
    f"PWD={DB_CONFIG['password']};"
    f"TrustServerCertificate=yes;Encrypt=yes"

)

config = FaceSystemConfig()
config.dirs['base_storage'] = STORAGE_DIR
config.dirs['temp'] = TEMP_DIR
config.db = DB_CONFIG

db_face_system = DatabaseFaceSystem(connection_string=DB_CONNECTION_STRING, config=config)

class CustomerBase(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

class CustomerCreate(CustomerBase):
    pass

class CustomerResponse(CustomerBase):
    id: str
    created_at: Optional[str] = None

    class Config:
        orm_mode = True

app = FastAPI(
    title="Face Recognition System API V2",
    description="API for detection, clustering, and search with database integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")


@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        yield conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection error"
        )
    finally:
        if conn is not None:
            conn.close()

async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required"
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key
    
def create_response(body: dict, status_code: int = status.HTTP_200_OK, headers=None):
    return JSONResponse(
        content=body,
        status_code=status_code,
        headers=headers
    )
        
@app.get("/")
async def root():
    return create_response(
        {
            "version": "2.0.0",
            "status": "operational"
        }
    )
    
@app.get("/customers", dependencies=[Depends(verify_api_key)], response_model=list[CustomerResponse])
async def get_customers():
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
            
            return create_response(body=customers)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Can't get customers: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve customers: {str(e)}"
        )

@app.get("/customers/{customer_id}", dependencies=[Depends(verify_api_key)], response_model=CustomerResponse)
async def get_customer_by_id(customer_id: str = Path(..., description="The ID of the customer to get")):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT customer_id, name, email, phone, created_at FROM customers WHERE customer_id = ?", (customer_id,)
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
            
            return create_response(body=customer)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving customer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve customer: {str(e)}"
        )
        
@app.post("/customers", dependencies=[Depends(verify_api_key)], response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_customer(customer: CustomerCreate):
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
        
     
        response_data = {
            "id": customer_id,
            "name": customer.name,
            "email": customer.email,
            "phone": customer.phone,
            "created_at": None
        }
        
   
        headers = {"Location": f"{BASE_URL}/customers/{customer_id}"}
        
        return create_response(
            body=response_data,
            status_code=status.HTTP_201_CREATED,
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating customer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create customer: {str(e)}"
        )

@app.put("/customers/{customer_id}", dependencies=[Depends(verify_api_key)], response_model=CustomerResponse)
async def update_customer(customer_id: str, customer: CustomerCreate):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
           
            cursor.execute(
                "SELECT customer_id FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer not found"
                )
            
       
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
            
            return create_response(body=updated_customer)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating customer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update customer: {str(e)}"
        )

@app.delete("/customers/{customer_id}", dependencies=[Depends(verify_api_key)], status_code=status.HTTP_204_NO_CONTENT)
async def delete_customer(customer_id: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            
            cursor.execute(
                "SELECT customer_id FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer not found"
                )
            
            
            cursor.execute(
                "DELETE FROM customers WHERE customer_id = ?", 
                (customer_id,)
            )
            
            conn.commit()
            
            return Response(status_code=status.HTTP_204_NO_CONTENT)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting customer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete customer: {str(e)}"
        )

@app.get("/health")
async def health_check():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        
        return {
            "status": "healthy",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    for dir_path in [STORAGE_DIR, TEMP_DIR]:
        os.makedirs(os.path.abspath(dir_path), exist_ok=True)
        
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="\\\\str.innotech.com.la\\Storage\\Temp\\ssl\\privkey.pem",
        ssl_certfile="\\\\str.innotech.com.la\\Storage\\Temp\\ssl\\fullchain.pem",
        reload=False
    )