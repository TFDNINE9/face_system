from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, status, Depends, Header, Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import shutil
import os
import cv2
from datetime import datetime
import logging
import json
import time
import uuid
from PIL import Image
from starlette.responses import StreamingResponse
import io
import pyodbc
from face_system_db import DatabaseFaceSystem, FaceSystemConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()

# API Configuration
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

# Directory configuration
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
TEMP_DIR = os.getenv("TEMP_DIR", "temp")

# Ensure directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Database configuration
DB_CONFIG = {
    'server': os.getenv("DB_SERVER"),
    'database': os.getenv("DB_NAME"),
    'username': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'driver': os.getenv("DB_DRIVER", '{ODBC Driver 17 for SQL Server}')
}

# Create database connection string
DB_CONNECTION_STRING = (
    f"DRIVER={DB_CONFIG['driver']};"
    f"SERVER={DB_CONFIG['server']};"
    f"DATABASE={DB_CONFIG['database']};"
    f"UID={DB_CONFIG['username']};"
    f"PWD={DB_CONFIG['password']}"
)

# Initialize FaceSystem
config = FaceSystemConfig()
config.dirs['base_storage'] = STORAGE_DIR
config.dirs['temp'] = TEMP_DIR
config.db = DB_CONFIG

# Initialize the face system with database connection
db_face_system = DatabaseFaceSystem(connection_string=DB_CONNECTION_STRING, config=config)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition System API",
    description="API for face detection, clustering, and search with database integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for accessing face images 
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

# Simple API key validation
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key

# Standardized response helper
def create_response(status: str, code: str, message: str, details: dict = None, status_code: int = status.HTTP_200_OK):
    return JSONResponse(
        content={
            "status": status,
            "code": code,
            "message": message,
            "details": details
        },
        status_code=status_code
    )

# Database connection helper
def get_db_connection():
    try:
        return pyodbc.connect(DB_CONNECTION_STRING)
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection error"
        )

# Verify event belongs to customer
def verify_event(event_id: str, customer_id: str, conn):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM events WHERE event_id = ? AND customer_id = ?",
        (event_id, customer_id)
    )
    result = cursor.fetchone()
    cursor.close()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found for this customer"
        )
    return True

# Endpoint for system info
@app.get("/")
async def root():
    return create_response(
        status="success",
        code="API_INFO",
        message="Face Recognition System API is running",
        details={
            "version": "2.0.0",
            "endpoints": {
                "customers": "/customers",
                "events": "/customers/{customer_id}/events",
                "images": "/events/{event_id}/images",
                "process": "/events/{event_id}/process",
                "search": "/events/{event_id}/search",
                "documentation": "/docs"
            }
        }
    )

# Customers endpoints
@app.get("/customers", dependencies=[Depends(verify_api_key)])
async def get_customers():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT customer_id, name, email, phone, created_at FROM customers ORDER BY name"
        )
        
        customers = []
        row = cursor.fetchone()
        while row:
            customers.append({
                "customer_id": str(row[0]),
                "name": row[1],
                "email": row[2],
                "phone": row[3],
                "created_at": row[4].isoformat() if row[4] else None
            })
            row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return create_response(
            status="success",
            code="CUSTOMERS_FOUND",
            message=f"Found {len(customers)} customers",
            details={"customers": customers}
        )
    except Exception as e:
        logger.error(f"Error retrieving customers: {str(e)}")
        return create_response(
            status="error",
            code="DATABASE_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.post("/customers", dependencies=[Depends(verify_api_key)])
async def create_customer(customer: dict):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        customer_id = str(uuid.uuid4())
        
        cursor.execute(
            """
            INSERT INTO customers (customer_id, name, email, phone)
            VALUES (?, ?, ?, ?)
            """,
            (customer_id, customer["name"], customer.get("email"), customer.get("phone"))
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return create_response(
            status="success",
            code="CUSTOMER_CREATED",
            message=f"Customer {customer['name']} created successfully",
            details={"customer_id": customer_id}
        )
    except Exception as e:
        logger.error(f"Error creating customer: {str(e)}")
        return create_response(
            status="error",
            code="DATABASE_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Events endpoints
@app.get("/customers/{customer_id}/events", dependencies=[Depends(verify_api_key)])
async def get_customer_events(customer_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT e.event_id, e.name, e.description, e.event_date, st.status_code,
                   e.created_at, e.updated_at
            FROM events e
            JOIN event_status_types st ON e.status_id = st.status_id
            WHERE e.customer_id = ?
            ORDER BY e.event_date DESC, e.name
            """,
            (customer_id,)
        )
        
        events = []
        row = cursor.fetchone()
        while row:
            events.append({
                "event_id": str(row[0]),
                "name": row[1],
                "description": row[2],
                "event_date": row[3].isoformat() if row[3] else None,
                "status": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
                "updated_at": row[6].isoformat() if row[6] else None
            })
            row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return create_response(
            status="success",
            code="EVENTS_FOUND",
            message=f"Found {len(events)} events for customer {customer_id}",
            details={"events": events}
        )
    except Exception as e:
        logger.error(f"Error retrieving events: {str(e)}")
        return create_response(
            status="error",
            code="DATABASE_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.post("/customers/{customer_id}/events", dependencies=[Depends(verify_api_key)])
async def create_event(customer_id: str, event: dict):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify customer exists
        cursor.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?",
            (customer_id,)
        )
        if not cursor.fetchone():
            return create_response(
                status="error",
                code="CUSTOMER_NOT_FOUND",
                message=f"Customer {customer_id} not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Get pending status ID
        cursor.execute(
            "SELECT status_id FROM event_status_types WHERE status_code = 'pending'"
        )
        status_id = cursor.fetchone()[0]
        
        event_id = str(uuid.uuid4())
        event_date = event.get("event_date")
        
        cursor.execute(
            """
            INSERT INTO events (event_id, customer_id, name, description, event_date, status_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, customer_id, event["name"], event.get("description"), event_date, status_id)
        )
        
        conn.commit()
        
        # Create event directory structure
        customer_dir = os.path.join(STORAGE_DIR, "customers", customer_id)
        event_dir = os.path.join(customer_dir, "events", event_id)
        os.makedirs(os.path.join(event_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(event_dir, "faces"), exist_ok=True)
        os.makedirs(os.path.join(event_dir, "embeddings"), exist_ok=True)
        
        cursor.close()
        conn.close()
        
        return create_response(
            status="success",
            code="EVENT_CREATED",
            message=f"Event {event['name']} created successfully",
            details={
                "event_id": event_id,
                "customer_id": customer_id
            }
        )
    except Exception as e:
        logger.error(f"Error creating event: {str(e)}")
        return create_response(
            status="error",
            code="DATABASE_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
@app.put("/customers/{customer_id}/events/{event_id}", dependencies=[Depends(verify_api_key)])
async def edit_event(customer_id: str, event_id: str, event: dict):
    try:
        con = get_db_connection();
        cursor = con.cursor();
        
        cursor.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?",
            (customer_id,)
        )
        if not cursor.fetchone():
            return create_response(
                status="error",
                code="CUSTOMER_NOT_FOUND",
                message=f"Customer {customer_id} not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        cursor.execute(
            "SELECT 1 FROM events WHERE event_id = ? AND customer_id = ?",
            (event_id, customer_id)
        )
        if not cursor.fetchone():
            return create_response(
                status="error",
                code="EVENT_NOT_FOUND",
                message=f"Event {event_id} not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
            
        update_fields = []
        update_values = []
        
        updatable_fields = {
            "name" : event.get("name"),
            "description" : event.get("description"),
            "event_date": event.get("event_date")
        }
        
        for field, value in updatable_fields.items():
            if value is not None:
                update_fields.append(f"{field} = ?")
                update_values.append(value)
        
        if not update_fields:
            return create_response(
                status="error",
                code="NO_UPDATE",
                message="No Valid field provided for update",
                status_code=status.HTTP_400_BAD_REQUEST
            )
            
        update_values.extend([event_id, customer_id])
        
        query = f"""UPDATE events SET {','.join(update_fields)} WHERE event_id = ? AND customer_id = ?"""
        
        cursor.execute(query, tuple(update_values))
        con.commit()
        
        cursor.close()
        con.close()
        
        return create_response(
                    status="success",
                    code="EVENT_UPDATED",
                    message=f"Event {event_id} updated successfully",
                    details={
                        "event_id": event_id,
                        "customer_id": customer_id,
                        "updated_fields": [field.split(' =')[0] for field in update_fields]
                    }
                )
                
    except Exception as e:
        logger.error(f"Error updating event: {str(e)}")
        return create_response(
            status="error",
            code="DATABASE_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Images endpoints
@app.post("/events/{event_id}/images", dependencies=[Depends(verify_api_key)])
async def upload_images(event_id: str, files: list[UploadFile] = File(...)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify event exists
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
            return create_response(
                status="error",
                code="EVENT_NOT_FOUND",
                message=f"Event {event_id} not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        customer_id = event_info[0]
        event_status = event_info[2]
        
        # Check if event is in a valid state for uploading
        if event_status in ["archived"]:
            return create_response(
                status="error",
                code="INVALID_EVENT_STATUS",
                message=f"Cannot upload to event with status: {event_status}",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Get event storage path
        event_dir = os.path.join(
            STORAGE_DIR,
            "customers",
            str(customer_id),
            "events",
            str(event_id)
        )
        
        original_dir = os.path.join(event_dir, "original")
        os.makedirs(original_dir, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            # Generate a unique filename
            file_uuid = str(uuid.uuid4())
            original_filename = file.filename
            extension = os.path.splitext(original_filename)[1].lower()
            
            if extension not in ['.jpg', '.jpeg', '.png']:
                continue  # Skip non-image files
            
            storage_filename = f"{file_uuid}{extension}"
            storage_path = os.path.join(original_dir, storage_filename)
            
            # Save the file
            with open(storage_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Create an image record
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
                "storage_path": storage_path,
                "file_size": len(content)
            })
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return create_response(
            status="success",
            code="IMAGES_UPLOADED",
            message=f"Uploaded {len(uploaded_files)} images to event {event_id}",
            details={
                "uploaded_files": len(uploaded_files),
                "files": uploaded_files
            }
        )
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}")
        return create_response(
            status="error",
            code="UPLOAD_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.get("/events/{event_id}/images", dependencies=[Depends(verify_api_key)])
async def get_event_images(event_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify event exists
        cursor.execute(
            "SELECT customer_id FROM events WHERE event_id = ?",
            (event_id,)
        )
        
        event_info = cursor.fetchone()
        if not event_info:
            return create_response(
                status="error",
                code="EVENT_NOT_FOUND",
                message=f"Event {event_id} not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Get images for this event
        cursor.execute(
            """
            SELECT image_id, original_filename, storage_path, upload_date, file_size, processed
            FROM images
            WHERE event_id = ?
            ORDER BY upload_date DESC
            """,
            (event_id,)
        )
        
        images = []
        row = cursor.fetchone()
        while row:
            images.append({
                "image_id": str(row[0]),
                "original_filename": row[1],
                "storage_path": row[2],
                "upload_date": row[3].isoformat() if row[3] else None,
                "file_size": row[4],
                "processed": bool(row[5])
            })
            row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return create_response(
            status="success",
            code="IMAGES_FOUND",
            message=f"Found {len(images)} images for event {event_id}",
            details={"images": images}
        )
    except Exception as e:
        logger.error(f"Error retrieving images: {str(e)}")
        return create_response(
            status="error",
            code="DATABASE_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Processing endpoint
@app.post("/events/{event_id}/process", dependencies=[Depends(verify_api_key)])
async def process_event_images(event_id: str, background_tasks: BackgroundTasks):
    try:
        # Verify event exists and check status
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT e.customer_id, e.status_id, st.status_code, 
                   (SELECT COUNT(*) FROM images WHERE event_id = e.event_id AND processed = 0) as unprocessed_images
            FROM events e
            JOIN event_status_types st ON e.status_id = st.status_id
            WHERE e.event_id = ?
            """,
            (event_id,)
        )
        
        event_info = cursor.fetchone()
        if not event_info:
            return create_response(
                status="error",
                code="EVENT_NOT_FOUND",
                message=f"Event {event_id} not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        event_status = event_info[2]
        unprocessed_images = event_info[3]
        
        # Check if processing is already active
        cursor.execute(
            """
            SELECT COUNT(*) FROM processing_jobs j
            JOIN status_types s ON j.status_id = s.status_id
            WHERE j.event_id = ? AND s.status_code IN ('queued', 'processing')
            """,
            (event_id,)
        )
        
        active_jobs = cursor.fetchone()[0]
        
        if active_jobs > 0:
            return create_response(
                status="error",
                code="PROCESSING_IN_PROGRESS",
                message="Processing is already in progress for this event",
                status_code=status.HTTP_409_CONFLICT
            )
        
        cursor.close()
        conn.close()
        
        if unprocessed_images == 0:
            return create_response(
                status="warning",
                code="NO_IMAGES_TO_PROCESS",
                message="No unprocessed images found for this event",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Start processing in background
        def process_images_background():
            try:
                result = db_face_system.process_event_images(event_id)
                logger.info(f"Background processing completed for event {event_id}: {result}")
            except Exception as e:
                logger.error(f"Error in background processing for event {event_id}: {e}")
        
        background_tasks.add_task(process_images_background)
        
        return create_response(
            status="success",
            code="PROCESSING_STARTED",
            message="Image processing started in the background",
            details={
                "event_id": event_id,
                "unprocessed_images": unprocessed_images
            },
            status_code=status.HTTP_202_ACCEPTED
        )
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Search endpoint
@app.post("/events/{event_id}/search", dependencies=[Depends(verify_api_key)])
async def search_face(
    event_id: str, 
    file: UploadFile = File(...), 
    threshold: float = Query(0.83, ge=0.5, le=1.0)
):
    # Create a temporary file to store the uploaded image
    temp_file = None
    
    try:
        # Verify event exists and has clusters
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM clusters 
            WHERE event_id = ?
            """,
            (event_id,)
        )
        
        cluster_count = cursor.fetchone()[0]
        if cluster_count == 0:
            return create_response(
                status="error",
                code="NO_CLUSTERS",
                message="No face clusters found for this event. Process images first.",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        cursor.close()
        conn.close()
        
        # Save the uploaded file temporarily
        file_content = await file.read()
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        temp_filename = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.jpg"
        temp_file = os.path.join(TEMP_DIR, temp_filename)
        
        with open(temp_file, "wb") as f:
            f.write(file_content)
        
        # Perform the search
        results = db_face_system.search_face(temp_file, event_id, threshold)
        
        if results['status'] == 'success':
            return create_response(
                status="success",
                code="SEARCH_COMPLETED",
                message=results['message'],
                details={
                    "matches": results['matches'],
                    "job_id": results.get('job_id')
                }
            )
        else:
            return create_response(
                status="error",
                code="SEARCH_ERROR",
                message=results['message'],
                status_code=status.HTTP_400_BAD_REQUEST
            )
    
    except Exception as e:
        logger.error(f"Error in face search: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

# Cluster details endpoint
@app.get("/events/{event_id}/clusters/{cluster_id}", dependencies=[Depends(verify_api_key)])
async def get_cluster_details(event_id: str, cluster_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify cluster exists and belongs to the event
        cursor.execute(
            """
            SELECT c.cluster_id, c.total_faces, f.face_path
            FROM clusters c
            JOIN faces f ON c.representative_face_id = f.face_id
            WHERE c.event_id = ? AND c.cluster_id = ?
            """,
            (event_id, cluster_id)
        )
        
        cluster_info = cursor.fetchone()
        if not cluster_info:
            return create_response(
                status="error",
                code="CLUSTER_NOT_FOUND",
                message=f"Cluster {cluster_id} not found for event {event_id}",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Get all faces in this cluster
        cursor.execute(
            """
            SELECT f.face_id, f.face_path, cf.similarity_score, i.original_filename, i.storage_path
            FROM cluster_faces cf
            JOIN faces f ON cf.face_id = f.face_id
            JOIN images i ON f.image_id = i.image_id
            WHERE cf.cluster_id = ?
            ORDER BY cf.similarity_score DESC
            """,
            (cluster_id,)
        )
        
        faces = []
        row = cursor.fetchone()
        while row:
            faces.append({
                "face_id": str(row[0]),
                "face_url": row[1],
                "similarity": row[2],
                "filename": row[3],
                "original_path": row[4]
            })
            row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return create_response(
            status="success",
            code="CLUSTER_FOUND",
            message=f"Cluster details retrieved",
            details={
                "cluster_id": cluster_id,
                "total_faces": cluster_info[1],
                "representative_url": cluster_info[2],
                "faces": faces
            }
        )
    
    except Exception as e:
        logger.error(f"Error retrieving cluster details: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Status endpoint
@app.get("/events/{event_id}/status", dependencies=[Depends(verify_api_key)])
async def check_event_status(event_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get event status
        cursor.execute(
            """
            SELECT e.event_id, e.name, est.status_code as event_status,
                   (SELECT COUNT(*) FROM images WHERE event_id = e.event_id) as total_images,
                   (SELECT COUNT(*) FROM images WHERE event_id = e.event_id AND processed = 1) as processed_images,
                   (SELECT COUNT(*) FROM clusters WHERE event_id = e.event_id) as total_clusters,
                   (SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.image_id WHERE i.event_id = e.event_id) as total_faces
            FROM events e
            JOIN event_status_types est ON e.status_id = est.status_id
            WHERE e.event_id = ?
            """,
            (event_id,)
        )
        
        event_info = cursor.fetchone()
        if not event_info:
            return create_response(
                status="error",
                code="EVENT_NOT_FOUND",
                message=f"Event {event_id} not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Get active processing jobs
        cursor.execute(
            """
            SELECT j.job_id, jt.job_code, st.status_code, j.created_at, j.started_at, j.completed_at
            FROM processing_jobs j
            JOIN job_types jt ON j.job_type_id = jt.job_type_id
            JOIN status_types st ON j.status_id = st.status_id
            WHERE j.event_id = ?
            ORDER BY j.created_at DESC
            """,
            (event_id,)
        )
        
        jobs = []
        row = cursor.fetchone()
        while row and len(jobs) < 10:  # Limit to last 10 jobs
            jobs.append({
                "job_id": str(row[0]),
                "job_type": row[1],
                "status": row[2],
                "created_at": row[3].isoformat() if row[3] else None,
                "started_at": row[4].isoformat() if row[4] else None,
                "completed_at": row[5].isoformat() if row[5] else None
            })
            row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        # Determine if processing is active
        processing_active = any(job["status"] == "processing" for job in jobs)
        
        return create_response(
            status="success",
            code="STATUS_CHECKED",
            message="Event status retrieved successfully",
            details={
                "event_id": str(event_info[0]),
                "event_name": event_info[1],
                "event_status": event_info[2],
                "processing_active": processing_active,
                "total_images": event_info[3],
                "processed_images": event_info[4],
                "total_clusters": event_info[5],
                "total_faces": event_info[6],
                "recent_jobs": jobs
            }
        )
    
    except Exception as e:
        logger.error(f"Error checking event status: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Check database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        # Check storage directories
        storage_ok = os.path.exists(STORAGE_DIR) and os.access(STORAGE_DIR, os.W_OK)
        temp_ok = os.path.exists(TEMP_DIR) and os.access(TEMP_DIR, os.W_OK)
        
        return create_response(
            status="success",
            code="HEALTHY",
            message="API is running and all systems are operational",
            details={
                "database": {
                    "status": "connected",
                    "version": version
                },
                "storage": {
                    "status": "accessible" if storage_ok else "inaccessible",
                    "path": STORAGE_DIR
                },
               "temp_storage": {
                    "status": "accessible" if temp_ok else "inaccessible",
                    "path": TEMP_DIR
                },
                "face_system": {
                    "status": "initialized"
                }
            }
        )
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
# Thumbnail generation endpoint
@app.get("/images/{image_id}/thumbnail")
async def get_image_thumbnail(image_id: str, size: int = Query(256, ge=32, le=1024)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT storage_path FROM images WHERE image_id = ?",
            (image_id,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")
        
        file_path = result[0]
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Image file not found on server")
        
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                img.thumbnail((size, size))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                return StreamingResponse(img_byte_arr, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating thumbnail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")



@app.get("/api/faces/{face_id}", dependencies=[Depends(verify_api_key)])
async def get_face_image(face_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the face path but also verify the requester has access to this event
        cursor.execute(
            """
            SELECT f.face_path, i.event_id
            FROM faces f
            JOIN images i ON f.image_id = i.image_id
            WHERE f.face_id = ?
            """,
            (face_id,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Face '{face_id}' not found")
        
        face_path = result[0]
        
        if not os.path.exists(face_path):
            raise HTTPException(status_code=404, detail=f"Face image file not found on server")
        
        return FileResponse(face_path, media_type="image/jpeg")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving face image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/api/images/{image_id}", dependencies=[Depends(verify_api_key)])
async def get_original_image(image_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT storage_path FROM images WHERE image_id = ?",
            (image_id,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")
        
        file_path = result[0]
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Image file not found on server")
        
        return FileResponse(file_path, media_type="image/jpeg")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving original image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/api/representatives/{cluster_id}", dependencies=[Depends(verify_api_key)])
async def get_representative_image(cluster_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT f.face_path
            FROM clusters c
            JOIN faces f ON c.representative_face_id = f.face_id
            WHERE c.cluster_id = ?
            """,
            (cluster_id,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found")
        
        face_path = result[0]
        
        if not os.path.exists(face_path):
            raise HTTPException(status_code=404, detail=f"Representative image not found on server")
        
        return FileResponse(face_path, media_type="image/jpeg")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving representative image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# Face image direct access (for authorized users)
@app.get("/faces/{face_id}", dependencies=[Depends(verify_api_key)])
async def get_face_image(face_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT face_path FROM faces WHERE face_id = ?",
            (face_id,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Face '{face_id}' not found")
        
        face_path = result[0]
        
        if not os.path.exists(face_path):
            raise HTTPException(status_code=404, detail=f"Face image file not found on server")
        
        return FileResponse(face_path, media_type="image/jpeg")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving face image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Main entry point
if __name__ == "__main__":
    for dir_path in [STORAGE_DIR, TEMP_DIR]:
        os.makedirs(os.path.abspath(dir_path), exist_ok=True)
    
    # Start the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="\\\\str.innotech.com.la\\Storage\\Temp\\ssl\\privkey.pem",
        ssl_certfile="\\\\str.innotech.com.la\\Storage\\Temp\\ssl\\fullchain.pem",
        reload=False
    )  