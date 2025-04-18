from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
from datetime import datetime
import logging
import json
import time
from face_system_v1 import FaceSystem, FaceSystemConfig, process_temp_album, clear_temp_folder
from PIL import Image
from starlette.responses import StreamingResponse
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FaceSystem
config = FaceSystemConfig()
face_system = FaceSystem(config=config)

# API Configuration
BASE_URL = os.getenv("BASE_URL", "https://top.innotech.com.la:8000")
TEMP_ALBUM_DIR = os.getenv("TEMP_ALBUM_DIR", "temp_album")
ALBUM_DIR = os.getenv("ALBUM_DIR", "album")  # Real directory, kept private
CLUSTER_DIR = os.getenv("CLUSTER_DIR", "clustered_faces")
RESULTS_DIR = os.getenv("RESULTS_DIR", "face_search_results")

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="API for face detection, clustering, and search using temporary album workflow",
    version="1.0.0",
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

# Create directories
for dir_path in [TEMP_ALBUM_DIR, ALBUM_DIR, CLUSTER_DIR, RESULTS_DIR]:
    dir_path = os.path.abspath(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Created directory: {dir_path}")

# Mount static files (only for clusters and results, not album)
app.mount("/images", StaticFiles(directory=CLUSTER_DIR), name="images")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
# Removed: app.mount("/album", StaticFiles(directory=ALBUM_DIR), name="album")

# Initialize FaceSystem
# face_system = FaceSystem(batch_size=16)

# Lock file helpers
def is_processing_active():
    lock_file = os.path.join(CLUSTER_DIR, "processing.lock")
    if os.path.exists(lock_file):
        lock_time = os.path.getmtime(lock_file)
        if time.time() - lock_time > 3600:
            os.remove(lock_file)
            return False
        return True
    return False

def create_processing_lock():
    lock_file = os.path.join(CLUSTER_DIR, "processing.lock")
    with open(lock_file, 'w') as f:
        f.write(datetime.now().isoformat())
    return lock_file

def release_processing_lock():
    lock_file = os.path.join(CLUSTER_DIR, "processing.lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)

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

@app.get("/")
async def root():
    return create_response(
        status="success",
        code="API_INFO",
        message="Face Recognition API is running",
        details={
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "status": "/status",
                "process": "/process",
                "search": "/search",
                "clear-temp-album": "/clear-temp-album",
                "upload-to-temp": "/upload-to-temp",
                "documentation": "/docs"
            }
        }
    )

@app.post("/process/", status_code=status.HTTP_202_ACCEPTED)
async def process_images(background_tasks: BackgroundTasks, force: bool = False):
    try:
        if is_processing_active() and not force:
            return create_response(
                status="error",
                code="PROCESSING_IN_PROGRESS",
                message="Image processing is already in progress. Use force=true to override.",
                status_code=status.HTTP_409_CONFLICT
            )
        
        if not os.path.exists(TEMP_ALBUM_DIR):
            return create_response(
                status="error",
                code="DIRECTORY_NOT_FOUND",
                message=f"Temporary album directory not found: {TEMP_ALBUM_DIR}",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        image_files = [f for f in os.listdir(TEMP_ALBUM_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            return create_response(
                status="error",
                code="NO_IMAGES_FOUND",
                message=f"No images found in {TEMP_ALBUM_DIR}",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        lock_file = create_processing_lock()
        
        def process_in_background():
            try:
                logger.info(f"Starting background processing of {len(image_files)} images...")
                result = process_temp_album(
                    temp_album_dir=TEMP_ALBUM_DIR,
                    cluster_dir=CLUSTER_DIR,
                    album_dir=ALBUM_DIR,
                    batch_size=16
                )
                logger.info(f"Background processing completed: {result}")
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                logger.exception("Detailed error:")
            finally:
                release_processing_lock()
        
        background_tasks.add_task(process_in_background)
        
        return create_response(
            status="success",
            code="PROCESSING_STARTED",
            message="Image processing started in the background.",
            details={"images_to_process": len(image_files)},
            status_code=status.HTTP_202_ACCEPTED
        )
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        release_processing_lock()
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.get("/status/")
async def check_processing_status():
    try:
        processing_active = is_processing_active()
        clusters_available = os.path.exists(os.path.join(CLUSTER_DIR, "representatives.json"))
        stats = {}
        manifest_path = os.path.join(CLUSTER_DIR, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                stats = json.load(f)
        
        temp_image_count = len([f for f in os.listdir(TEMP_ALBUM_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        album_image_count = len([f for f in os.listdir(ALBUM_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        return create_response(
            status="success",
            code="STATUS_CHECKED",
            message="Processing status retrieved.",
            details={
                "processing_active": processing_active,
                "clusters_available": clusters_available,
                "temp_album_image_count": temp_image_count,
                "album_image_count": album_image_count,
                "stats": stats
            }
        )
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.post("/search/")
async def search_face(file: UploadFile = File(...), threshold: float = Query(0.83, ge=0.5, le=1.0)):
    temp_file = None
    try:
        if not os.path.exists(os.path.join(CLUSTER_DIR, "representatives.json")):
            return create_response(
                status="error",
                code="NO_CLUSTERS_FOUND",
                message="No clustered faces available. Please process images first.",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        temp_dir = os.path.join(os.getcwd(), "temp_search")
        os.makedirs(temp_dir, exist_ok=True)
        file_content = await file.read()
        temp_filename = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.jpg"
        temp_file = os.path.join(temp_dir, temp_filename)
        
        with open(temp_file, "wb") as f:
            f.write(file_content)
        
        query_img = cv2.imread(temp_file)
        if query_img is None:
            return create_response(
                status="error",
                code="INVALID_IMAGE",
                message="Could not read the uploaded image",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        if not face_system.representative_faces:
            if not face_system.load_representative_faces(CLUSTER_DIR):
                return create_response(
                    status="error",
                    code="LOADING_ERROR",
                    message="Failed to load representative faces",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        results = face_system.find_matching_faces(
            query_image=query_img_rgb,
            clustered_dir=CLUSTER_DIR,
            album_dir=ALBUM_DIR,
            similarity_threshold=threshold
        )
        
        if results['status'] == 'success':
            if not results['matches']:
                return create_response(
                    status="success",
                    code="NO_MATCHES_FOUND",
                    message="No matching faces found above the threshold.",
                    details={"matches": []}
                )
            
            processed_matches = []
            for match in results['matches']:
                person_id = int(match['person_id'])
                source_file_urls = [
                    {"filename": str(source_file), "face_url": f"{BASE_URL}/photo/{source_file}"}
                    for source_file in match['source_files']
                ]
                processed_matches.append({
                    "person_id": person_id,
                    "similarity": float(match['similarity']),
                    "source_files": source_file_urls,
                    "representative_url": f"{BASE_URL}/images/person_{person_id}/representative.jpg"
                })
            
            return create_response(
                status="success",
                code="MATCHES_FOUND",
                message=f"Found {len(processed_matches)} matching persons",
                details={"matches": processed_matches}
            )
        else:
            return create_response(
                status="error",
                code="SEARCH_ERROR",
                message=results['message'],
                details={"matches": []},
                status_code=status.HTTP_400_BAD_REQUEST
            )
    except Exception as e:
        logger.error(f"Error in face search: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            details={"matches": []},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        return create_response(
            status="success",
            code="HEALTHY",
            message="API is running and healthy",
            details={
                "temp_album_dir": TEMP_ALBUM_DIR,
                "album_dir": ALBUM_DIR,
                "cluster_dir": CLUSTER_DIR,
                "processing_active": is_processing_active(),
                "temp_album_image_count": len([f for f in os.listdir(TEMP_ALBUM_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]),
                "album_image_count": len([f for f in os.listdir(ALBUM_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]),
                "clusters_available": os.path.exists(os.path.join(CLUSTER_DIR, "representatives.json"))
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

@app.get("/clear-temp-album")
async def clear_temp_album(force: bool = False):
    try:
        if is_processing_active() and not force:
            return create_response(
                status="error",
                code="PROCESSING_IN_PROGRESS",
                message="Processing is active. Use force=true to clear anyway.",
                status_code=status.HTTP_409_CONFLICT
            )
        
        image_count = len([f for f in os.listdir(TEMP_ALBUM_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        cleared_count = clear_temp_folder(TEMP_ALBUM_DIR)
        
        return create_response(
            status="success",
            code="TEMP_CLEARED",
            message=f"Cleared {cleared_count} files from temporary album directory.",
            details={"cleared_files": cleared_count}
        )
    except Exception as e:
        logger.error(f"Error clearing temp album: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.get("/person/{person_id}")
async def get_person_details(person_id: int):
    """Get details for a specific person by ID."""
    try:
        # Check if the person exists
        person_dir = os.path.join(CLUSTER_DIR, f"person_{person_id}")
        if not os.path.exists(person_dir):
            return create_response(
                status="error",
                code="PERSON_NOT_FOUND",
                message=f"Person ID {person_id} not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Get representative face URL
        representative_url = f"{BASE_URL}/images/person_{person_id}/representative.jpg"
        
        # Read sources file to get source images
        sources_path = os.path.join(person_dir, "sources.txt")
        source_files = []
        
        if os.path.exists(sources_path):
            with open(sources_path, 'r') as f:
                sources = f.read().splitlines()
                
            # Create source file objects
            for source in sources:
                source = source.strip()
                if source:
                    source_files.append({
                        "filename": source,
                        "face_url": f"{BASE_URL}/photo/{source}",
                        "face_url_thumbnail": f"{BASE_URL}/thumbnail/{source}"
                    })
        
        # Get face variants (if they exist)
        face_variants = []
        for i in range(50):  # Check up to 50 face variants
            face_path = os.path.join(person_dir, f"face_{i}.jpg")
            if os.path.exists(face_path):
                face_variants.append({
                    "index": i,
                    "url": f"{BASE_URL}/images/person_{person_id}/face_{i}.jpg"
                })
            elif i > 0 and not os.path.exists(os.path.join(person_dir, f"face_{i+1}.jpg")):
                # If this face doesn't exist and neither does the next one, we've likely reached the end
                break
        
        # Get person data from representatives.json if it exists
        rep_file = os.path.join(CLUSTER_DIR, "representatives.json")
        extra_info = {}
        
        if os.path.exists(rep_file):
            try:
                with open(rep_file, 'r') as f:
                    rep_info = json.load(f)
                    
                # If this person is in the representatives file, get additional info
                if str(person_id) in rep_info:
                    person_info = rep_info[str(person_id)]
                    extra_info = {
                        "original_image": person_info.get("original_image", ""),
                        "total_source_files": len(person_info.get("source_files", [])),
                        "creation_date": os.path.getctime(person_dir)
                    }
            except Exception as e:
                logger.error(f"Error reading representatives file: {e}")
        
        # Return complete person data
        return create_response(
            status="success",
            code="PERSON_FOUND",
            message=f"Person {person_id} details retrieved successfully",
            details={
                "person_id": person_id,
                "representative_url": representative_url,
                "source_files": source_files,
                "face_variants": face_variants,
                "extra_info": extra_info
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving person details: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.get("/thumbnail/{filename}")
async def get_thumbnail(filename: str, size: int = 256):
    file_path = os.path.join(ALBUM_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    
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

@app.get("/photo/{source}")
async def get_photo(source: str):
    """Serve an image file from ALBUM_DIR securely."""
    # Sanitize the source to prevent directory traversal
    source = os.path.basename(source)
    file_path = os.path.join(ALBUM_DIR, source)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Photo '{source}' not found")
    
    try:
        return FileResponse(file_path, media_type="image/jpeg")  # Adjust media_type if needed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving photo: {str(e)}")

@app.post("/upload-to-temp/")
async def upload_to_temp(file: UploadFile = File(...)):
    try:
        os.makedirs(TEMP_ALBUM_DIR, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        file_path = os.path.join(TEMP_ALBUM_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return create_response(
            status="success",
            code="UPLOAD_SUCCESS",
            message=f"File {filename} uploaded to temp_album",
            details={"file_path": file_path}
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return create_response(
            status="error",
            code="SYSTEM_ERROR",
            message=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

if __name__ == "__main2__":
    for dir_path in [TEMP_ALBUM_DIR, ALBUM_DIR, CLUSTER_DIR, RESULTS_DIR]:
        os.makedirs(os.path.abspath(dir_path), exist_ok=True)
    
    uvicorn.run(
        "main2:app",
        host="0.0.0.0",
        ssl_keyfile="\\\\str.innotech.com.la\\Storage\\Temp\\ssl\\privkey.pem",
        ssl_certfile="\\\\str.innotech.com.la\\Storage\\Temp\\ssl\\fullchain.pem",
        port=8000,
        reload=False
    )