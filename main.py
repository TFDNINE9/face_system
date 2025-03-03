from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import numpy as np
from datetime import datetime
import logging
import json
import tempfile
import time
from face_system import FaceSystem, process_temp_album, clear_temp_folder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = os.getenv("BASE_URL", "https://top.innotech.com.la:8000")
TEMP_ALBUM_DIR = os.getenv("TEMP_ALBUM_DIR", "temp_album")  # Temporary folder for uploads
ALBUM_DIR = os.getenv("ALBUM_DIR", "album")                 # Permanent storage for original images
CLUSTER_DIR = os.getenv("CLUSTER_DIR", "clustered_faces")   # Storage for face clusters
RESULTS_DIR = os.getenv("RESULTS_DIR", "face_search_results")  # Directory for search results

# Initialize FastAPI app with metadata for docs
app = FastAPI(
    title="Face Recognition API",
    description="API for face detection, clustering, and search using temporary album workflow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create necessary directories
for dir_path in [TEMP_ALBUM_DIR, ALBUM_DIR, CLUSTER_DIR, RESULTS_DIR]:
    dir_path = os.path.abspath(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Created directory: {dir_path}")

# Mount static files for serving images
app.mount("/images", StaticFiles(directory=CLUSTER_DIR), name="images")
app.mount("/album", StaticFiles(directory=ALBUM_DIR), name="album")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Initialize FaceSystem
face_system = FaceSystem(batch_size=16)

# Check for active processing
def is_processing_active():
    lock_file = os.path.join(CLUSTER_DIR, "processing.lock")
    if os.path.exists(lock_file):
        # Check if the lock is stale (older than 1 hour)
        lock_time = os.path.getmtime(lock_file)
        current_time = time.time()
        if current_time - lock_time > 3600:  # 1 hour in seconds
            os.remove(lock_file)
            return False
        return True
    return False

# Create processing lock
def create_processing_lock():
    lock_file = os.path.join(CLUSTER_DIR, "processing.lock")
    with open(lock_file, 'w') as f:
        f.write(datetime.now().isoformat())
    return lock_file

# Release processing lock
def release_processing_lock():
    lock_file = os.path.join(CLUSTER_DIR, "processing.lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)

@app.get("/")
async def root():
    """Root endpoint that provides API information."""
    return {
        "name": "Face Recognition API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "process": "/process",
            "search": "/search",
            "clear-temp-album": "/clear-temp-album",
            "documentation": "/docs"
        }
    }

@app.post("/process/", status_code=status.HTTP_202_ACCEPTED)
async def process_images(background_tasks: BackgroundTasks, force: bool = False):
    """
    Process images from temp_album, add to clusters, and copy to permanent album.
    
    - **force**: Override processing lock if one exists
    
    Returns:
        JSON response with processing status
    """
    try:
        # Check if processing is already in progress
        if is_processing_active() and not force:
            return JSONResponse({
                "status": "error",
                "code": "PROCESSING_IN_PROGRESS",
                "message": "Image processing is already in progress. Please try again later or use force=true."
            }, status_code=status.HTTP_409_CONFLICT)
        
        # Check for images in temp_album directory
        if not os.path.exists(TEMP_ALBUM_DIR):
            return JSONResponse({
                "status": "error",
                "code": "DIRECTORY_NOT_FOUND",
                "message": f"Temporary album directory not found: {TEMP_ALBUM_DIR}"
            }, status_code=status.HTTP_400_BAD_REQUEST)
        
        image_files = [
            f for f in os.listdir(TEMP_ALBUM_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not image_files:
            return JSONResponse({
                "status": "error",
                "code": "NO_IMAGES_FOUND",
                "message": f"No images found in {TEMP_ALBUM_DIR}"
            }, status_code=status.HTTP_400_BAD_REQUEST)
        
        # Create processing lock
        lock_file = create_processing_lock()
        
        # Define background processing function
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
                logger.info("Processing lock released")
        
        # Start background processing
        background_tasks.add_task(process_in_background)
        
        # Return immediate response
        return JSONResponse({
            "status": "success",
            "code": "PROCESSING_STARTED",
            "message": "Image processing started in the background. Check status endpoint for updates.",
            "details": {
                "images_to_process": len(image_files)
            }
        })
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        logger.exception("Detailed error:")
        # Make sure to release the lock in case of error
        release_processing_lock()
        return JSONResponse({
            "status": "error",
            "code": "SYSTEM_ERROR",
            "message": str(e)
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
@app.get("/status/")
async def check_processing_status():
    """
    Check the status of the processing job and cluster statistics.
    
    Returns:
        JSON with processing status and statistics
    """
    try:
        # Check if processing is active
        processing_active = is_processing_active()
        
        # Check if clusters are available
        clusters_available = os.path.exists(os.path.join(CLUSTER_DIR, "representatives.json"))
        
        # Get processing stats if available
        stats = {}
        manifest_path = os.path.join(CLUSTER_DIR, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    stats = json.load(f)
            except Exception as e:
                logger.error(f"Error reading manifest: {str(e)}")
        
        # Get image counts
        temp_image_count = len([f for f in os.listdir(TEMP_ALBUM_DIR) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        album_image_count = len([f for f in os.listdir(ALBUM_DIR) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        return {
            "processing_active": processing_active,
            "clusters_available": clusters_available,
            "temp_album_image_count": temp_image_count,
            "album_image_count": album_image_count,
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        logger.exception("Detailed error:")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/search/")
async def search_face(
    file: UploadFile = File(...), 
    threshold: float = Query(0.83, ge=0.5, le=1.0, description="Similarity threshold (0.5-1.0)")
):
    """
    Search for matching faces in the clustered results.
    
    - **file**: Image file containing a face to search for
    - **threshold**: Similarity threshold (0.5-1.0)
    
    Returns:
        JSON with matching faces found in the clusters
    """
    temp_file = None
    try:
        # Check if clusters are available
        if not os.path.exists(os.path.join(CLUSTER_DIR, "representatives.json")):
            return JSONResponse({
                "status": "error",
                "code": "NO_CLUSTERS_FOUND",
                "message": "No clustered faces available. Please process images first."
            }, status_code=status.HTTP_400_BAD_REQUEST)
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_search")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the uploaded file with a unique name in our own temp directory
        file_content = await file.read()
        temp_filename = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.jpg"
        temp_file_path = os.path.join(temp_dir, temp_filename)
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Load and convert the image
        query_img = cv2.imread(temp_file_path)
        if query_img is None:
            return JSONResponse({
                "status": "error",
                "code": "INVALID_IMAGE",
                "message": "Could not read the uploaded image"
            }, status_code=status.HTTP_400_BAD_REQUEST)
        
        # Release the file handle by explicitly closing the image
        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # Load representative faces if not already loaded
        if not face_system.representative_faces:
            if not face_system.load_representative_faces(CLUSTER_DIR):
                return JSONResponse({
                    "status": "error",
                    "code": "LOADING_ERROR",
                    "message": "Failed to load representative faces"
                }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Perform search
        results = face_system.find_matching_faces(
            query_image=query_img_rgb,
            clustered_dir=CLUSTER_DIR,
            album_dir=ALBUM_DIR,
            similarity_threshold=threshold
        )
        
        if results['status'] == 'success':
            if not results['matches']:
                return JSONResponse({
                    "status": "success",
                    "code": "NO_MATCHES_FOUND",
                    "message": "No matching faces were found above the similarity threshold.",
                    "matches": []
                })
            
            # Process matches to add proper URLs
            processed_matches = []
            for match in results['matches']:
                person_id = int(match['person_id'])
                
                source_file_urls = []
                for source_file in match['source_files']:
                    source_file = str(source_file)
                    image_url = f"{BASE_URL}/album/{source_file}"
                    source_file_urls.append({
                        'filename': source_file,
                        'face_url': image_url
                    })
                
                processed_match = {
                    'person_id': person_id,
                    'similarity': float(match['similarity']),
                    'source_files': source_file_urls,
                    'representative_url': f"{BASE_URL}/images/person_{person_id}/representative.jpg"
                }
                processed_matches.append(processed_match)
            
            return JSONResponse({
                "status": "success",
                "code": "MATCHES_FOUND",
                "message": f"Found {len(processed_matches)} matching persons",
                "matches": processed_matches
            })
        
        else:
            return JSONResponse({
                "status": "error",
                "code": "SEARCH_ERROR",
                "message": results['message'],
                "matches": []
            }, status_code=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        logger.error(f"Error in face search: {str(e)}")
        logger.exception("Detailed error:")
        return JSONResponse({
            "status": "error",
            "code": "SYSTEM_ERROR",
            "message": str(e),
            "matches": []
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Try to clean up the temporary file
        try:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            # Just log the error but don't fail the request
            logger.warning(f"Could not remove temporary file: {str(e)}")
            
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running and check directory status.
    
    Returns:
        JSON with health information
    """
    return {
        "status": "healthy",
        "temp_album_dir": TEMP_ALBUM_DIR,
        "album_dir": ALBUM_DIR,
        "cluster_dir": CLUSTER_DIR,
        "processing_active": is_processing_active(),
        "temp_album_image_count": len([f for f in os.listdir(TEMP_ALBUM_DIR) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]),
        "album_image_count": len([f for f in os.listdir(ALBUM_DIR) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]),
        "clusters_available": os.path.exists(os.path.join(CLUSTER_DIR, "representatives.json"))
    }

@app.get("/clear-temp-album")
async def clear_temp_album(force: bool = False):
    """
    Clear all files from the temporary album directory.
    
    - **force**: Force clearing even if processing is active
    
    Returns:
        JSON response with clear operation status
    """
    try:
        # Safety check - if processing is active, don't clear unless forced
        if is_processing_active() and not force:
            return JSONResponse({
                "status": "error",
                "code": "PROCESSING_IN_PROGRESS",
                "message": "Processing is active. Use force=true to clear anyway."
            }, status_code=status.HTTP_409_CONFLICT)
        
        # Count files before clearing
        image_count = len([f for f in os.listdir(TEMP_ALBUM_DIR) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Clear the directory
        clear_temp_folder(TEMP_ALBUM_DIR)
        
        return JSONResponse({
            "status": "success",
            "message": f"Cleared {image_count} files from temporary album directory."
        })
    except Exception as e:
        logger.error(f"Error clearing temp album: {str(e)}")
        logger.exception("Detailed error:")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Add a test upload endpoint for placing files directly in temp_album
@app.post("/upload-to-temp/")
async def upload_to_temp(file: UploadFile = File(...)):
    """
    Upload an image directly to the temp_album directory for testing.
    
    - **file**: Image file to upload to temp_album
    
    Returns:
        JSON response with upload status
    """
    try:
        # Ensure temp_album directory exists
        os.makedirs(TEMP_ALBUM_DIR, exist_ok=True)
        
        # Generate a unique filename
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        file_path = os.path.join(TEMP_ALBUM_DIR, filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse({
            "status": "success",
            "message": f"File {filename} uploaded to temp_album",
            "file_path": file_path
        })
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        logger.exception("Detailed error:")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

if __name__ == "__main__":
    # Create required directories at startup
    for dir_path in [TEMP_ALBUM_DIR, ALBUM_DIR, CLUSTER_DIR, RESULTS_DIR]:
        os.makedirs(os.path.abspath(dir_path), exist_ok=True)
    
    # Start the API server
    uvicorn.run("main:app", host="0.0.0.0",ssl_keyfile="\\\\str.innotech.com.la\\Storage\\Temp\ssl\\privkey.pem", ssl_certfile="\\\\str.innotech.com.la\\Storage\\Temp\\ssl\\fullchain.pem" , port=8000, reload=False)