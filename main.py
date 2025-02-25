from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import cv2
import numpy as np
from datetime import datetime
from face_system_cuda import FaceSystem
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
ALBUM_DIR = os.getenv("ALBUM_DIR", "album")  # Directory with existing images
CLUSTER_DIR = "clustered_faces"  # Output directory for clusters
UPLOAD_DIR = "uploads"  # For query images in search

# Initialize FastAPI app
app = FastAPI(title="Face Clustering and Search API")

# Initialize FaceSystem
face_system = FaceSystem(batch_size=16)

# Create necessary directories with absolute paths
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)
CLUSTER_DIR = os.path.abspath(CLUSTER_DIR)
ALBUM_DIR = os.path.abspath(ALBUM_DIR)

for dir_path in [CLUSTER_DIR, UPLOAD_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Created directory: {dir_path}")

# Mount static files for serving images
app.mount("/images", StaticFiles(directory=CLUSTER_DIR), name="images")
app.mount("/album", StaticFiles(directory=ALBUM_DIR), name="album")

@app.post("/cluster/")
async def cluster_images():
    """Perform face clustering on images in the album directory."""
    try:
        # Check if album directory exists and contains images
        if not os.path.exists(ALBUM_DIR):
            return JSONResponse({
                "status": "error",
                "code": "DIRECTORY_NOT_FOUND",
                "message": f"Album directory not found: {ALBUM_DIR}"
            }, status_code=400)

        # Get all image paths
        image_paths = [
            os.path.join(ALBUM_DIR, f) for f in os.listdir(ALBUM_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not image_paths:
            return JSONResponse({
                "status": "error",
                "code": "NO_IMAGES_FOUND",
                "message": f"No images found in {ALBUM_DIR}"
            }, status_code=400)
            
        logger.info(f"Found {len(image_paths)} images to process")
            
        # Process images in batches
        all_faces = []
        for i in range(0, len(image_paths), face_system.batch_size):
            batch = image_paths[i:i + face_system.batch_size]
            faces = face_system.process_image_batch(batch)
            all_faces.extend(faces)
            logger.info(f"Processed {min(i + face_system.batch_size, len(image_paths))}/{len(image_paths)} images...")
        
        # Perform clustering
        logger.info("Clustering faces...")
        labels = face_system.cluster_faces(all_faces)
        
        # Save results
        logger.info("Saving clustered results...")
        face_system.save_clusters(all_faces, labels, CLUSTER_DIR)
        
        unique_people = len(set(labels) - {-1})
        total_faces = len(all_faces)
        
        return JSONResponse({
            "status": "success",
            "code": "CLUSTERING_COMPLETE",
            "message": f"Successfully clustered faces",
            "details": {
                "unique_people": unique_people,
                "total_faces": total_faces,
                "clusters_directory": CLUSTER_DIR
            }
        })
        
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}")
        return JSONResponse({
            "status": "error",
            "code": "CLUSTERING_ERROR",
            "message": str(e)
        }, status_code=500)

@app.post("/search/")
async def search_face(file: UploadFile = File(...), threshold: float = 0.83):
    """Search for matching faces in clustered results."""
    try:
        # Check if clustering has been done
        if not os.path.exists(CLUSTER_DIR) or not os.listdir(CLUSTER_DIR):
            return JSONResponse({
                "status": "error",
                "code": "NO_CLUSTERS_FOUND",
                "message": "No clustered faces available. Please run clustering first."
            }, status_code=400)

        # Create a unique filename for the query image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"query_{timestamp}_{os.urandom(4).hex()}.jpg"
        query_path = os.path.join(UPLOAD_DIR, filename)
        
        # Ensure the uploads directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Save the uploaded file
        logger.info(f"Saving query image to: {query_path}")
        with open(query_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"Failed to save file to {query_path}")
            
        # Load query image
        query_img = cv2.imread(query_path)
        if query_img is None:
            return JSONResponse({
                "status": "error",
                "code": "INVALID_IMAGE",
                "message": "Could not read the uploaded image"
            }, status_code=400)
            
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # Load representative faces if not already loaded
        if not face_system.representative_faces:
            if not face_system.load_representative_faces(CLUSTER_DIR):
                return JSONResponse({
                    "status": "error",
                    "code": "LOADING_ERROR",
                    "message": "Failed to load representative faces"
                }, status_code=500)
        
        # Perform search
        results = face_system.find_matching_faces(query_img, CLUSTER_DIR, threshold)
        
        if results['status'] == 'success':
            if not results['matches']:
                return JSONResponse({
                    "status": "success",
                    "code": "NO_MATCHES_FOUND",
                    "message": "No matching faces were found above the similarity threshold.",
                    "matches": []
                })

            # Process matches
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
            }, status_code=400)
            
    except Exception as e:
        logger.error(f"Error in face search: {str(e)}")
        return JSONResponse({
            "status": "error",
            "code": "SYSTEM_ERROR",
            "message": str(e),
            "matches": []
        }, status_code=500)
    finally:
        # Cleanup: remove the temporary query image
        try:
            if 'query_path' in locals() and os.path.exists(query_path):
                os.remove(query_path)
                logger.info(f"Cleaned up temporary file: {query_path}")
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "album_dir": ALBUM_DIR,
        "cluster_dir": CLUSTER_DIR
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)