#!/usr/bin/env python3
"""
Simple client for the Face Recognition API.
Demonstrates the temp_album workflow.
"""

import requests
import os
import time
import json
import argparse
import shutil
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

class FaceRecognitionClient:
    """Client for interacting with the Face Recognition API."""
    
    def __init__(self, base_url: str = BASE_URL):
        """Initialize the client with the API server URL."""
        self.base_url = base_url.rstrip('/')
    
    def check_health(self) -> Dict[str, Any]:
        """Check API server health and get directory information."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def check_status(self) -> Dict[str, Any]:
        """Get current processing status and statistics."""
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def start_processing(self) -> Dict[str, Any]:
        """Start processing images from temp_album directory."""
        response = requests.post(f"{self.base_url}/process")
        response.raise_for_status()
        return response.json()
    
    def clear_temp_album(self, force: bool = False) -> Dict[str, Any]:
        """Clear all files from the temporary album directory."""
        response = requests.get(f"{self.base_url}/clear-temp-album", params={"force": force})
        response.raise_for_status()
        return response.json()
    
    def search_face(self, image_path: str, threshold: float = 0.83) -> Dict[str, Any]:
        """Search for a face in the processed clusters."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(
                f"{self.base_url}/search",
                params={"threshold": threshold},
                files=files
            )
        
        response.raise_for_status()
        return response.json()
    
    def wait_for_processing(self, timeout: int = 600, interval: int = 5):
        """Wait for processing to complete."""
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Processing timed out")
            
            status = self.check_status()
            if not status["processing_active"]:
                return status
            
            logger.info("Processing still active. Waiting...")
            time.sleep(interval)
    
    def copy_images_to_temp_album(self, source_dir: str, temp_album_dir: str = None):
        """Copy images from a source directory to the temp_album directory."""
        # Get temp_album directory from API if not specified
        if not temp_album_dir:
            health = self.check_health()
            temp_album_dir = health.get("temp_album_dir", "temp_album")
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        # Ensure temp_album directory exists
        os.makedirs(temp_album_dir, exist_ok=True)
        
        # Get list of image files
        image_files = [
            f for f in os.listdir(source_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not image_files:
            logger.warning(f"No images found in {source_dir}")
            return 0
        
        # Copy images to temp_album
        copied_count = 0
        for filename in image_files:
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(temp_album_dir, filename)
            
            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                logger.error(f"Error copying {filename}: {e}")
        
        logger.info(f"Copied {copied_count} images to {temp_album_dir}")
        return copied_count

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Client")
    parser.add_argument("--url", default=BASE_URL, help="API server URL")
    parser.add_argument("--action", choices=["health", "status", "process", "copy", "search", "clear"], 
                      required=True, help="Action to perform")
    parser.add_argument("--source", help="Source directory for image copy")
    parser.add_argument("--image", help="Image file for face search")
    parser.add_argument("--threshold", type=float, default=0.83, help="Search similarity threshold")
    parser.add_argument("--wait", action="store_true", help="Wait for processing to complete")
    parser.add_argument("--force", action="store_true", help="Force operation")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    client = FaceRecognitionClient(args.url)
    
    try:
        if args.action == "health":
            result = client.check_health()
            print(json.dumps(result, indent=2))
        
        elif args.action == "status":
            result = client.check_status()
            print(json.dumps(result, indent=2))
        
        elif args.action == "process":
            result = client.start_processing()
            print(json.dumps(result, indent=2))
            
            if args.wait and result.get("status") == "success":
                print("\nWaiting for processing to complete...")
                try:
                    final_status = client.wait_for_processing()
                    print("\nProcessing completed:")
                    print(json.dumps(final_status, indent=2))
                except TimeoutError:
                    print("\nProcessing timed out")
        
        elif args.action == "copy":
            if not args.source:
                print("Error: --source is required for copy action")
                return
            
            count = client.copy_images_to_temp_album(args.source)
            print(f"Copied {count} images to temporary album")
        
        elif args.action == "search":
            if not args.image:
                print("Error: --image is required for search action")
                return
            
            result = client.search_face(args.image, args.threshold)
            
            # Save results to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to {args.output}")
            
            print(json.dumps(result, indent=2))
        
        elif args.action == "clear":
            result = client.clear_temp_album(args.force)
            print(json.dumps(result, indent=2))
    
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with API: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()