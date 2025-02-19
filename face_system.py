import numpy as np
import os
from sklearn.cluster import DBSCAN
import shutil
import cv2
import logging
from typing import List, Dict
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import openvino as ov
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceSystem:
    def __init__(self, batch_size=16):
        try:
            # Force GPU usage
           
            os.environ['OPENVINO_DEVICE'] = 'GPU'
            os.environ['ONNXRUNTIME_PROVIDER_PRIORITY'] = 'OpenVINOExecutionProvider'

            # Initialize OpenVINO
            core = ov.Core()
            if 'GPU' not in core.available_devices:
                raise ValueError("GPU device not available")
            
            logger.info(f"GPU device: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")
            
            # Initialize with GPU
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=['OpenVINOExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Model preparation completed with GPU optimization")

            self.batch_size = batch_size
            self.image_size_limit = 1920
            self.thread_lock = threading.Lock()
            
            # Clustering parameters
            self.clustering_params = {
                'eps': 0.9,
                'min_samples': 2,
                'same_image_threshold': 0.70,
                'face_similarity_metric': 'cosine'
            }
            
            # Representative faces storage
            self.representative_faces = {}

        except Exception as e:
            logger.error(f"Error initializing GPU: {str(e)}")
            raise

    def process_image_batch(self, image_paths: List[str]) -> List[Dict]:
        """Process a batch of images for face detection and recognition."""
        all_faces = []
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                if max(h, w) > self.image_size_limit:
                    scale = self.image_size_limit / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_AREA)
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                with self.thread_lock:
                    faces = self.app.get(img_rgb)
                
                if not faces:
                    continue
                    
                for idx, face in enumerate(faces):
                    try:
                        # Get face crop using alignment
                        face_img = face_align.norm_crop(img_rgb, face.kps)
                        
                        # Calculate face size from bbox
                        x1, y1, x2, y2 = face.bbox
                        face_size = (x2 - x1) * (y2 - y1)
                        
                        face_info = {
                            'embedding': face.embedding,
                            'face_image': face_img,
                            'original_path': path,
                            'face_index': idx,
                            'det_score': float(face.det_score),
                            'bbox': face.bbox,
                            'kps': face.kps,  # Add keypoints for alignment calculation
                            'face_size': face_size  # Add face size
                        }
                        all_faces.append(face_info)
                    except Exception as e:
                        logger.error(f"Error processing face {idx} in {path}: {str(e)}")
                        continue
                    
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                continue
                
        return all_faces

    def cluster_faces(self, face_data: List[Dict]) -> List[int]:
        """Cluster faces and store representative faces."""
        if not face_data:
            return []
        
        embeddings = np.array([face['embedding'] for face in face_data])
        norms = np.linalg.norm(embeddings, axis=1)
        normalized_embeddings = embeddings / norms[:, np.newaxis]
        
        clustering = DBSCAN(
            eps=self.clustering_params['eps'],
            min_samples=self.clustering_params['min_samples'],
            metric='euclidean',
            n_jobs=-1
        ).fit(normalized_embeddings)
        
        # Store representative faces
        self.representative_faces = {}
        labels = clustering.labels_
        
        for label in set(labels):
            if label != -1:  # Skip noise
                # Get all faces in this cluster
                cluster_faces = [face_data[i] for i, l in enumerate(labels) if l == label]
                try:
                    # Choose representative face
                    representative = self.select_representative_face(cluster_faces)
                    if representative:
                        self.representative_faces[label] = {
                            'embedding': representative['embedding'],
                            'face_image': representative['face_image'],
                            'original_path': representative['original_path']
                        }
                except Exception as e:
                    logger.error(f"Error processing cluster {label}: {str(e)}")
                    continue
        
        return labels
    
    def load_representative_faces(self, clustered_dir: str):
            """Load representative faces directly with saved embeddings."""
            try:
                # Load representatives info
                rep_file = os.path.join(clustered_dir, "representatives.json")
                logger.info(f"Looking for representatives file at: {rep_file}")
                
                if not os.path.exists(rep_file):
                    logger.error("Representatives file not found")
                    return False

                with open(rep_file, 'r') as f:
                    rep_info = json.load(f)
                logger.info(f"Loaded representative info for {len(rep_info)} clusters")

                self.representative_faces = {}
                
                # Load each representative face
                for label, info in rep_info.items():
                    full_img_path = os.path.abspath(info['path'])
                    full_emb_path = os.path.abspath(info['embedding_path'])
                    
                    # logger.info(f"Processing representative for cluster {label}")
                    
                    if not os.path.exists(full_img_path):
                        logger.error(f"Representative image not found: {full_img_path}")
                        continue
                        
                    if not os.path.exists(full_emb_path):
                        logger.error(f"Representative embedding not found: {full_emb_path}")
                        continue

                    # Load the image
                    img = cv2.imread(full_img_path)
                    if img is None:
                        logger.error(f"Could not read image: {full_img_path}")
                        continue
                    
                    # Load the pre-computed embedding
                    try:
                        embedding = np.load(full_emb_path)
                    except Exception as e:
                        logger.error(f"Could not load embedding from {full_emb_path}: {str(e)}")
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Store the face info
                    self.representative_faces[int(label)] = {
                        'embedding': embedding,
                        'face_image': img_rgb,
                        'original_path': info['original_image']
                    }
                    # logger.info(f"Successfully loaded representative for cluster {label}")

                num_loaded = len(self.representative_faces)
                logger.info(f"Successfully loaded {num_loaded} representative faces")
                return num_loaded > 0

            except Exception as e:
                logger.error(f"Error loading representative faces: {str(e)}")
                logger.exception("Detailed error:")
                return False


    def save_clusters(self, face_data: List[Dict], labels: List[int], output_dir: str):
            """Save clustered faces and track source files for each cluster."""
            try:
                output_dir = os.path.abspath(output_dir)
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                
                # Track source files for each cluster
                cluster_sources = {}
                
                for label in set(labels):
                    if label == -1:
                        continue
                        
                    cluster_dir = os.path.join(output_dir, f"person_{label}")
                    os.makedirs(cluster_dir)
                    
                    # Initialize sources for this cluster
                    cluster_sources[label] = set()
                    
                    # Save representative face
                    if label in self.representative_faces:
                        rep_face = self.representative_faces[label]
                        rep_path = os.path.join(cluster_dir, "representative.jpg")
                        cv2.imwrite(rep_path, cv2.cvtColor(rep_face['face_image'], cv2.COLOR_RGB2BGR))
                        
                        # Save embedding
                        embedding_path = os.path.join(cluster_dir, "representative.npy")
                        np.save(embedding_path, rep_face['embedding'])
                        
                        # Add source file
                        cluster_sources[label].add(os.path.basename(rep_face['original_path']))
                    
                    # Save all faces in cluster and track sources
                    cluster_faces = [face_data[i] for i, l in enumerate(labels) if l == label]
                    for idx, face in enumerate(cluster_faces):
                        face_path = os.path.join(cluster_dir, f"face_{idx}.jpg")
                        cv2.imwrite(face_path, cv2.cvtColor(face['face_image'], cv2.COLOR_RGB2BGR))
                        
                        # Add source file
                        cluster_sources[label].add(os.path.basename(face['original_path']))
                    
                    # Save sources file for this cluster
                    sources_path = os.path.join(cluster_dir, "sources.txt")
                    with open(sources_path, 'w') as f:
                        for source in sorted(cluster_sources[label]):
                            f.write(f"{source}\n")
                
                # Save representative faces info
                rep_info = {}
                for label, face in self.representative_faces.items():
                    rep_info[str(label)] = {
                        'path': os.path.join(output_dir, f"person_{label}", "representative.jpg"),
                        'embedding_path': os.path.join(output_dir, f"person_{label}", "representative.npy"),
                        'original_image': face['original_path'],
                        'source_files': list(cluster_sources[label])
                    }
                
                rep_info_path = os.path.join(output_dir, "representatives.json")
                with open(rep_info_path, 'w') as f:
                    json.dump(rep_info, f, indent=4)
                logger.info(f"Saved representatives info to: {rep_info_path}")

            except Exception as e:
                logger.error(f"Error saving clusters: {str(e)}")
                logger.exception("Detailed error:")


    def select_representative_face(self, cluster_faces: List[Dict]) -> Dict:
        """Select the best representative face from a cluster."""
        try:
            if not cluster_faces:
                return None

            # Use simpler criteria initially
            face_scores = []
            for face in cluster_faces:
                # Combine detection score and size
                size_score = face['face_size'] / max(f['face_size'] for f in cluster_faces)
                total_score = 0.7 * face['det_score'] + 0.3 * size_score
                face_scores.append((face, total_score))
            
            # Select face with highest score
            best_face, best_score = max(face_scores, key=lambda x: x[1])
            logger.info(f"Selected representative face with score {best_score:.3f}")
            return best_face

        except Exception as e:
            logger.error(f"Error selecting representative face: {str(e)}")
            # Fallback to simple detection score
            return max(cluster_faces, key=lambda x: x['det_score'])

   
    def find_matching_faces(self, query_image: np.ndarray, clustered_dir: str, similarity_threshold: float = 0.83) -> Dict:
            """Find matches using representative faces and return source files."""
            try:
                if not self.representative_faces:
                    return {
                        'status': 'error',
                        'message': 'No representative faces loaded',
                        'matches': [],
                        'sources_file': None
                    }

                # Get query face
                faces = self.app.get(query_image)
                if not faces:
                    return {
                        'status': 'error',
                        'message': 'No face detected in query image',
                        'matches': [],
                        'sources_file': None
                    }
                
                query_face = max(faces, key=lambda x: x.det_score)
                matches = []
                matched_sources = {}  # Store filenames for each matching cluster
                
                # Compare with representative faces
                for label, rep_face in self.representative_faces.items():
                    similarity = self.calculate_similarity(query_face.embedding, rep_face['embedding'])
                    
                    if similarity >= similarity_threshold:
                        # Get source files for this cluster
                        cluster_dir = os.path.join(clustered_dir, f"person_{int(label)}")
                        sources_path = os.path.join(cluster_dir, "sources.txt")
                        
                        if os.path.exists(sources_path):
                            with open(sources_path, 'r') as f:
                                source_files = f.read().splitlines()
                        else:
                            source_files = []

                        matched_sources[label] = source_files
                        
                        matches.append({
                            'person_id': label,
                            'similarity': float(similarity * 100),
                            'original_image': rep_face['original_path'],
                            'source_files': source_files
                        })
                
                # Sort matches by similarity
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Create results directory
                results_dir = "face_search_results"
                os.makedirs(results_dir, exist_ok=True)
                
                # Generate sources file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sources_file = os.path.join(results_dir, f"sources_{timestamp}.txt")
                
                with open(sources_file, 'w') as f:
                    f.write(f"Face Search Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 50 + "\n\n")
                    
                    if not matches:
                        f.write("No matching faces found above similarity threshold.\n")
                    else:
                        f.write(f"Found matches in {len(matches)} clusters:\n\n")
                        for match in matches:
                            f.write(f"Person {match['person_id']} (Similarity: {match['similarity']:.1f}%)\n")
                            f.write("Found in files:\n")
                            for source in match['source_files']:
                                f.write(f"  - {source}\n")
                            f.write("\n")
                
                return {
                    'status': 'success',
                    'message': f"Found {len(matches)} matching persons",
                    'matches': matches,
                    'sources_file': sources_file
                }
                
            except Exception as e:
                logger.error(f"Error in face matching: {str(e)}")
                return {
                    'status': 'error',
                    'message': str(e),
                    'matches': [],
                    'sources_file': None
                }

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate normalized cosine similarity."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = np.dot(embedding1 / norm1, embedding2 / norm2)
        return (similarity + 1.0) / 2.0

def main():
    try:
        parser = argparse.ArgumentParser(description='Face Clustering and Search System')
        parser.add_argument('--input', default="album",
                          help='Input directory containing images')
        parser.add_argument('--output', default="clustered_faces",
                          help='Output directory for clustered faces')
        parser.add_argument('--batch-size', type=int, default=16,
                          help='Batch size for processing')
        parser.add_argument('--mode', choices=['cluster', 'search'], required=True,
                          help='Operation mode: cluster or search')
        parser.add_argument('--query', help='Query image for search mode')
        parser.add_argument('--threshold', type=float, default=0.83,
                          help='Similarity threshold for face matching')
        
        args = parser.parse_args()
        
        # Initialize system
        face_system = FaceSystem(batch_size=args.batch_size)
        
        if args.mode == 'cluster':
            # Load and process images
            image_paths = [
                os.path.join(args.input, f) for f in os.listdir(args.input)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if not image_paths:
                logger.error(f"No images found in {args.input}")
                return
                
            logger.info(f"Processing {len(image_paths)} images...")
            
            # Process in batches
            all_faces = []
            for i in range(0, len(image_paths), args.batch_size):
                batch = image_paths[i:i + args.batch_size]
                faces = face_system.process_image_batch(batch)
                all_faces.extend(faces)
                logger.info(f"Processed {min(i + args.batch_size, len(image_paths))}/{len(image_paths)} images...")
            
            # Perform clustering
            logger.info("Clustering faces...")
            labels = face_system.cluster_faces(all_faces)
            
            # Save results
            logger.info("Saving results...")
            face_system.save_clusters(all_faces, labels, args.output)
            
            unique_people = len(set(labels) - {-1})
            logger.info(f"Found {unique_people} unique people across {len(all_faces)} faces")
            
        elif args.mode == 'search':
            # Search mode
            if not args.query:
                logger.error("Query image required for search mode")
                return
                
            if not os.path.exists(args.output):
                logger.error("Clustered faces directory not found. Run clustering first.")
                return
                
            # Load representative faces first
            if not face_system.load_representative_faces(args.output):
                logger.error("Failed to load representative faces")
                return
                
            # Load query image
            query_img = cv2.imread(args.query)
            if query_img is None:
                logger.error("Could not read query image")
                return
                
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            
         # Perform search
            results = face_system.find_matching_faces(query_img, args.output, args.threshold)
            
            if results['status'] == 'success':
                logger.info(results['message'])
                if results['matches']:
                    logger.info(f"Results saved to: {results['sources_file']}")
                    for match in results['matches']:
                        logger.info(f"Match found in cluster {match['person_id']}")
                        logger.info(f"Similarity: {match['similarity']:.2f}%")
                        logger.info(f"Original image: {match['original_image']}")
                        logger.info("-" * 50)
                else:
                    logger.info("No matches found above threshold")
            else:
                logger.error(results['message'])
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()