import numpy as np
import os
from sklearn.cluster import DBSCAN
import shutil
import cv2
import logging
from typing import List, Dict
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import threading
import openvino as ov
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from datetime import datetime
import time

# Force OpenVINO to use GPU
os.environ['OPENVINO_DEVICE'] = 'GPU'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceSystemArc:
    def __init__(self, batch_size=16):
        try:
            # Only use OpenVINO provider - force it
            providers = ['OpenVINOExecutionProvider']
            
            # Initialize OpenVINO Core to check GPU
            try:
                core = ov.Core()
                available_devices = core.available_devices
                logger.info(f"Available OpenVINO devices: {available_devices}")
                
                if 'GPU' in available_devices:
                    logger.info(f"Intel GPU detected: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")
                else:
                    logger.warning("No Intel GPU detected by OpenVINO")
            except Exception as e:
                logger.warning(f"Could not initialize OpenVINO Core: {e}")
            
            # Initialize face analysis with ONLY OpenVINO provider
            logger.info("Initializing FaceAnalysis with forced OpenVINO provider")
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
            
            # Prepare with optimal detection size
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"Model preparation completed with providers: {providers}")

            # Initialize other parameters
            self.batch_size = batch_size
            self.image_size_limit = 1920
            self.thread_lock = threading.Lock()
            
            # Initialize thread pool for parallel processing
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
            
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
            logger.error(f"Error initializing system: {str(e)}")
            logger.exception("Detailed error:")
            # If OpenVINO fails, fall back to CPU
            logger.info("Falling back to CPU provider")
            providers = ['CPUExecutionProvider']
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Initialized with CPU fallback")
            
            # Continue with other initializations
            self.batch_size = batch_size
            self.image_size_limit = 1920
            self.thread_lock = threading.Lock()
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
            self.clustering_params = {
                'eps': 0.9,
                'min_samples': 2,
                'same_image_threshold': 0.70,
                'face_similarity_metric': 'cosine'
            }
            self.representative_faces = {}

    def process_image_batch(self, image_paths: List[str]) -> List[Dict]:
        """Process a batch of images with parallel processing."""
        all_faces = []
        try:
            # Process images in parallel
            def process_single_image(path):
                try:
                    img = cv2.imread(path)
                    if img is None:
                        return []
                        
                    h, w = img.shape[:2]
                    if max(h, w) > self.image_size_limit:
                        scale = self.image_size_limit / max(h, w)
                        img = cv2.resize(img, None, fx=scale, fy=scale, 
                                       interpolation=cv2.INTER_AREA)
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    with self.thread_lock:
                        faces = self.app.get(img_rgb)
                    
                    if not faces:
                        return []
                        
                    return [{
                        'embedding': face.embedding,
                        'face_image': face_align.norm_crop(img_rgb, face.kps),
                        'original_path': path,
                        'face_index': idx,
                        'det_score': float(face.det_score),
                        'bbox': face.bbox,
                        'kps': face.kps,
                        'face_size': (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                    } for idx, face in enumerate(faces)]
                        
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
                    return []

            # Process images in parallel using thread pool
            results = list(self.thread_pool.map(process_single_image, image_paths))
            for face_list in results:
                all_faces.extend(face_list)

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            logger.exception("Detailed error:")
        
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
        parser.add_argument('--batch-size', type=int, default=32,
                          help='Batch size for processing')
        parser.add_argument('--mode', choices=['cluster', 'search'], required=True,
                          help='Operation mode: cluster or search')
        parser.add_argument('--query', help='Query image for search mode')
        parser.add_argument('--threshold', type=float, default=0.83,
                          help='Similarity threshold for face matching')
        
        args = parser.parse_args()
        
        # Initialize system with Intel optimizations
        face_system = FaceSystemArc(batch_size=args.batch_size)
        
        if args.mode == 'cluster':
            start_time = time.time()
            logger.info("Starting clustering process...")
            
            # Load and process images
            image_paths = [
                os.path.join(args.input, f) for f in os.listdir(args.input)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if not image_paths:
                logger.error(f"No images found in {args.input}")
                return
            
            total_images = len(image_paths)
            logger.info(f"Found {total_images} images to process")
            
            # Process in batches with timing
            all_faces = []
            batch_start_time = time.time()
            for i in range(0, total_images, args.batch_size):
                batch = image_paths[i:i + args.batch_size]
                faces = face_system.process_image_batch(batch)
                all_faces.extend(faces)
                
                # Log progress with timing
                images_processed = min(i + args.batch_size, total_images)
                batch_time = time.time() - batch_start_time
                if batch_time > 0:  # Avoid division by zero
                    images_per_second = len(batch) / batch_time
                    logger.info(f"Processed {images_processed}/{total_images} images "
                              f"({images_per_second:.2f} images/second)")
                else:
                    logger.info(f"Processed {images_processed}/{total_images} images")
                batch_start_time = time.time()
            
            # Make sure we have faces before proceeding
            if not all_faces:
                logger.error("No faces were detected in any of the images")
                return
            
            # Perform clustering with timing
            logger.info(f"Starting face clustering for {len(all_faces)} detected faces...")
            cluster_start_time = time.time()
            labels = face_system.cluster_faces(all_faces)
            cluster_time = time.time() - cluster_start_time
            
            # Save results with timing
            logger.info("Saving clustered results...")
            save_start_time = time.time()
            face_system.save_clusters(all_faces, labels, args.output)
            save_time = time.time() - save_start_time
            
            # Final statistics
            unique_people = len(set(labels) - {-1})
            total_time = time.time() - start_time
            
            logger.info(f"""
Clustering completed:
- Total time: {total_time:.2f} seconds
- Images processed: {total_images}
- Processing speed: {total_images/total_time:.2f} images/second
- Faces found: {len(all_faces)}
- Unique people: {unique_people}
- Clustering time: {cluster_time:.2f} seconds
- Saving time: {save_time:.2f} seconds
""")
            
        elif args.mode == 'search':
            # Search mode timing
            start_time = time.time()
            logger.info("Starting face search...")
            
            if not args.query:
                logger.error("Query image required for search mode")
                return
            
            if not os.path.exists(args.output):
                logger.error("Clustered faces directory not found. Run clustering first.")
                return
            
            # Load representative faces
            load_start_time = time.time()
            if not face_system.load_representative_faces(args.output):
                logger.error("Failed to load representative faces")
                return
            load_time = time.time() - load_start_time
            
            # Load and process query image
            query_img = cv2.imread(args.query)
            if query_img is None:
                logger.error("Could not read query image")
                return
            
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            
            # Perform search with timing
            search_start_time = time.time()
            results = face_system.find_matching_faces(query_img, args.output, args.threshold)
            search_time = time.time() - search_start_time
            
            total_time = time.time() - start_time
            
            if results['status'] == 'success':
                logger.info(f"""
Search completed:
- Total time: {total_time:.2f} seconds
- Loading time: {load_time:.2f} seconds
- Search time: {search_time:.2f} seconds
- Matches found: {len(results['matches'])}
""")
                
                if results['matches']:
                    logger.info(f"Results saved to: {results['sources_file']}")
                    for match in results['matches']:
                        logger.info(f"Match found in cluster {match['person_id']}")
                        logger.info(f"Similarity: {match['similarity']:.2f}%")
                        logger.info("-" * 50)
                else:
                    logger.info("No matches found above threshold")
            else:
                logger.error(results['message'])
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.exception("Detailed error trace:")

if __name__ == "__main__":
    main()