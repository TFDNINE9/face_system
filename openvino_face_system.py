import numpy as np
import os
import cv2
import logging
import shutil
import threading
import argparse
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
import openvino as ov

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenVinoFaceSystem:
    def __init__(self, batch_size=16):
        try:
            self.batch_size = batch_size
            self.image_size_limit = 1920
            self.thread_lock = threading.Lock()
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
            
            # Initialize OpenVINO Core
            self.core = ov.Core()
            available_devices = self.core.available_devices
            logger.info(f"Available OpenVINO devices: {available_devices}")
            
            # Set device priority
            if 'GPU' in available_devices:
                self.device = 'GPU'
                logger.info(f"Using Intel GPU: {self.core.get_property('GPU', 'FULL_DEVICE_NAME')}")
            else:
                self.device = 'CPU'
                logger.info(f"No GPU found. Using CPU.")
            
            # Model paths from insightface directory
            user_home = os.path.expanduser("~")
            self.model_dir = os.path.join(user_home, ".insightface", "models", "buffalo_l")
            
            # Load detection model
            det_model_path = os.path.join(self.model_dir, "det_10g.onnx")
            if not os.path.exists(det_model_path):
                raise FileNotFoundError(f"Detection model not found at {det_model_path}. Run InsightFace once to download models.")
                
            logger.info(f"Loading detection model from {det_model_path}")
            self.det_model = self.core.compile_model(det_model_path, self.device)
            logger.info(f"Detection model loaded on {self.device}")
            
            # Load recognition model
            rec_model_path = os.path.join(self.model_dir, "w600k_r50.onnx")
            if not os.path.exists(rec_model_path):
                raise FileNotFoundError(f"Recognition model not found at {rec_model_path}. Run InsightFace once to download models.")
                
            logger.info(f"Loading recognition model from {rec_model_path}")
            self.rec_model = self.core.compile_model(rec_model_path, self.device)
            logger.info(f"Recognition model loaded on {self.device}")
            
            # Image preprocessing parameters
            self.det_mean = 127.5
            self.det_std = 128.0
            self.rec_mean = 127.5
            self.rec_std = 127.5
            
            # Detection parameters
            self.det_size = (640, 640)
            self.det_threshold = 0.5
            
            # Clustering parameters
            self.clustering_params = {
                'eps': 0.9,
                'min_samples': 2,
                'same_image_threshold': 0.70,
                'face_similarity_metric': 'cosine'
            }
            
            # Representative faces storage
            self.representative_faces = {}
            
            logger.info("OpenVINO Face System initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing Face System: {str(e)}")
            logger.exception("Detailed error:")
            raise
    
    def preprocess_for_detection(self, img):
        """Preprocess image for face detection"""
        # Resize image to detection size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.det_size[0]) / self.det_size[1]
        
        if im_ratio > model_ratio:
            new_height = self.det_size[0]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.det_size[1]
            new_height = int(new_width * im_ratio)
            
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # Create a canvas of detection size and paste the resized image
        canvas = np.zeros((self.det_size[0], self.det_size[1], 3), dtype=np.uint8)
        canvas[:new_height, :new_width, :] = resized_img
        
        # Normalize
        canvas = (canvas - self.det_mean) / self.det_std
        
        # HWC to CHW format
        canvas = canvas.transpose(2, 0, 1)
        
        # Add batch dimension
        canvas = np.expand_dims(canvas, axis=0).astype(np.float32)
        
        return canvas, (new_height / img.shape[0], new_width / img.shape[1])
    
    def detect_faces(self, img):
        """Detect faces using the detection model"""
        input_blob, scale = self.preprocess_for_detection(img)
        
        # Get input and output names
        input_name = self.det_model.inputs[0].get_any_name()
        output_names = [output.get_any_name() for output in self.det_model.outputs]
        
        # Run inference
        results = self.det_model({input_name: input_blob})
        
        # Parse results
        bboxes = []
        for output_name in output_names:
            if output_name in results:
                output = results[output_name]
                # Different formats for different outputs
                if len(output.shape) == 3:  # [1, N, 15] output with scores and landmarks
                    for det in output[0]:
                        if det[4] < self.det_threshold:
                            continue
                        
                        # Extract bbox and landmarks
                        bbox = det[0:4].astype(np.float32)
                        score = det[4].astype(np.float32)
                        landmarks = det[5:15].reshape(5, 2).astype(np.float32)
                        
                        # Rescale to original image
                        bbox[0] /= scale[1]
                        bbox[1] /= scale[0]
                        bbox[2] /= scale[1]
                        bbox[3] /= scale[0]
                        
                        landmarks[:, 0] /= scale[1]
                        landmarks[:, 1] /= scale[0]
                        
                        # Convert to int for display
                        bbox = bbox.astype(np.int32)
                        
                        # Store face info
                        bboxes.append({
                            'bbox': (bbox[0], bbox[1], bbox[2], bbox[3]),
                            'score': float(score),
                            'landmarks': landmarks
                        })
                elif len(output.shape) == 2:  # Different format
                    for det in output:
                        if len(det) >= 15 and det[4] >= self.det_threshold:
                            bbox = det[0:4].astype(np.float32)
                            score = det[4].astype(np.float32)
                            landmarks = det[5:15].reshape(5, 2).astype(np.float32)
                            
                            # Rescale
                            bbox[0] /= scale[1]
                            bbox[1] /= scale[0]
                            bbox[2] /= scale[1]
                            bbox[3] /= scale[0]
                            
                            landmarks[:, 0] /= scale[1]
                            landmarks[:, 1] /= scale[0]
                            
                            # Store face info
                            bboxes.append({
                                'bbox': (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                                'score': float(score),
                                'landmarks': landmarks
                            })
        
        return bboxes
    
    def align_face(self, img, landmarks, face_size=(112, 112)):
        """Align face using landmarks"""
        # Standard 5 facial landmarks positions in normalized coordinates
        src = np.array([
            [30.2946, 51.6963],  # Left eye
            [65.5318, 51.6963],  # Right eye
            [48.0252, 71.7366],  # Nose
            [33.5493, 92.3655],  # Left mouth
            [62.7299, 92.3655],  # Right mouth
        ], dtype=np.float32)
        
        # Scale source landmarks to target size
        src[:, 0] *= face_size[0] / 96.0
        src[:, 1] *= face_size[1] / 96.0
        
        dst = landmarks.astype(np.float32)
        
        # Get transformation matrix
        M = cv2.estimateAffinePartial2D(dst, src)[0]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(img, M, face_size, borderValue=0.0)
        
        return aligned_face
    
    def extract_embedding(self, aligned_face):
        """Extract face embedding using recognition model"""
        # Preprocess aligned face - convert to RGB first (if BGR)
        if aligned_face.shape[2] == 3 and aligned_face.dtype == np.uint8:
            img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        else:
            img = aligned_face
            
        # Normalize
        img = (img.astype(np.float32) - self.rec_mean) / self.rec_std
        
        # HWC to CHW format
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Get input and output names
        input_name = self.rec_model.inputs[0].get_any_name()
        output_name = self.rec_model.outputs[0].get_any_name()
        
        # Run inference
        result = self.rec_model({input_name: img})
        embedding = result[output_name][0]
        
        return embedding
    
    def process_image_batch(self, image_paths: List[str]) -> List[Dict]:
        """Process a batch of images for face detection and recognition"""
        all_faces = []
        
        def process_single_image(path):
            try:
                img = cv2.imread(path)
                if img is None:
                    logger.warning(f"Could not read image: {path}")
                    return []
                    
                h, w = img.shape[:2]
                if max(h, w) > self.image_size_limit:
                    scale = self.image_size_limit / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_AREA)
                
                # Detect faces
                with self.thread_lock:
                    faces = self.detect_faces(img)
                
                if not faces:
                    logger.debug(f"No faces detected in image: {os.path.basename(path)}")
                    return []
                
                logger.debug(f"Detected {len(faces)} faces in image: {os.path.basename(path)}")
                processed_faces = []
                
                for idx, face in enumerate(faces):
                    try:
                        if face['landmarks'] is None:
                            continue
                            
                        # Align face
                        aligned_face = self.align_face(img, face['landmarks'])
                        
                        # Extract embedding
                        embedding = self.extract_embedding(aligned_face)
                        
                        # Calculate face size
                        x1, y1, x2, y2 = face['bbox']
                        face_size = (x2 - x1) * (y2 - y1)
                        
                        face_info = {
                            'embedding': embedding,
                            'face_image': aligned_face,
                            'original_path': path,
                            'face_index': idx,
                            'det_score': face['score'],
                            'bbox': face['bbox'],
                            'landmarks': face['landmarks'],
                            'face_size': face_size
                        }
                        processed_faces.append(face_info)
                    except Exception as e:
                        logger.error(f"Error processing face {idx} in {path}: {str(e)}")
                        continue
                
                return processed_faces
                
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                return []
                
        # Process images in parallel
        results = list(self.thread_pool.map(process_single_image, image_paths))
        for face_list in results:
            all_faces.extend(face_list)
            
        return all_faces
    
    def cluster_faces(self, face_data: List[Dict]) -> List[int]:
        """Cluster faces and store representative faces"""
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
                    cv2.imwrite(rep_path, rep_face['face_image'])
                    
                    # Save embedding
                    embedding_path = os.path.join(cluster_dir, "representative.npy")
                    np.save(embedding_path, rep_face['embedding'])
                    
                    # Add source file
                    cluster_sources[label].add(os.path.basename(rep_face['original_path']))
                
                # Save all faces in cluster and track sources
                cluster_faces = [face_data[i] for i, l in enumerate(labels) if l == label]
                for idx, face in enumerate(cluster_faces):
                    face_path = os.path.join(cluster_dir, f"face_{idx}.jpg")
                    cv2.imwrite(face_path, face['face_image'])
                    
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
                
                # Store the face info
                self.representative_faces[int(label)] = {
                    'embedding': embedding,
                    'face_image': img,
                    'original_path': info['original_image']
                }

            num_loaded = len(self.representative_faces)
            logger.info(f"Successfully loaded {num_loaded} representative faces")
            return num_loaded > 0

        except Exception as e:
            logger.error(f"Error loading representative faces: {str(e)}")
            logger.exception("Detailed error:")
            return False
            
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate normalized cosine similarity."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = np.dot(embedding1 / norm1, embedding2 / norm2)
        return (similarity + 1.0) / 2.0
        
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

            # Detect faces in query image
            faces = self.detect_faces(query_image)
            if not faces:
                return {
                    'status': 'error',
                    'message': 'No face detected in query image',
                    'matches': [],
                    'sources_file': None
                }
            
            # Get the face with highest detection score
            query_face = max(faces, key=lambda x: x['score'])
            
            if query_face['landmarks'] is None:
                return {
                    'status': 'error',
                    'message': 'No landmarks detected for query face',
                    'matches': [],
                    'sources_file': None
                }
                
            # Align face and get embedding
            aligned_face = self.align_face(query_image, query_face['landmarks'])
            query_embedding = self.extract_embedding(aligned_face)
            
            matches = []
            matched_sources = {}  # Store filenames for each matching cluster
            
            # Compare with representative faces
            for label, rep_face in self.representative_faces.items():
                similarity = self.calculate_similarity(query_embedding, rep_face['embedding'])
                
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

def main():
    try:
        parser = argparse.ArgumentParser(description='OpenVINO Face Clustering and Search System')
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
        parser.add_argument('--det-threshold', type=float, default=0.4,
                          help='Detection confidence threshold (default: 0.4)')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug logging')
        
        args = parser.parse_args()
        
        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize system
        face_system = OpenVinoFaceSystem(batch_size=args.batch_size)
        face_system.det_threshold = args.det_threshold
        
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
            face_counts = []
            
            for i in range(0, total_images, args.batch_size):
                batch = image_paths[i:i + args.batch_size]
                faces = face_system.process_image_batch(batch)
                all_faces.extend(faces)
                face_counts.append(len(faces))
                
                # Log progress with timing
                images_processed = min(i + args.batch_size, total_images)
                batch_time = time.time() - batch_start_time
                if batch_time > 0:  # Avoid division by zero
                    images_per_second = len(batch) / batch_time
                    logger.info(f"Processed {images_processed}/{total_images} images "
                              f"({images_per_second:.2f} images/second) - Found {len(faces)} faces in this batch")
                else:
                    logger.info(f"Processed {images_processed}/{total_images} images")
                batch_start_time = time.time()
            
            # Make sure we have faces before proceeding
            if not all_faces:
                logger.error("No faces were detected in any of the images. Try lowering the detection threshold with --det-threshold")
                return
            
            logger.info(f"Total faces found: {len(all_faces)} in {total_images} images")
            logger.info(f"Face detection statistics: {sum(face_counts)/len(face_counts):.2f} faces per batch on average")
            
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
            noise_points = list(labels).count(-1)
            total_time = time.time() - start_time
            
            logger.info(f"""
Clustering completed:
- Total time: {total_time:.2f} seconds
- Images processed: {total_images}
- Processing speed: {total_images/total_time:.2f} images/second
- Faces found: {len(all_faces)}
- Unique people: {unique_people}
- Noise points (unassigned faces): {noise_points}
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