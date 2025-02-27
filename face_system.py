import numpy as np
import os
import shutil
import cv2
import logging
import threading
import json
import time
from datetime import datetime
from typing import List, Dict
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceSystem:
    def __init__(self, batch_size=16):
        try:
            # Configure providers based on available hardware
            providers = []
            
            # Check if CUDA is available and add it as the first provider
            if self._is_cuda_available():
                providers.append('CUDAExecutionProvider')
                logger.info("CUDA support enabled")
            
            # Add OpenVINO if available
            if self._is_openvino_available():
                providers.append('OpenVINOExecutionProvider')
                logger.info("OpenVINO support enabled")
                
            # Always include CPU as a fallback
            providers.append('CPUExecutionProvider')
            
            # Initialize face analysis with available providers
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
            
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"Model preparation completed with providers: {providers}")
            
            # Initialize other parameters
            self.batch_size = batch_size
            self.image_size_limit = 1920
            self.thread_lock = threading.Lock()
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
            logger.error(f"Error initializing FaceSystem: {str(e)}")
            logger.exception("Detailed error:")
            raise
    
    def _is_cuda_available(self):
        """Check if CUDA is available for acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _is_openvino_available(self):
        """Check if OpenVINO is available for acceleration."""
        try:
            import openvino as ov
            core = ov.Core()
            return 'GPU' in core.available_devices
        except ImportError:
            return False

    def process_image_batch(self, image_paths: List[str]) -> List[Dict]:
        """Process a batch of images for face detection and recognition."""
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
                
        return all_faces

    def cluster_faces(self, face_data: List[Dict]) -> List[int]:
        """Cluster faces based on embeddings."""
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

            # Use detection score and size to select representative
            face_scores = []
            for face in cluster_faces:
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

    def save_clusters(self, face_data: List[Dict], labels: List[int], output_dir: str, album_dir: str):
        """Save clustered faces and copy original images to album."""
        try:
            output_dir = os.path.abspath(output_dir)
            album_dir = os.path.abspath(album_dir)
            
            # Ensure album directory exists
            os.makedirs(album_dir, exist_ok=True)
            
            # Make sure the output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Load existing representatives to retain information
            existing_reps = {}
            existing_rep_file = os.path.join(output_dir, "representatives.json")
            if os.path.exists(existing_rep_file):
                try:
                    with open(existing_rep_file, 'r') as f:
                        existing_reps = json.load(f)
                    logger.info(f"Loaded {len(existing_reps)} existing representative faces")
                except Exception as e:
                    logger.error(f"Error loading existing representatives: {str(e)}")
            
            # Track source files for each cluster
            cluster_sources = {}
            copied_files = set()
            
            # Get max person ID to ensure we don't overlap with existing clusters
            max_existing_id = -1
            for label_str in existing_reps.keys():
                try:
                    label = int(label_str)
                    max_existing_id = max(max_existing_id, label)
                except:
                    pass
            
            # Create ID mapping from new clusters to avoid collisions
            id_mapping = {}
            next_id = max_existing_id + 1
            
            for label in set(labels):
                if label == -1:  # Skip noise
                    continue
                    
                # Assign new ID to avoid collisions
                id_mapping[label] = next_id
                next_id += 1
                
                new_label = id_mapping[label]
                cluster_dir = os.path.join(output_dir, f"person_{new_label}")
                os.makedirs(cluster_dir, exist_ok=True)
                
                # Initialize sources for this cluster
                cluster_sources[new_label] = set()
                
                # Save representative face
                if label in self.representative_faces:
                    rep_face = self.representative_faces[label]
                    rep_path = os.path.join(cluster_dir, "representative.jpg")
                    cv2.imwrite(rep_path, cv2.cvtColor(rep_face['face_image'], cv2.COLOR_RGB2BGR))
                    
                    # Save embedding
                    embedding_path = os.path.join(cluster_dir, "representative.npy")
                    np.save(embedding_path, rep_face['embedding'])
                    
                    # Copy original image to album if it's not already there
                    orig_filename = os.path.basename(rep_face['original_path'])
                    album_path = os.path.join(album_dir, orig_filename)
                    
                    if not os.path.exists(album_path):
                        shutil.copy2(rep_face['original_path'], album_path)
                        logger.debug(f"Copied {orig_filename} to album")
                    
                    copied_files.add(orig_filename)
                    cluster_sources[new_label].add(orig_filename)
                
                # Save all faces in cluster and track sources
                cluster_faces = [face_data[i] for i, l in enumerate(labels) if l == label]
                for idx, face in enumerate(cluster_faces):
                    face_path = os.path.join(cluster_dir, f"face_{idx}.jpg")
                    cv2.imwrite(face_path, cv2.cvtColor(face['face_image'], cv2.COLOR_RGB2BGR))
                    
                    # Copy original image to album if it's not already there
                    orig_filename = os.path.basename(face['original_path'])
                    album_path = os.path.join(album_dir, orig_filename)
                    
                    if not os.path.exists(album_path):
                        shutil.copy2(face['original_path'], album_path)
                        logger.debug(f"Copied {orig_filename} to album")
                    
                    copied_files.add(orig_filename)
                    cluster_sources[new_label].add(orig_filename)
                
                # Save sources file for this cluster
                sources_path = os.path.join(cluster_dir, "sources.txt")
                with open(sources_path, 'w') as f:
                    for source in sorted(cluster_sources[new_label]):
                        f.write(f"{source}\n")
            
            # Save representative faces info (merging with existing)
            rep_info = existing_reps.copy()  # Start with existing representatives
            for old_label, new_label in id_mapping.items():
                if old_label in self.representative_faces:
                    rep_face = self.representative_faces[old_label]
                    rep_info[str(new_label)] = {
                        'path': os.path.join(output_dir, f"person_{new_label}", "representative.jpg"),
                        'embedding_path': os.path.join(output_dir, f"person_{new_label}", "representative.npy"),
                        'original_image': os.path.basename(rep_face['original_path']),
                        'source_files': list(cluster_sources[new_label])
                    }
            
            # Save updated representatives file
            rep_info_path = os.path.join(output_dir, "representatives.json")
            with open(rep_info_path, 'w') as f:
                json.dump(rep_info, f, indent=4)
            logger.info(f"Saved representatives info to: {rep_info_path}")
            
            # Update or create a manifest file with processing details
            manifest_path = os.path.join(output_dir, "manifest.json")
            manifest = {}
            
            # Load existing manifest if it exists
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading manifest: {str(e)}")
            
            # Update manifest with latest processing info
            manifest.update({
                "last_processed_at": datetime.now().isoformat(),
                "total_images": manifest.get("total_images", 0) + len(copied_files),
                "total_faces": manifest.get("total_faces", 0) + len(face_data),
                "unique_people": len(rep_info),
                "latest_batch": {
                    "images": len(copied_files),
                    "faces": len(face_data),
                    "people": len(id_mapping)
                }
            })
            
            # Save updated manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=4)
            
            logger.info(f"Processed {len(face_data)} faces into {len(id_mapping)} new clusters")
            logger.info(f"Copied {len(copied_files)} images to album directory")
            
            return {
                "existing_people": len(existing_reps),
                "new_people": len(id_mapping),
                "total_people": len(rep_info),
                "processed_images": len(copied_files),
                "processed_faces": len(face_data)
            }

        except Exception as e:
            logger.error(f"Error saving clusters: {str(e)}")
            logger.exception("Detailed error:")
            raise

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

    def find_matching_faces(self, query_image: np.ndarray, clustered_dir: str, album_dir: str, similarity_threshold: float = 0.83) -> Dict:
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
            
            # Use face with highest detection score
            query_face = max(faces, key=lambda x: x.det_score)
            matches = []
            
            # Compare with representative faces
            for label, rep_face in self.representative_faces.items():
                similarity = self.calculate_similarity(query_face.embedding, rep_face['embedding'])
                
                if similarity >= similarity_threshold:
                    # Get source files for this cluster
                    cluster_dir = os.path.join(clustered_dir, f"person_{int(label)}")
                    sources_path = os.path.join(cluster_dir, "sources.txt")
                    
                    source_files = []
                    if os.path.exists(sources_path):
                        with open(sources_path, 'r') as f:
                            source_files = f.read().splitlines()
                    
                    # Verify all source files exist in album
                    verified_sources = []
                    for source in source_files:
                        album_path = os.path.join(album_dir, source)
                        if os.path.exists(album_path):
                            verified_sources.append(source)
                        else:
                            logger.warning(f"Source file not found in album: {source}")
                    
                    matches.append({
                        'person_id': label,
                        'similarity': float(similarity * 100),
                        'representative_image': os.path.basename(rep_face['original_path']),
                        'source_files': verified_sources
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
        
def clear_temp_folder(temp_dir):
    """Clear all files from temporary directory after processing."""
    try:
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            logger.info(f"Cleared all files from temporary directory: {temp_dir}")
        else:
            logger.warning(f"Temporary directory does not exist: {temp_dir}")
    except Exception as e:
        logger.error(f"Error clearing temporary directory: {str(e)}")
        logger.exception("Detailed error:")

def process_temp_album(temp_album_dir, cluster_dir, album_dir, batch_size=16):
    """
    Process images from temp_album with integration into existing clusters.
    
    Args:
        temp_album_dir: Directory containing temporary uploaded images
        cluster_dir: Directory where face clusters are stored
        album_dir: Directory where original images should be copied
        batch_size: Number of images to process in each batch
    
    Returns:
        Dictionary with processing results
    """
    try:
        temp_album_dir = os.path.abspath(temp_album_dir)
        cluster_dir = os.path.abspath(cluster_dir)
        album_dir = os.path.abspath(album_dir)
        
        # Ensure directories exist
        os.makedirs(temp_album_dir, exist_ok=True)
        os.makedirs(cluster_dir, exist_ok=True)
        os.makedirs(album_dir, exist_ok=True)
        
        # Check if there are images to process
        image_paths = [
            os.path.join(temp_album_dir, f) for f in os.listdir(temp_album_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not image_paths:
            logger.warning(f"No images found in {temp_album_dir}")
            return {
                "status": "warning",
                "message": "No images found in temporary album",
                "details": None
            }
        
        logger.info(f"Found {len(image_paths)} images to process in temporary album")
        
        # Start processing
        start_time = time.time()
        
        # Initialize face system
        face_system = FaceSystem(batch_size=batch_size)
        
        # First, load existing representatives if available
        existing_representatives = {}
        existing_clusters = {}
        existing_rep_file = os.path.join(cluster_dir, "representatives.json")
        
        if os.path.exists(existing_rep_file):
            logger.info("Loading existing representative faces")
            if face_system.load_representative_faces(cluster_dir):
                existing_representatives = face_system.representative_faces.copy()
                logger.info(f"Loaded {len(existing_representatives)} existing representative faces")
                
                # Also load details about each existing cluster
                try:
                    with open(existing_rep_file, 'r') as f:
                        existing_clusters = json.load(f)
                    logger.info(f"Loaded details for {len(existing_clusters)} existing clusters")
                except Exception as e:
                    logger.error(f"Error loading cluster details: {str(e)}")
        
        # Process images in batches
        all_faces = []
        for i in range(0, len(image_paths), face_system.batch_size):
            batch = image_paths[i:i + face_system.batch_size]
            faces = face_system.process_image_batch(batch)
            all_faces.extend(faces)
            logger.info(f"Processed {min(i + face_system.batch_size, len(image_paths))}/{len(image_paths)} images...")
        
        if not all_faces:
            logger.warning("No faces detected in any images")
            # Still copy images to album even if no faces detected
            for image_path in image_paths:
                filename = os.path.basename(image_path)
                album_path = os.path.join(album_dir, filename)
                if not os.path.exists(album_path):
                    shutil.copy2(image_path, album_path)
            
            # Clear temp folder
            clear_temp_folder(temp_album_dir)
            
            return {
                "status": "warning",
                "message": "No faces detected in any images, but files were copied to album",
                "details": {
                    "processed_images": len(image_paths),
                    "processed_faces": 0,
                    "new_people": 0,
                    "total_people": len(existing_representatives)
                }
            }
        
        # IMPROVED APPROACH: Compare with existing representatives first
        if existing_representatives:
            logger.info("Comparing new faces with existing clusters")
            # Maps faces to their matching existing cluster (if any)
            face_to_existing_cluster = {}
            
            # For each new face, check if it matches any existing cluster
            for face_idx, face in enumerate(all_faces):
                face_embedding = face['embedding']
                best_match = None
                best_score = 0
                
                # Compare with all existing representatives
                for cluster_id, rep_face in existing_representatives.items():
                    similarity = face_system.calculate_similarity(face_embedding, rep_face['embedding'])
                    
                    # If similarity exceeds threshold, consider it a match
                    if similarity >= 0.90:  # Using slightly lower threshold for matching to existing clusters
                        if similarity > best_score:
                            best_score = similarity
                            best_match = cluster_id
                
                # If we found a match, map this face to that cluster
                if best_match is not None:
                    face_to_existing_cluster[face_idx] = (best_match, best_score)
                    logger.debug(f"Face {face_idx} matched to existing cluster {best_match} with score {best_score:.2f}")
            
            logger.info(f"Found {len(face_to_existing_cluster)} faces matching existing clusters")
        else:
            face_to_existing_cluster = {}
            logger.info("No existing clusters to compare with")
        
        # Now perform clustering on remaining faces that didn't match existing clusters
        faces_to_cluster = []
        face_indices_to_cluster = []
        
        for idx, face in enumerate(all_faces):
            if idx not in face_to_existing_cluster:
                faces_to_cluster.append(face)
                face_indices_to_cluster.append(idx)
        
        if faces_to_cluster:
            logger.info(f"Clustering {len(faces_to_cluster)} faces that didn't match existing clusters")
            embeddings = np.array([face['embedding'] for face in faces_to_cluster])
            norms = np.linalg.norm(embeddings, axis=1)
            normalized_embeddings = embeddings / norms[:, np.newaxis]
            
            clustering = DBSCAN(
                eps=0.9,
                min_samples=2,
                metric='euclidean',
                n_jobs=-1
            ).fit(normalized_embeddings)
            
            cluster_labels = clustering.labels_
            logger.info(f"Created {len(set(cluster_labels) - {-1})} new clusters")
        else:
            cluster_labels = []
            logger.info("No new clusters needed - all faces matched to existing clusters")
        
        # Get next available cluster ID
        next_cluster_id = 0
        if existing_representatives:
            next_cluster_id = max(int(k) for k in existing_representatives.keys()) + 1
            logger.info(f"Next available cluster ID: {next_cluster_id}")
        
        # Initialize storage for all processed faces
        final_clusters = {}  # Maps cluster ID to list of face indices
        
        # First, add faces that matched existing clusters
        for face_idx, (cluster_id, score) in face_to_existing_cluster.items():
            if cluster_id not in final_clusters:
                final_clusters[cluster_id] = []
            final_clusters[cluster_id].append(face_idx)
        
        # Then add new clusters
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Skip noise points
                # Map the relative index back to the original face index
                original_idx = face_indices_to_cluster[idx]
                
                # New unique cluster ID
                new_cluster_id = next_cluster_id + label
                
                if new_cluster_id not in final_clusters:
                    final_clusters[new_cluster_id] = []
                final_clusters[new_cluster_id].append(original_idx)
        
        # Also include unmatched singletons (noise points) as their own clusters
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # This is a noise point
                original_idx = face_indices_to_cluster[idx]
                # Create a unique cluster ID for this singleton
                singleton_cluster_id = next_cluster_id + len(set(cluster_labels) - {-1}) + idx
                final_clusters[singleton_cluster_id] = [original_idx]
        
        # Now process all clusters (existing matches + new clusters)
        logger.info(f"Processing {len(final_clusters)} total clusters")
        
        # Create representatives for all clusters
        representatives = {}
        
        # First, copy existing representatives that were matched
        for cluster_id in final_clusters.keys():
            if cluster_id in existing_representatives:
                representatives[cluster_id] = existing_representatives[cluster_id]
                logger.debug(f"Using existing representative for cluster {cluster_id}")
        
        # For new clusters, select representatives
        for cluster_id, face_indices in final_clusters.items():
            if cluster_id not in representatives:
                # Get all faces in this cluster
                cluster_faces = [all_faces[idx] for idx in face_indices]
                
                # Find best representative
                if cluster_faces:
                    # Choose representative face
                    representative = face_system.select_representative_face(cluster_faces)
                    if representative:
                        representatives[cluster_id] = {
                            'embedding': representative['embedding'],
                            'face_image': representative['face_image'],
                            'original_path': representative['original_path']
                        }
                        logger.debug(f"Selected new representative for cluster {cluster_id}")
        
        # Save processed results
        logger.info("Saving processed results")
        
        # First, copy all images to album
        copied_files = set()
        for face in all_faces:
            orig_filename = os.path.basename(face['original_path'])
            album_path = os.path.join(album_dir, orig_filename)
            if not os.path.exists(album_path):
                shutil.copy2(face['original_path'], album_path)
            copied_files.add(orig_filename)
        
        # Create output directories for each cluster and save faces
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Track source files for each cluster
        cluster_sources = {}
        
        # Initialize cluster sources from existing data
        if existing_clusters:
            for cluster_id_str, info in existing_clusters.items():
                try:
                    cluster_id = int(cluster_id_str)
                    if 'source_files' in info:
                        cluster_sources[cluster_id] = set(info['source_files'])
                except (ValueError, KeyError) as e:
                    logger.error(f"Error processing existing cluster {cluster_id_str}: {e}")
        
        # Process each cluster
        for cluster_id, face_indices in final_clusters.items():
            cluster_dir_path = os.path.join(cluster_dir, f"person_{cluster_id}")
            os.makedirs(cluster_dir_path, exist_ok=True)
            
            # Initialize sources for this cluster if not already
            if cluster_id not in cluster_sources:
                cluster_sources[cluster_id] = set()
            
            # Save representative face
            if cluster_id in representatives:
                rep_face = representatives[cluster_id]
                rep_path = os.path.join(cluster_dir_path, "representative.jpg")
                cv2.imwrite(rep_path, cv2.cvtColor(rep_face['face_image'], cv2.COLOR_RGB2BGR))
                
                # Save embedding
                embedding_path = os.path.join(cluster_dir_path, "representative.npy")
                np.save(embedding_path, rep_face['embedding'])
                
                # Add source file
                orig_filename = os.path.basename(rep_face['original_path'])
                cluster_sources[cluster_id].add(orig_filename)
            
            # Save all faces in this cluster
            cluster_faces = [all_faces[idx] for idx in face_indices]
            for idx, face in enumerate(cluster_faces):
                face_path = os.path.join(cluster_dir_path, f"face_{idx}.jpg")
                cv2.imwrite(face_path, cv2.cvtColor(face['face_image'], cv2.COLOR_RGB2BGR))
                
                # Add source file
                orig_filename = os.path.basename(face['original_path'])
                cluster_sources[cluster_id].add(orig_filename)
            
            # Save sources file for this cluster
            sources_path = os.path.join(cluster_dir_path, "sources.txt")
            with open(sources_path, 'w') as f:
                for source in sorted(cluster_sources[cluster_id]):
                    f.write(f"{source}\n")
        
        # Save updated representatives info
        rep_info = {}
        for cluster_id, rep_face in representatives.items():
            rep_info[str(cluster_id)] = {
                'path': os.path.join(cluster_dir, f"person_{cluster_id}", "representative.jpg"),
                'embedding_path': os.path.join(cluster_dir, f"person_{cluster_id}", "representative.npy"),
                'original_image': os.path.basename(rep_face['original_path']),
                'source_files': list(cluster_sources[cluster_id])
            }
        
        rep_info_path = os.path.join(cluster_dir, "representatives.json")
        with open(rep_info_path, 'w') as f:
            json.dump(rep_info, f, indent=4)
        logger.info(f"Saved representatives info to: {rep_info_path}")
        
        # Update or create a manifest file with processing details
        manifest_path = os.path.join(cluster_dir, "manifest.json")
        manifest = {}
        
        # Load existing manifest if it exists
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            except Exception as e:
                logger.error(f"Error loading manifest: {str(e)}")
        
        # Count matched faces and new clusters
        matched_faces = sum(len(face_indices) for cluster_id, face_indices in final_clusters.items() 
                          if cluster_id in existing_representatives)
        new_faces = len(all_faces) - matched_faces
        existing_cluster_count = len(existing_representatives)
        new_cluster_count = len(final_clusters) - len(existing_representatives)
        
        # Update manifest with latest processing info
        manifest.update({
            "last_processed_at": datetime.now().isoformat(),
            "total_images": manifest.get("total_images", 0) + len(copied_files),
            "total_faces": manifest.get("total_faces", 0) + len(all_faces),
            "unique_people": len(rep_info),
            "latest_batch": {
                "images": len(copied_files),
                "faces": len(all_faces),
                "matched_faces": matched_faces,
                "new_faces": new_faces,
                "existing_clusters": existing_cluster_count,
                "new_clusters": new_cluster_count
            }
        })
        
        # Save updated manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        
        logger.info(f"Processed {len(all_faces)} faces:")
        logger.info(f"- {matched_faces} faces matched to existing clusters")
        logger.info(f"- {new_faces} faces in {new_cluster_count} new clusters")
        logger.info(f"Copied {len(copied_files)} images to album directory")
        
        # Clear temp folder
        clear_temp_folder(temp_album_dir)
        
        return {
            "status": "success",
            "message": "Directory processed successfully",
            "details": {
                "existing_people": existing_cluster_count,
                "new_people": new_cluster_count,
                "total_people": len(rep_info),
                "processed_images": len(copied_files),
                "processed_faces": len(all_faces),
                "matched_faces": matched_faces,
                "new_faces": new_faces
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing temporary album: {str(e)}")
        logger.exception("Detailed error:")
        return {
            "status": "error",
            "message": str(e),
            "details": None
        }