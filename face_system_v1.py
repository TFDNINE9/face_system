import numpy as np
import os
import shutil
import cv2
import logging
import threading
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaceSystemConfig:
    """Centralized configuration for the face recognition system."""
    
    def __init__(self):
        # Face detection configuration
        self.detection = {
            'det_size': (640, 640),
            'batch_size': 16,
            'image_size_limit': 1920
        }
        
        # Similarity and clustering configuration
        self.similarity = {
            'metric': 'cosine',           # cosine or euclidean
            'match_threshold': 0.85,      # For matching to existing clusters  
            'clustering_eps': 0.30,       # For DBSCAN clustering
            'min_samples': 2,             # Minimum samples for a cluster
            'search_threshold': 0.83      # Default threshold for search
        }
        
        # Directory configuration
        self.dirs = {
            'temp_album': 'temp_album',
            'album': 'album',
            'clusters': 'clustered_faces',
            'results': 'face_search_results'
        }


class FaceSystem:
    def __init__(self, config: Optional[FaceSystemConfig] = None):
        try:
            # Use provided config or create default
            self.config = config if config else FaceSystemConfig()
            
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
            
            self.app.prepare(ctx_id=0, det_size=self.config.detection['det_size'])
            logger.info(f"Model preparation completed with providers: {providers}")
            
            # Initialize other parameters
            self.batch_size = self.config.detection['batch_size']
            self.image_size_limit = self.config.detection['image_size_limit']
            self.thread_lock = threading.Lock()
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
            
            # Storage for representative faces
            self.representative_faces = {}
            self.cluster_history = {}

        except Exception as e:
            logger.error(f"Error initializing FaceSystem: {str(e)}")
            logger.exception("Detailed error:")
            raise
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _is_openvino_available(self) -> bool:
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

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two face embeddings.
        Returns a value between 0 and 1, where higher means more similar.
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        norm_emb1 = embedding1 / norm1
        norm_emb2 = embedding2 / norm2
        
        # Use the configured similarity metric
        if self.config.similarity['metric'] == 'cosine':
            # Cosine similarity: dot product of normalized vectors
            similarity = np.dot(norm_emb1, norm_emb2)
            # Rescale from [-1, 1] to [0, 1] range
            similarity = (similarity + 1.0) / 2.0
        else:
            # Euclidean distance: lower means more similar
            distance = np.linalg.norm(norm_emb1 - norm_emb2)
            # Convert to similarity where 1.0 is identical
            similarity = max(0.0, 1.0 - distance)
        
        return similarity

    def select_best_representative(self, faces: List[Dict], existing_representative: Optional[Dict] = None) -> Optional[Dict]:
        """
        Select the best representative face from a cluster.
        
        Args:
            faces: List of face dictionaries
            existing_representative: Existing representative face (optional)
            
        Returns:
            Best face to use as representative
        """
        if not faces and not existing_representative:
            return None
        
        if not faces and existing_representative:
            return existing_representative
        
        # If there's an existing representative, include it in the selection
        all_candidates = faces.copy()
        if existing_representative:
            # Add the existing representative with a slight bonus
            existing_rep_copy = existing_representative.copy()
            existing_rep_copy['bonus_score'] = 0.05  # Slight preference for stability
            all_candidates.append(existing_rep_copy)
        
        # Score candidates based on detection score, size, and quality
        candidate_scores = []
        for face in all_candidates:
            # Get base detection score
            det_score = face.get('det_score', 0.5)
            
            # Calculate relative size score
            max_size = max((f.get('face_size', 0) for f in all_candidates))
            size_score = face.get('face_size', 0) / max_size if max_size > 0 else 0.5
            
            # Add bonus if this is existing representative
            bonus = face.get('bonus_score', 0)
            
            # Calculate total score with weights
            total_score = (0.6 * det_score) + (0.35 * size_score) + bonus
            candidate_scores.append((face, total_score))
        
        # Select face with highest score
        best_face, best_score = max(candidate_scores, key=lambda x: x[1])
        logger.info(f"Selected representative face with score {best_score:.3f}")
        return best_face

    def unified_clustering(self, all_faces: List[Dict], existing_representatives: Optional[Dict] = None) -> Dict[str, List[int]]:
        """
        Process all faces together in a unified manner, integrating with existing clusters.
        
        Args:
            all_faces: List of face dictionaries for new faces
            existing_representatives: Dictionary of existing cluster representatives
            
        Returns:
            Dictionary mapping cluster IDs to lists of face indices
        """
        try:
            # First extract all embeddings
            all_embeddings = []
            face_indices = []
            
            # Add existing representative embeddings
            existing_indices = {}
            next_cluster_id = 0
            
            if existing_representatives:
                for cluster_id_str, rep_data in existing_representatives.items():
                    try:
                        cluster_id = int(cluster_id_str)
                        next_cluster_id = max(next_cluster_id, cluster_id + 1)
                        
                        all_embeddings.append(rep_data['embedding'])
                        face_indices.append(None)  # No direct face index for existing reps
                        existing_indices[len(all_embeddings) - 1] = cluster_id
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Skipping invalid cluster ID {cluster_id_str}: {e}")
            
            # Add new face embeddings
            for i, face in enumerate(all_faces):
                all_embeddings.append(face['embedding'])
                face_indices.append(i)
            
            # Convert to numpy array and normalize
            embeddings_array = np.array(all_embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            normalized_embeddings = embeddings_array / norms
            
            # Run clustering on ALL embeddings together
            clustering = DBSCAN(
                eps=self.config.similarity['clustering_eps'],
                min_samples=self.config.similarity['min_samples'],
                metric=self.config.similarity['metric'],
                n_jobs=-1
            ).fit(normalized_embeddings)
            
            labels = clustering.labels_
            
            # Process results - this now integrates existing and new clusters
            final_clusters = {}
            
            # First process existing representatives to maintain their cluster IDs
            existing_rep_labels = {}  # Maps DBSCAN labels to existing cluster IDs
            
            for i, label in enumerate(labels):
                if i in existing_indices and label != -1:
                    existing_id = existing_indices[i]
                    if label not in existing_rep_labels:
                        existing_rep_labels[label] = existing_id
                    elif existing_rep_labels[label] != existing_id:
                        logger.warning(f"Conflict: DBSCAN grouped representatives from clusters {existing_rep_labels[label]} and {existing_id}")
            
            # Now process all labels to create final clusters
            processed_labels = set()
            
            # First process faces that matched with existing clusters
            for i, label in enumerate(labels):
                if i in existing_indices:
                    continue  # Skip the representative itself
                
                face_idx = face_indices[i]
                if face_idx is None:
                    continue  # Skip representatives
                
                if label in existing_rep_labels:
                    # This face matches an existing cluster
                    cluster_id = existing_rep_labels[label]
                    if cluster_id not in final_clusters:
                        final_clusters[cluster_id] = []
                    final_clusters[cluster_id].append(face_idx)
                    processed_labels.add(i)
            
            # Now process new clusters
            for label_value in set(labels):
                if label_value == -1:
                    continue  # Process noise points separately
                
                if label_value in existing_rep_labels:
                    continue  # Already processed
                
                # This is a new cluster
                cluster_id = next_cluster_id
                next_cluster_id += 1
                
                cluster_faces = []
                for i, label in enumerate(labels):
                    if label == label_value and i not in processed_labels:
                        face_idx = face_indices[i]
                        if face_idx is not None:  # Only add actual faces
                            cluster_faces.append(face_idx)
                            processed_labels.add(i)
                
                if cluster_faces:
                    final_clusters[cluster_id] = cluster_faces
            
            # Finally process noise points as individual clusters
            for i, label in enumerate(labels):
                if label == -1 and i not in processed_labels:
                    face_idx = face_indices[i]
                    if face_idx is not None:  # Only add actual faces
                        cluster_id = next_cluster_id
                        next_cluster_id += 1
                        final_clusters[cluster_id] = [face_idx]
            
            logger.info(f"Unified clustering created/updated {len(final_clusters)} clusters")
            return final_clusters
            
        except Exception as e:
            logger.error(f"Error in unified clustering: {str(e)}")
            logger.exception("Detailed error:")
            return {}

    def save_clusters(self, all_faces: List[Dict], final_clusters: Dict[str, List[int]], 
                     output_dir: str, album_dir: str, existing_representatives: Optional[Dict] = None) -> Dict:
        """
        Save clustered faces and copy original images to album.
        
        Args:
            all_faces: List of face dictionaries
            final_clusters: Dictionary mapping cluster IDs to lists of face indices
            output_dir: Directory to save clusters
            album_dir: Directory to copy original images
            existing_representatives: Dictionary of existing representative faces
            
        Returns:
            Dictionary with processing statistics
        """
        try:
            output_dir = os.path.abspath(output_dir)
            album_dir = os.path.abspath(album_dir)
            
            # Ensure directories exist
            os.makedirs(album_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Load existing cluster sources if available
            cluster_sources = {}
            existing_reps = {}
            
            # Load existing representatives if provided
            if existing_representatives:
                existing_reps = existing_representatives
                
                # Initialize sources from existing data
                existing_rep_file = os.path.join(output_dir, "representatives.json")
                if os.path.exists(existing_rep_file):
                    try:
                        with open(existing_rep_file, 'r') as f:
                            rep_info = json.load(f)
                        
                        for cluster_id_str, info in rep_info.items():
                            try:
                                cluster_id = int(cluster_id_str)
                                if 'source_files' in info:
                                    cluster_sources[cluster_id] = set(info['source_files'])
                            except (ValueError, KeyError) as e:
                                logger.error(f"Error processing existing cluster {cluster_id_str}: {e}")
                    except Exception as e:
                        logger.error(f"Error loading existing representatives: {str(e)}")
            
            # Track copied files and updated clusters
            copied_files = set()
            updated_clusters = set()
            new_clusters = set()
            representatives = {}
            
            # Process each cluster
            for cluster_id, face_indices in final_clusters.items():
                try:
                    cluster_id = int(cluster_id)  # Ensure cluster_id is an integer
                    cluster_dir_path = os.path.join(output_dir, f"person_{cluster_id}")
                    os.makedirs(cluster_dir_path, exist_ok=True)
                    
                    # Get all faces in this cluster
                    cluster_faces = [all_faces[idx] for idx in face_indices]
                    
                    # Initialize sources for this cluster if not already
                    if cluster_id not in cluster_sources:
                        cluster_sources[cluster_id] = set()
                        new_clusters.add(cluster_id)
                    else:
                        updated_clusters.add(cluster_id)
                    
                    # Determine if this is an existing cluster
                    existing_representative = None
                    if str(cluster_id) in existing_reps:
                        existing_representative = existing_reps[str(cluster_id)]
                    
                    # Select representative face for this cluster
                    representative = self.select_best_representative(cluster_faces, existing_representative)
                    
                    if representative:
                        # Save representative face
                        rep_path = os.path.join(cluster_dir_path, "representative.jpg")
                        cv2.imwrite(rep_path, cv2.cvtColor(representative['face_image'], cv2.COLOR_RGB2BGR))
                        
                        # Save embedding
                        embedding_path = os.path.join(cluster_dir_path, "representative.npy")
                        np.save(embedding_path, representative['embedding'])
                        
                        # Add to representatives dictionary
                        representatives[str(cluster_id)] = {
                            'embedding': representative['embedding'],
                            'face_image': representative['face_image'],
                            'original_path': representative['original_path'],
                            'path': rep_path,
                            'embedding_path': embedding_path
                        }
                        
                        # Add source file for representative
                        orig_filename = os.path.basename(representative['original_path'])
                        cluster_sources[cluster_id].add(orig_filename)
                        
                        # Copy original to album if needed
                        album_path = os.path.join(album_dir, orig_filename)
                        if not os.path.exists(album_path) and os.path.exists(representative['original_path']):
                            shutil.copy2(representative['original_path'], album_path)
                            copied_files.add(orig_filename)
                    
                    # Save all faces in this cluster
                    for idx, face in enumerate(cluster_faces):
                        # Save face image
                        face_path = os.path.join(cluster_dir_path, f"face_{idx}.jpg")
                        cv2.imwrite(face_path, cv2.cvtColor(face['face_image'], cv2.COLOR_RGB2BGR))
                        
                        # Add source file
                        orig_filename = os.path.basename(face['original_path'])
                        cluster_sources[cluster_id].add(orig_filename)
                        
                        # Copy original to album if needed
                        album_path = os.path.join(album_dir, orig_filename)
                        if not os.path.exists(album_path):
                            shutil.copy2(face['original_path'], album_path)
                            copied_files.add(orig_filename)
                    
                    # Save sources file for this cluster
                    sources_path = os.path.join(cluster_dir_path, "sources.txt")
                    with open(sources_path, 'w') as f:
                        for source in sorted(cluster_sources[cluster_id]):
                            f.write(f"{source}\n")
                            
                except Exception as e:
                    logger.error(f"Error processing cluster {cluster_id}: {str(e)}")
                    continue
            
            # Save updated representatives info
            rep_info = {}
            for cluster_id, rep_data in representatives.items():
                rep_info[cluster_id] = {
                    'path': os.path.join(output_dir, f"person_{cluster_id}", "representative.jpg"),
                    'embedding_path': os.path.join(output_dir, f"person_{cluster_id}", "representative.npy"),
                    'original_image': os.path.basename(rep_data['original_path']),
                    'source_files': list(cluster_sources[int(cluster_id)])
                }
            
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
            
            # Prepare latest batch statistics
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            batch_stats = {
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "images": len(copied_files),
                "faces": len(all_faces),
                "updated_clusters": len(updated_clusters),
                "new_clusters": len(new_clusters),
                "total_clusters": len(rep_info)
            }
            
            # Update manifest with latest processing info
            manifest.update({
                "last_processed_at": datetime.now().isoformat(),
                "total_images": manifest.get("total_images", 0) + len(copied_files),
                "total_faces": manifest.get("total_faces", 0) + len(all_faces),
                "unique_people": len(rep_info),
                "latest_batch": batch_stats
            })
            
            # Add processing history
            if "processing_history" not in manifest:
                manifest["processing_history"] = []
            
            manifest["processing_history"].append(batch_stats)
            
            # Save updated manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=4)
            
            logger.info(f"Processed {len(all_faces)} faces into {len(final_clusters)} clusters")
            logger.info(f"Updated {len(updated_clusters)} existing clusters and created {len(new_clusters)} new clusters")
            logger.info(f"Copied {len(copied_files)} images to album directory")
            
            # Save cluster history
            self.cluster_history = {
                "updated_clusters": list(updated_clusters),
                "new_clusters": list(new_clusters),
                "total_clusters": len(rep_info)
            }
            
            # Return processing statistics
            return {
                "existing_clusters": len(updated_clusters),
                "new_clusters": len(new_clusters),
                "total_clusters": len(rep_info),
                "processed_images": len(copied_files),
                "processed_faces": len(all_faces)
            }

        except Exception as e:
            logger.error(f"Error saving clusters: {str(e)}")
            logger.exception("Detailed error:")
            raise

    def load_representative_faces(self, clustered_dir: str) -> bool:
        """
        Load representative faces with their embeddings.
        
        Args:
            clustered_dir: Directory containing cluster data
            
        Returns:
            Boolean indicating success
        """
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
                try:
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
                    
                    # Get original image path from info
                    original_path = info.get('original_image', os.path.basename(full_img_path))
                    
                    # Store the face info
                    self.representative_faces[label] = {
                        'embedding': embedding,
                        'face_image': img_rgb,
                        'original_path': original_path,
                        'path': full_img_path,
                        'embedding_path': full_emb_path
                    }
                    
                    if 'source_files' in info:
                        self.representative_faces[label]['source_files'] = info['source_files']
                
                except Exception as e:
                    logger.error(f"Error loading representative for cluster {label}: {str(e)}")
                    continue

            num_loaded = len(self.representative_faces)
            logger.info(f"Successfully loaded {num_loaded} representative faces")
            return num_loaded > 0

        except Exception as e:
            logger.error(f"Error loading representative faces: {str(e)}")
            logger.exception("Detailed error:")
            return False

    def find_matching_faces(self, query_image: np.ndarray, clustered_dir: str, album_dir: str, 
                           similarity_threshold: Optional[float] = None) -> Dict:
        """
        Find matches using representative faces and return source files.
        
        Args:
            query_image: Image containing a face to search for
            clustered_dir: Directory where face clusters are stored
            album_dir: Directory containing original images
            similarity_threshold: Threshold for considering a match (0-1)
            
        Returns:
            Dictionary with match results
        """
        try:
            # Use default threshold if not specified
            if similarity_threshold is None:
                similarity_threshold = self.config.similarity['search_threshold']
            
            # Ensure representative faces are loaded
            if not self.representative_faces:
                if not self.load_representative_faces(clustered_dir):
                    return {
                        'status': 'error',
                        'message': 'No representative faces loaded',
                        'matches': []
                    }

            # Get query face
            faces = self.app.get(query_image)
            if not faces:
                return {
                    'status': 'error',
                    'message': 'No face detected in query image',
                    'matches': []
                }
            
            # Use face with highest detection score
            query_face = max(faces, key=lambda x: x.det_score)
            query_embedding = query_face.embedding
            matches = []
            
            # Compare with representative faces
            for label, rep_face in self.representative_faces.items():
                # Calculate similarity using the consistent method
                similarity = self.calculate_similarity(query_embedding, rep_face['embedding'])
                
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
                        source = source.strip()
                        if not source:
                            continue
                            
                        album_path = os.path.join(album_dir, source)
                        if os.path.exists(album_path):
                            verified_sources.append({
                                'filename': source,
                                'face_url': f"{album_path}"  # You'll need to adjust this URL format
                            })
                        else:
                            logger.warning(f"Source file not found in album: {source}")
                    
                    matches.append({
                        'person_id': int(label),
                        'similarity': float(similarity * 100),
                        'representative_url': rep_face['path'],
                        'source_files': verified_sources
                    })
            
            # Sort matches by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'status': 'success',
                'message': f"Found {len(matches)} matching persons",
                'matches': matches
            }
            
        except Exception as e:
            logger.error(f"Error in face matching: {str(e)}")
            logger.exception("Detailed error:")
            return {
                'status': 'error',
                'message': str(e),
                'matches': []
            }


def clear_temp_folder(temp_dir: str) -> int:
    """
    Clear all files from temporary directory after processing.
    
    Args:
        temp_dir: Directory to clear
        
    Returns:
        Number of files cleared
    """
    try:
        cleared_count = 0
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    cleared_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    cleared_count += 1
            logger.info(f"Cleared {cleared_count} files from temporary directory: {temp_dir}")
        else:
            logger.warning(f"Temporary directory does not exist: {temp_dir}")
        return cleared_count
    except Exception as e:
        logger.error(f"Error clearing temporary directory: {str(e)}")
        logger.exception("Detailed error:")
        return 0
    
    
def process_temp_album(temp_album_dir: str, cluster_dir: str, album_dir: str, batch_size: int = 16) -> Dict:
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
        face_system = FaceSystem()
        
        # Load existing representatives if available
        existing_representatives = {}
        
        if os.path.exists(os.path.join(cluster_dir, "representatives.json")):
            logger.info("Loading existing representative faces")
            if face_system.load_representative_faces(cluster_dir):
                existing_representatives = face_system.representative_faces
                logger.info(f"Loaded {len(existing_representatives)} existing representative faces")
        
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
                    "new_clusters": 0,
                    "total_clusters": len(existing_representatives)
                }
            }
        
        # Use the unified clustering approach to integrate with existing clusters
        final_clusters = face_system.unified_clustering(all_faces, existing_representatives)
        
        # Save the updated clusters
        result = face_system.save_clusters(
            all_faces=all_faces,
            final_clusters=final_clusters,
            output_dir=cluster_dir,
            album_dir=album_dir,
            existing_representatives=existing_representatives
        )
        
        # Clear temp folder after successful processing
        clear_temp_folder(temp_album_dir)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return comprehensive results
        return {
            "status": "success",
            "message": "Directory processed successfully",
            "details": {
                "processing_time_seconds": processing_time,
                "existing_clusters": result["existing_clusters"],
                "new_clusters": result["new_clusters"],
                "total_clusters": result["total_clusters"],
                "processed_images": result["processed_images"],
                "processed_faces": result["processed_faces"]
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