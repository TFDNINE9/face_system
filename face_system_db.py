import numpy as np
import os
import cv2
import logging
import threading
import uuid
import pyodbc  # For SQL Server connection
from datetime import datetime
from typing import Optional
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
            'base_storage': 'storage',
            'temp': 'temp',
            'faces': 'faces',
            'embeddings': 'embeddings',
            'thumbnails': 'thumbnails'
        }
        
        # Database configuration
        self.db = {
            'server': '100.102.194.74',
            'database': 'TopDb2',
            'username': 'usrtop',
            'password': 'TTTop@2o25',
            'driver': '{ODBC Driver 17 for SQL Server}'
        }


class DatabaseFaceSystem:
    def __init__(self, connection_string=None, config: Optional[FaceSystemConfig] = None):
        """
        Initialize face system with database connection.
        
        Args:
            connection_string: Database connection string (optional)
            config: Configuration object
        """
        try:
            # Use provided config or create default
            self.config = config if config else FaceSystemConfig()
            
            # Set up database connection
            if connection_string:
                self.connection_string = connection_string
            else:
                self.connection_string = (
                    f"DRIVER={self.config.db['driver']};"
                    f"SERVER={self.config.db['server']};"
                    f"DATABASE={self.config.db['database']};"
                    f"UID={self.config.db['username']};"
                    f"PWD={self.config.db['password']}"
                )
            
            # Test database connection
            self._test_db_connection()
            
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
            
            # Ensure storage directories exist
            for dir_key, dir_name in self.config.dirs.items():
                os.makedirs(dir_name, exist_ok=True)
                logger.info(f"Ensured directory exists: {dir_name}")

        except Exception as e:
            logger.error(f"Error initializing DatabaseFaceSystem: {str(e)}")
            logger.exception("Detailed error:")
            raise
    
    def _test_db_connection(self):
        """Test the database connection."""
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
            logger.info(f"Successfully connected to database: {version}")
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def get_db_connection(self):
        """Get a database connection."""
        return pyodbc.connect(self.connection_string)
    
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
    
    def _get_status_id(self, cursor, status_code, status_type='status_types'):
        """Get a status ID from its code."""
        cursor.execute(
            f"SELECT status_id FROM {status_type} WHERE status_code = ?",
            (status_code,)
        )
        result = cursor.fetchone()
        if result:
            return result[0]  # First column contains the ID
        raise ValueError(f"Unknown status code: {status_code}")
    
    def _get_job_type_id(self, cursor, job_code):
        """Get a job type ID from its code."""
        cursor.execute(
            "SELECT job_type_id FROM job_types WHERE job_code = ?",
            (job_code,)
        )
        result = cursor.fetchone()
        if result:
            return result[0]  # First column contains the ID
        raise ValueError(f"Unknown job code: {job_code}")
    
    def _create_processing_job(self, event_id, job_type):
        """Create a new processing job record."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            job_type_id = self._get_job_type_id(cursor, job_type)
            status_id = self._get_status_id(cursor, 'queued')
            job_id = str(uuid.uuid4())
            
            cursor.execute(
                """
                INSERT INTO processing_jobs 
                (job_id, event_id, job_type_id, status_id, created_at, started_at)
                VALUES (?, ?, ?, ?, SYSUTCDATETIME(), SYSUTCDATETIME())
                """,
                (job_id, event_id, job_type_id, status_id)
            )
            conn.commit()
            
            # Update event status to processing
            event_status_id = self._get_status_id(cursor, 'processing', 'event_status_types')
            cursor.execute(
                "UPDATE events SET status_id = ?, updated_at = SYSUTCDATETIME() WHERE event_id = ?",
                (event_status_id, event_id)
            )
            conn.commit()
            
            cursor.close()
            conn.close()
            
            return job_id
        except Exception as e:
            logger.error(f"Error creating processing job: {str(e)}")
            raise
    
    def _update_job_status(self, job_id, status_code, error_message=None):
        """Update a job's status."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            status_id = self._get_status_id(cursor, status_code)
            
            if status_code == 'completed':
                cursor.execute(
                    """
                    UPDATE processing_jobs 
                    SET status_id = ?, completed_at = SYSUTCDATETIME()
                    WHERE job_id = ?
                    """,
                    (status_id, job_id)
                )
            elif status_code == 'failed':
                cursor.execute(
                    """
                    UPDATE processing_jobs 
                    SET status_id = ?, completed_at = SYSUTCDATETIME(), error_message = ?
                    WHERE job_id = ?
                    """,
                    (status_id, error_message or "Unknown error", job_id)
                )
            else:
                cursor.execute(
                    "UPDATE processing_jobs SET status_id = ? WHERE job_id = ?",
                    (status_id, job_id)
                )
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating job status: {str(e)}")
            raise
    
    def _complete_processing_job(self, job_id):
        """Mark a processing job as completed."""
        self._update_job_status(job_id, 'completed')
    
    def _fail_processing_job(self, job_id, error_message):
        """Mark a processing job as failed."""
        self._update_job_status(job_id, 'failed', error_message)
    
    def _get_customer_event_path(self, event_id):
        """Get the storage path for an event."""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT e.event_id, e.name AS event_name, c.customer_id, c.name AS customer_name
            FROM events e
            JOIN customers c ON e.customer_id = c.customer_id
            WHERE e.event_id = ?
            """,
            (event_id,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise ValueError(f"Event not found: {event_id}")
        
        customer_id = result[2]
        event_id = result[0]
        
        base_path = os.path.join(
            self.config.dirs['base_storage'],
            'customers',
            str(customer_id),
            'events',
            str(event_id)
        )
        
        return base_path
    
    def _get_storage_path(self, event_id, subdirectory):
        """Get a storage path for a specific subdirectory of an event."""
        base_path = self._get_customer_event_path(event_id)
        directory = os.path.join(base_path, subdirectory)
        os.makedirs(directory, exist_ok=True)
        return directory
    
    def process_event_images(self, event_id):
        """
        Process all unprocessed images for an event.
        
        Args:
            event_id: UUID of the event to process
            
        Returns:
            Dictionary with processing status and results
        """
        try:
            # Create a processing job
            job_id = self._create_processing_job(event_id, 'detect')
            
            # Get unprocessed images for this event
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT image_id, storage_path 
                FROM images 
                WHERE event_id = ? AND processed = 0
                """,
                (event_id,)
            )
            
            images = []
            row = cursor.fetchone()
            while row:
                images.append({
                    'image_id': row[0],
                    'storage_path': row[1]
                })
                row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if not images:
                logger.info(f"No unprocessed images found for event {event_id}")
                self._complete_processing_job(job_id)
                return {
                    "status": "success",
                    "message": "No images to process",
                    "details": {"job_id": job_id}
                }
            
            # Process images in batches
            all_faces = []
            processed_image_ids = []
            
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                image_paths = [img['storage_path'] for img in batch]
                image_ids = [img['image_id'] for img in batch]
                
                # Process batch
                batch_faces = self._process_image_batch(image_ids, image_paths, event_id)
                all_faces.extend(batch_faces)
                processed_image_ids.extend(image_ids)
                
                # Mark images as processed
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                for img_id in image_ids:
                    cursor.execute(
                        "UPDATE images SET processed = 1 WHERE image_id = ?",
                        (img_id,)
                    )
                
                conn.commit()
                cursor.close()
                conn.close()
            
            # Complete the detection job
            self._complete_processing_job(job_id)
            
            # If any faces were detected, start clustering
            if all_faces:
                cluster_job_id = self.cluster_event_faces(event_id)
                
            return {
                "status": "success",
                "message": f"Processed {len(images)} images, detected {len(all_faces)} faces",
                "details": {
                    "job_id": job_id,
                    "processed_images": len(images),
                    "detected_faces": len(all_faces)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing images for event {event_id}: {str(e)}")
            try:
                self._fail_processing_job(job_id, str(e))
            except:
                pass
            return {
                "status": "error",
                "message": str(e),
                "details": None
            }
        
    def _process_image_batch(self, image_ids, image_paths, event_id):
        all_faces = []
        faces_dir = self._get_storage_path(event_id, 'faces')
        embeddings_dir = self._get_storage_path(event_id, 'embeddings')
        
        # Ensure directories exist
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Step 1: Process images in parallel using ThreadPoolExecutor
        def process_single_image(img_id, img_path):
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not read image: {img_path}")
                    return []
                
                # Resize if needed
                h, w = img.shape[:2]
                if max(h, w) > self.image_size_limit:
                    scale = self.image_size_limit / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
                # Convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                with self.thread_lock:
                    faces = self.app.get(img_rgb)
                
                if not isinstance(faces, list) or not faces:
                    return []
                
                # Process detected faces
                image_faces = []
                for face_idx, face in enumerate(faces):
                    try:
                        # Convert numpy types to Python native types
                        det_score = float(face.det_score)
                        embedding = face.embedding
                        bbox = face.bbox
                        kps = face.kps
                        
                        face_img = face_align.norm_crop(img_rgb, kps)
                        face_id = str(uuid.uuid4())
                        
                        face_filename = f"{face_id}.jpg"
                        face_path = os.path.join(faces_dir, face_filename)
                        embedding_filename = f"{face_id}.npy"
                        embedding_path = os.path.join(embeddings_dir, embedding_filename)
                        
                        # Save face image and embedding
                        cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                        np.save(embedding_path, embedding)
                        
                        # Calculate face size with Python float
                        face_size = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                        
                        # Store face data
                        image_faces.append({
                            'face_id': face_id,
                            'image_id': img_id,
                            'face_index': face_idx,
                            'detection_score': det_score,
                            'face_size': face_size,
                            'face_path': face_path,
                            'embedding_path': embedding_path
                        })
                    except Exception as face_error:
                        logger.error(f"Error processing face {face_idx} in image {img_id}: {face_error}")
                
                return image_faces
            except Exception as img_error:
                logger.error(f"Error processing image {img_path}: {img_error}")
                return []
        
        # Process images in parallel with at most 4 threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(process_single_image, img_id, img_path) 
                    for img_id, img_path in zip(image_ids, image_paths)]
            
            # Collect results as they complete
            for future in futures:
                image_faces = future.result()
                all_faces.extend(image_faces)
        
        # Step 2: Do a single bulk insert into the database
        if all_faces:
            try:
                # Extract just the fields needed for database insert
                face_records = [(
                    face['face_id'],
                    face['image_id'],
                    face['face_index'],
                    face['detection_score'],
                    face['face_size'],
                    face['face_path'],
                    face['embedding_path']
                ) for face in all_faces]
                
                # Open a single connection for all inserts
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                # Use a single transaction for all inserts
                try:
                    # Use larger batch size for better performance
                    batch_size = 500
                    
                    for i in range(0, len(face_records), batch_size):
                        batch = face_records[i:i + batch_size]
                        cursor.executemany(
                            """
                            INSERT INTO faces 
                            (face_id, image_id, face_index, detection_score, face_size, face_path, embedding_path)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            batch
                        )
                    
                    # Commit once at the end
                    conn.commit()
                    logger.info(f"Successfully inserted {len(face_records)} face records")
                except Exception as insert_error:
                    conn.rollback()
                    logger.error(f"DB insert error: {insert_error}")
                    # Try with a single insert to see detailed error
                    if face_records:
                        try:
                            cursor.execute(
                                """
                                INSERT INTO faces 
                                (face_id, image_id, face_index, detection_score, face_size, face_path, embedding_path)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                face_records[0]
                            )
                            conn.commit()
                        except Exception as single_error:
                            logger.error(f"Single record insert error: {single_error}")
                            logger.error(f"Record values: {face_records[0]}")
                            logger.error(f"Value types: {[type(v).__name__ for v in face_records[0]]}")
                finally:
                    cursor.close()
                    conn.close()
            except Exception as db_error:
                logger.error(f"Database connection error: {db_error}")
        
        logger.info(f"Processed {len(image_paths)} images, found {len(all_faces)} faces")
        return all_faces
        
    def cluster_event_faces(self, event_id):
        """
        Cluster all unclustered faces for an event.
        
        Args:
            event_id: UUID of the event
            
        Returns:
            Job ID for the clustering task
        """
        try:
            # Create a clustering job
            job_id = self._create_processing_job(event_id, 'cluster')
            
            # Run clustering in a background thread to not block
            threading.Thread(
                target=self._perform_clustering,
                args=(event_id, job_id),
                daemon=True
            ).start()
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting clustering for event {event_id}: {str(e)}")
            try:
                self._fail_processing_job(job_id, str(e))
            except:
                pass
            raise
    
    def _perform_clustering(self, event_id, job_id):
        """
        Perform the actual clustering (runs in background thread).
        
        Args:
            event_id: UUID of the event
            job_id: UUID of the processing job
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get all unclustered faces
            cursor.execute(
                """
                SELECT f.face_id, f.detection_score, f.face_size, f.face_path, f.embedding_path, 
                       i.original_filename
                FROM faces f
                JOIN images i ON f.image_id = i.image_id
                WHERE i.event_id = ? AND f.cluster_id IS NULL
                """,
                (event_id,)
            )
            
            new_faces = []
            row = cursor.fetchone()
            while row:
                new_faces.append({
                    'face_id': row[0],
                    'detection_score': row[1],
                    'face_size': row[2],
                    'face_path': row[3],
                    'embedding_path': row[4],
                    'original_filename': row[5]
                })
                row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if not new_faces:
                logger.info(f"No unclustered faces found for event {event_id}")
                self._complete_processing_job(job_id)
                return
            
            # Get existing clusters and their representatives
            existing_clusters = self._get_existing_clusters(event_id)
            
            # Load face embeddings
            face_data = []
            for face in new_faces:
                try:
                    embedding = np.load(face['embedding_path'])
                    face_image = cv2.imread(face['face_path'])
                    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    
                    face_data.append({
                        'face_id': face['face_id'],
                        'embedding': embedding,
                        'face_image': face_image_rgb,
                        'det_score': face['detection_score'],
                        'face_size': face['face_size'],
                        'face_path': face['face_path'],
                        'embedding_path': face['embedding_path'],
                        'original_filename': face['original_filename']
                    })
                except Exception as e:
                    logger.error(f"Error loading embedding for face {face['face_id']}: {str(e)}")
                    continue
            
            # Perform clustering using unified approach
            cluster_results = self._unified_clustering_db(face_data, existing_clusters, event_id)
            
            # Update event status if the clustering is complete
            self._update_event_after_clustering(event_id)
            
            # Mark job as completed
            self._complete_processing_job(job_id)
            
            logger.info(f"Clustering completed for event {event_id}: {len(cluster_results['clusters'])} clusters")
            
        except Exception as e:
            logger.error(f"Error during clustering for event {event_id}: {str(e)}")
            self._fail_processing_job(job_id, str(e))
    
    def _get_existing_clusters(self, event_id):
        """
        Get existing clusters for an event.
        
        Args:
            event_id: UUID of the event
            
        Returns:
            Dictionary of existing clusters with their representative faces
        """
        existing_clusters = {}
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT c.cluster_id, f.face_id, f.embedding_path, f.face_path
                FROM clusters c
                JOIN faces f ON c.representative_face_id = f.face_id
                WHERE c.event_id = ?
                """,
                (event_id,)
            )
            
            row = cursor.fetchone()
            while row:
                cluster_id = row[0]
                face_id = row[1]
                embedding_path = row[2]
                face_path = row[3]
                
                try:
                    # Load the embedding and face image
                    embedding = np.load(embedding_path)
                    face_image = cv2.imread(face_path)
                    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    
                    existing_clusters[cluster_id] = {
                        'face_id': face_id,
                        'embedding': embedding,
                        'face_image': face_image_rgb,
                        'embedding_path': embedding_path,
                        'face_path': face_path
                    }
                except Exception as e:
                    logger.error(f"Error loading representative for cluster {cluster_id}: {str(e)}")
                
                row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return existing_clusters
            
        except Exception as e:
            logger.error(f"Error getting existing clusters for event {event_id}: {str(e)}")
            return {}
    
    def _unified_clustering_db(self, new_faces, existing_clusters, event_id):
        """
        Perform unified clustering with database integration.
        
        Args:
            new_faces: List of new face dictionaries
            existing_clusters: Dictionary of existing clusters
            event_id: UUID of the event
            
        Returns:
            Dictionary with clustering results
        """
        try:
            # Extract embeddings for clustering
            all_embeddings = []
            face_indices = []
            
            # Add existing representative embeddings
            cluster_id_mapping = {}
            rep_idx_to_cluster = {}
            
            for idx, (cluster_id, rep_data) in enumerate(existing_clusters.items()):
                all_embeddings.append(rep_data['embedding'])
                face_indices.append(None)  # No direct face index
                rep_idx_to_cluster[len(all_embeddings) - 1] = cluster_id
            
            # Add new face embeddings
            for i, face in enumerate(new_faces):
                all_embeddings.append(face['embedding'])
                face_indices.append(i)
            
            # Normalize embeddings for clustering
            embeddings_array = np.array(all_embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            normalized_embeddings = embeddings_array / norms
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=self.config.similarity['clustering_eps'],
                min_samples=self.config.similarity['min_samples'],
                metric=self.config.similarity['metric'],
                n_jobs=-1
            ).fit(normalized_embeddings)
            
            labels = clustering.labels_
            
            # Process results to create final clusters
            final_clusters = {}
            processed_indices = set()
            existing_cluster_labels = {}
            
            # First, process existing representatives to maintain their cluster IDs
            for i, label in enumerate(labels):
                if i in rep_idx_to_cluster and label != -1:
                    cluster_id = rep_idx_to_cluster[i]
                    if label not in existing_cluster_labels:
                        existing_cluster_labels[label] = cluster_id
                    else:
                        logger.warning(f"Conflict: DBSCAN grouped representatives from clusters {existing_cluster_labels[label]} and {cluster_id}")
            
            # Process faces that matched with existing clusters
            for i, label in enumerate(labels):
                if i in rep_idx_to_cluster:
                    continue  # Skip representatives themselves
                
                if i >= len(face_indices) or face_indices[i] is None:
                    continue  # Skip invalid indices
                
                face_idx = face_indices[i]
                
                if label in existing_cluster_labels:
                    # This face matches an existing cluster
                    cluster_id = existing_cluster_labels[label]
                    if cluster_id not in final_clusters:
                        final_clusters[cluster_id] = []
                    final_clusters[cluster_id].append(face_idx)
                    processed_indices.add(i)
            
            # Generate new cluster IDs for new clusters
            next_cluster_id = str(uuid.uuid4())
            
            # Process new clusters
            for label_value in set(labels):
                if label_value == -1 or label_value in existing_cluster_labels:
                    continue  # Already processed or noise
                
                # Create a new cluster
                cluster_id = str(uuid.uuid4())
                cluster_id_mapping[label_value] = cluster_id
                final_clusters[cluster_id] = []
                
                for i, label in enumerate(labels):
                    if label == label_value and i not in processed_indices:
                        if i >= len(face_indices) or face_indices[i] is None:
                            continue  # Skip invalid indices
                        
                        face_idx = face_indices[i]
                        final_clusters[cluster_id].append(face_idx)
                        processed_indices.add(i)
            
            # Process noise points (create individual clusters)
            for i, label in enumerate(labels):
                if label == -1 and i not in processed_indices:
                    if i >= len(face_indices) or face_indices[i] is None:
                        continue  # Skip invalid indices
                    
                    face_idx = face_indices[i]
                    cluster_id = str(uuid.uuid4())
                    final_clusters[cluster_id] = [face_idx]
            
            # Save cluster results to database
            cluster_info = self._save_clusters_to_db(final_clusters, new_faces, existing_clusters, event_id)
            
            return {
                "status": "success",
                "clusters": cluster_info,
                "total_clusters": len(final_clusters),
                "processed_faces": len(new_faces)
            }
            
        except Exception as e:
            logger.error(f"Error in unified clustering: {str(e)}")
            raise
    
    def _save_clusters_to_db(self, final_clusters, new_faces, existing_clusters, event_id):
        """
        Save clustering results to the database.
        
        Args:
            final_clusters: Dictionary mapping cluster IDs to lists of face indices
            new_faces: List of face dictionaries
            existing_clusters: Dictionary of existing clusters
            event_id: UUID of the event
            
        Returns:
            List of cluster info dictionaries
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cluster_info = []
            
            # Process each cluster
            for cluster_id, face_indices in final_clusters.items():
                if not face_indices:
                    continue
                
                # Get faces in this cluster
                cluster_faces = [new_faces[idx] for idx in face_indices]
                
                # Determine if this is an existing cluster
                is_existing = cluster_id in existing_clusters
                
                # Select or create representative face
                if is_existing:
                    rep_face_id = existing_clusters[cluster_id]['face_id']
                    rep_face_path = existing_clusters[cluster_id]['face_path']
                else:
                    # Create a new cluster
                    best_face = self._select_best_representative(cluster_faces)
                    rep_face_id = best_face['face_id']
                    rep_face_path = best_face['face_path']
                    
                    # Insert new cluster record
                    cursor.execute(
                        """
                        INSERT INTO clusters
                        (cluster_id, event_id, representative_face_id, total_faces, created_at, updated_at)
                        VALUES (?, ?, ?, ?, SYSUTCDATETIME(), SYSUTCDATETIME())
                        """,
                        (cluster_id, event_id, rep_face_id, len(cluster_faces))
                    )
                    conn.commit()
                
                # If existing cluster, update total_faces count
                if is_existing:
                    cursor.execute(
                        """
                        UPDATE clusters
                        SET total_faces = total_faces + ?, updated_at = SYSUTCDATETIME()
                        WHERE cluster_id = ?
                        """,
                        (len(cluster_faces), cluster_id)
                    )
                    conn.commit()
                
                # Associate faces with this cluster
                for face in cluster_faces:
                    face_id = face['face_id']
                    
                    # Calculate similarity to representative
                    if is_existing:
                        rep_embedding = existing_clusters[cluster_id]['embedding']
                    else:
                        rep_embedding = best_face['embedding']
                    
                    similarity = self._calculate_similarity(face['embedding'], rep_embedding)
                    
                    # Update face record with cluster_id
                    cursor.execute(
                        "UPDATE faces SET cluster_id = ? WHERE face_id = ?",
                        (cluster_id, face_id)
                    )
                    
                    # Insert cluster_faces association
                    cursor.execute(
                        """
                        INSERT INTO cluster_faces (cluster_id, face_id, similarity_score)
                        VALUES (?, ?, ?)
                        """,
                        (cluster_id, face_id, similarity)
                    )
                
                conn.commit()
                
                # Add to cluster info results
                cluster_info.append({
                    'cluster_id': cluster_id,
                    'representative_id': rep_face_id,
                    'representative_path': rep_face_path,
                    'total_faces': len(cluster_faces) + (existing_clusters[cluster_id]['total_faces'] if is_existing else 0),
                    'is_new': not is_existing
                })
            
            cursor.close()
            conn.close()
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Error saving clusters to database: {str(e)}")
            raise
    
    def _select_best_representative(self, faces):
        """
        Select the best representative face from a cluster.
        
        Args:
            faces: List of face dictionaries
            
        Returns:
            Dictionary with the best face
        """
        if not faces:
            return None
        
        # Score candidates based on detection score and size
        candidate_scores = []
        for face in faces:
            det_score = face.get('det_score', 0.5)
            
            # Calculate relative size score
            max_size = max((f.get('face_size', 0) for f in faces))
            size_score = face.get('face_size', 0) / max_size if max_size > 0 else 0.5
            
            # Calculate total score with weights
            total_score = (0.6 * det_score) + (0.4 * size_score)
            candidate_scores.append((face, total_score))
        
        # Select face with highest score
        best_face, best_score = max(candidate_scores, key=lambda x: x[1])
        logger.info(f"Selected representative face with score {best_score:.3f}")
        return best_face
    
    def _calculate_similarity(self, embedding1, embedding2):
        """
        Calculate similarity between two face embeddings.
        
        Args:
            embedding1: First embedding array
            embedding2: Second embedding array
            
        Returns:
            Similarity score from 0 to 1
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
    
    def _update_event_after_clustering(self, event_id):
        """
        Update event status after clustering is complete.
        
        Args:
            event_id: UUID of the event
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Check if all faces are clustered
            cursor.execute(
                """
                SELECT COUNT(*) FROM faces f
                JOIN images i ON f.image_id = i.image_id
                WHERE i.event_id = ? AND f.cluster_id IS NULL
                """,
                (event_id,)
            )
            
            unclustered_count = cursor.fetchone()[0]
            
            if unclustered_count == 0:
                # All faces are clustered, mark event as completed
                status_id = self._get_status_id(cursor, 'completed', 'event_status_types')
                cursor.execute(
                    "UPDATE events SET status_id = ?, updated_at = SYSUTCDATETIME() WHERE event_id = ?",
                    (status_id, event_id)
                )
                conn.commit()
                logger.info(f"Event {event_id} marked as completed")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating event status: {str(e)}")
    
    def search_face(self, query_image_path, event_id, similarity_threshold=None):
        """
        Search for matching faces within an event.
        
        Args:
            query_image_path: Path to query image
            event_id: UUID of the event to search in
            similarity_threshold: Similarity threshold (0-1)
            
        Returns:
            Dictionary with search results
        """
        try:
            # Use default threshold if not specified
            if similarity_threshold is None:
                similarity_threshold = self.config.similarity['search_threshold']
            
            # Create a search job
            job_id = self._create_processing_job(event_id, 'search')
            
            # Read the query image
            img = cv2.imread(query_image_path)
            if img is None:
                self._fail_processing_job(job_id, "Could not read query image")
                return {
                    "status": "error",
                    "message": "Could not read query image",
                    "matches": []
                }
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            with self.thread_lock:
                faces = self.app.get(img_rgb)
            
            if not faces:
                self._fail_processing_job(job_id, "No face detected in query image")
                return {
                    "status": "error",
                    "message": "No face detected in query image",
                    "matches": []
                }
            
            # Use face with highest detection score
            query_face = max(faces, key=lambda x: x.det_score)
            query_embedding = query_face.embedding
            
            # Get all clusters for this event
            clusters = self._get_event_clusters(event_id)
            
            matches = []
            
            # Compare with cluster representatives
            for cluster in clusters:
                # Load representative embedding
                try:
                    embedding = np.load(cluster['embedding_path'])
                    
                    # Calculate similarity
                    similarity = self._calculate_similarity(query_embedding, embedding)
                    
                    
                    if similarity >= similarity_threshold:
                        # Get faces in this cluster
                        cluster_faces = self._get_cluster_faces(cluster['cluster_id'])

                        processed_source_files = []
                        for face in cluster_faces:
                            processed_source_files.append({
                                'face_id': face['face_id'],
                                'similarity': face['similarity'],
                                'filename': face['filename'],
                                'image_id': face['image_id']
                            })
                        
                        matches.append({
                            'cluster_id': cluster['cluster_id'],
                            'person_id': cluster['cluster_id'],  # For backward compatibility
                            'similarity': float(similarity * 100),
                            'representative_url': cluster['representative_id'],
                            'face_count': cluster['total_faces'],
                            'source_files': processed_source_files
                        })
                except Exception as e:
                    logger.error(f"Error comparing with cluster {cluster['cluster_id']}: {str(e)}")
                    continue
            
            # Sort matches by similarity (highest first)
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Complete the search job
            self._complete_processing_job(job_id)
            
            return {
                "status": "success",
                "message": f"Found {len(matches)} matching persons",
                "matches": matches,
                "job_id": job_id
            }
            
        except Exception as e:
            logger.error(f"Error in face search: {str(e)}")
            try:
                self._fail_processing_job(job_id, str(e))
            except:
                pass
            return {
                "status": "error",
                "message": str(e),
                "matches": []
            }
    
    def _get_event_clusters(self, event_id):
        """
        Get all clusters for an event.
        
        Args:
            event_id: UUID of the event
            
        Returns:
            List of cluster dictionaries
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT c.cluster_id, c.total_faces, f.face_id, f.face_path, f.embedding_path
                FROM clusters c
                JOIN faces f ON c.representative_face_id = f.face_id
                WHERE c.event_id = ?
                """,
                (event_id,)
            )
            
            clusters = []
            row = cursor.fetchone()
            while row:
                clusters.append({
                    'cluster_id': row[0],
                    'total_faces': row[1],
                    'representative_id': row[2],
                    'representative_path': row[3],
                    'embedding_path': row[4]
                })
                row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error getting clusters for event {event_id}: {str(e)}")
            return []
    
    def _get_cluster_faces(self, cluster_id):
        """
        Get all faces in a cluster.
        
        Args:
            cluster_id: UUID of the cluster
            
        Returns:
            List of face dictionaries
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT f.face_id, cf.similarity_score, i.original_filename, i.image_id
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
                    'face_id': row[0],
                    'similarity': row[1],
                    'filename': row[2],
                    'image_id': row[3]
                })
                row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return faces
            
        except Exception as e:
            logger.error(f"Error getting faces for cluster {cluster_id}: {str(e)}")
            return []
    
    def process_temp_file(self, temp_file_path, event_id=None):
        """
        Process a temporary file for face detection (without saving to DB).
        
        Args:
            temp_file_path: Path to temporary image file
            event_id: Optional event ID for context
            
        Returns:
            Dictionary with detected faces
        """
        try:
            img = cv2.imread(temp_file_path)
            if img is None:
                return {
                    "status": "error",
                    "message": "Could not read image file",
                    "faces": []
                }
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with self.thread_lock:
                faces = self.app.get(img_rgb)
            
            if not faces:
                return {
                    "status": "success",
                    "message": "No faces detected",
                    "faces": []
                }
            
            result_faces = []
            for idx, face in enumerate(faces):
                face_img = face_align.norm_crop(img_rgb, face.kps)
                
                # Save to temporary location
                temp_dir = self.config.dirs['temp']
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_face_id = str(uuid.uuid4())
                face_filename = f"temp_{temp_face_id}.jpg"
                face_path = os.path.join(temp_dir, face_filename)
                
                cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                
                result_faces.append({
                    'face_index': idx,
                    'detection_score': float(face.det_score),
                    'face_path': face_path,
                    'bbox': face.bbox.tolist()
                })
            
            return {
                "status": "success",
                "message": f"Detected {len(result_faces)} faces",
                "faces": result_faces
            }
            
        except Exception as e:
            logger.error(f"Error processing temporary file: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "faces": []
            }