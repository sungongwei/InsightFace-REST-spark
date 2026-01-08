import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from if_rest.core.database import FaceDatabase


class FaceSearchEngine:
    """
    FAISS-based face similarity search engine with CPU as default (GPU optional).
    """

    def __init__(self, db_path: str = "faces.db", dimension: int = 512, use_gpu: bool = False):
        self.db = FaceDatabase(db_path)
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.db_path = db_path  # Store the database path to derive FAISS index path

        # Initialize FAISS index - CPU by default
        if use_gpu:
            try:
                # Check if GPU is available
                if hasattr(faiss, 'GpuIndexFlatIP'):
                    # Configure GPU resources
                    res = faiss.StandardGpuResources()
                    self.index = faiss.GpuIndexFlatIP(res, dimension)  # GPU index for inner product
                    print("Using GPU-accelerated FAISS index")
                else:
                    print("GPU FAISS not available, using CPU")
                    self.index = faiss.IndexFlatIP(dimension)  # CPU fallback
                    self.use_gpu = False
            except Exception as e:
                print(f"Error initializing GPU FAISS: {e}, using CPU")
                self.index = faiss.IndexFlatIP(dimension)  # CPU fallback
                self.use_gpu = False
        else:
            # Use CPU by default
            self.index = faiss.IndexFlatIP(dimension)  # CPU index for inner product
            print("Using CPU FAISS index")

        self.face_ids = []  # Store face IDs corresponding to index vectors
        self.load_index()

    def get_faiss_index_path(self):
        """Generate FAISS index file path based on database path."""
        base_path = os.path.splitext(self.db_path)[0]
        return f"{base_path}_faiss.index"

    def save_index(self):
        """Save FAISS index and face_ids to disk."""
        faiss_index_path = self.get_faiss_index_path()

        # Save FAISS index
        faiss.write_index(self.index, faiss_index_path)

        # Save face_ids separately (since they're Python objects)
        face_ids_path = f"{os.path.splitext(faiss_index_path)[0]}_face_ids.pkl"
        with open(face_ids_path, 'wb') as f:
            pickle.dump(self.face_ids, f)

    def load_persistent_index(self):
        """Load FAISS index and face_ids from disk if they exist."""
        faiss_index_path = self.get_faiss_index_path()
        face_ids_path = f"{os.path.splitext(faiss_index_path)[0]}_face_ids.pkl"

        # Check if both files exist
        if os.path.exists(faiss_index_path) and os.path.exists(face_ids_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(faiss_index_path)

                # Load face_ids
                with open(face_ids_path, 'rb') as f:
                    self.face_ids = pickle.load(f)

                print(f"Loaded FAISS index from {faiss_index_path} with {len(self.face_ids)} faces")
                return True
            except Exception as e:
                print(f"Error loading persistent FAISS index: {e}")
                return False
        return False
    
    def load_index(self):
        """Load all face embeddings from database into FAISS index."""
        # Try to load persistent index first
        if self.load_persistent_index():
            return  # Successfully loaded from persistent storage

        # If no persistent index exists, load from database
        self.index.reset()
        self.face_ids = []

        faces = self.db.get_all_faces()
        if not faces:
            return

        # Prepare embeddings matrix
        embeddings = []
        for face in faces:
            embedding = face['embedding']
            # Normalize the embedding for cosine similarity
            embedding_norm = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding_norm)
            self.face_ids.append(face['id'])

        if embeddings:
            embeddings_matrix = np.vstack(embeddings).astype('float32')
            self.index.add(embeddings_matrix)
    
    def add_face(self, face_id: int, embedding: np.ndarray):
        """Add a face embedding to the search index."""
        # Normalize the embedding for cosine similarity
        embedding_norm = embedding / np.linalg.norm(embedding)
        embedding_norm = embedding_norm.astype('float32').reshape(1, -1)

        self.index.add(embedding_norm)
        self.face_ids.append(face_id)

        # Persist the updated index
        self.save_index()
    
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.6) -> List[Tuple[int, float]]:
        """
        Search for similar faces to the query embedding.
        
        Args:
            query_embedding: The embedding to search for
            k: Number of top results to return
            threshold: Minimum similarity threshold (cosine similarity)
        
        Returns:
            List of tuples (face_id, similarity_score) sorted by similarity
        """
        # Normalize the query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.astype('float32').reshape(1, -1)
        
        # Perform similarity search
        # Ensure k is at least 1 and we have faces in the index
        if len(self.face_ids) == 0:
            return []  # No faces to search

        similarities, indices = self.index.search(query_norm, min(k, len(self.face_ids)))
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            similarity = float(similarities[0][i])
            
            # Only return results above threshold
            if similarity >= threshold and idx < len(self.face_ids):
                face_id = self.face_ids[idx]
                results.append((face_id, similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def remove_face(self, face_id: int):
        """Remove a face from the search index."""
        try:
            idx = self.face_ids.index(face_id)
            # Note: FAISS doesn't support efficient deletion of individual vectors
            # So we rebuild the index without the removed face
            self.rebuild_index()
            # Persist the updated index
            self.save_index()
        except ValueError:
            # Face ID not found in index
            pass
    
    def rebuild_index(self):
        """Rebuild the search index from the database."""
        self.load_index()
        # After rebuilding from database, save the updated index
        self.save_index()
    
    def update_face(self, face_id: int, new_embedding: np.ndarray):
        """Update a face embedding in the search index."""
        try:
            idx = self.face_ids.index(face_id)
            # Rebuild index since FAISS doesn't support efficient updates
            self.rebuild_index()
            # Persist the updated index
            self.save_index()
        except ValueError:
            # Face ID not found, add it as new
            self.add_face(face_id, new_embedding)