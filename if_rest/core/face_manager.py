import base64
import io
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from if_rest.core.database import FaceDatabase
from if_rest.core.face_search import FaceSearchEngine
from if_rest.core.processing import Processing


class FaceManager:
    """
    High-level face management interface that combines database and search functionality.
    """

    def __init__(self, db_path: str = "faces.db", processing: Optional[Processing] = None, use_gpu: bool = False):
        self.db = FaceDatabase(db_path)
        self.search_engine = FaceSearchEngine(db_path, use_gpu=use_gpu)
        self.processing = processing
        self.use_gpu = use_gpu
    
    async def add_face_from_image(self, name: str, image_data: bytes, 
                                  extract_embedding: bool = True) -> Optional[Dict[str, Any]]:
        """
        Add a face to the database by extracting embedding from an image.
        
        Args:
            name: Name/label for the face
            image_data: Raw image bytes
            extract_embedding: Whether to extract embedding from the image
        
        Returns:
            Dictionary with face information or None if no face detected
        """
        if not self.processing:
            raise ValueError("Processing module not provided")
        
        # Extract face data from image
        from if_rest.schemas import Images
        images = Images(data=[base64.b64encode(image_data).decode('utf-8')])
        
        result = await self.processing.extract(
            images=images,
            extract_embedding=extract_embedding,
            extract_ga=True,
            return_face_data=True,
            threshold=0.5
        )
        
        # Get the first face from the first image
        if result['data'] and result['data'][0]['faces']:
            face_data = result['data'][0]['faces'][0]
            
            if extract_embedding and 'vec' in face_data:
                embedding = np.array(face_data['vec'], dtype=np.float32)
                gender = face_data.get('gender')
                age = face_data.get('age')
                
                # Get the face crop if available
                face_crop = None
                if face_data.get('facedata') is not None:
                    # Convert the face crop to bytes
                    try:
                        is_success, buffer = cv2.imencode(".jpg", face_data['facedata'])
                        if is_success:
                            face_crop = buffer.tobytes()
                    except Exception as e:
                        print(f"Error encoding face image: {e}")
                        face_crop = None
                
                # Add to database
                face_id = self.db.add_face(
                    name=name,
                    embedding=embedding,
                    image_data=face_crop,
                    gender=gender,
                    age=age
                )
                
                # Add to search index
                self.search_engine.add_face(face_id, embedding)
                
                return {
                    'id': face_id,
                    'name': name,
                    'embedding': embedding.tolist() if embedding is not None else None,
                    'image_data': face_crop,
                    'gender': gender,
                    'age': age,
                    'similarity': 1.0  # New face, perfect match with itself
                }
        
        return None
    
    def get_face_by_id(self, face_id: int) -> Optional[Dict[str, Any]]:
        """Get face information by ID."""
        face = self.db.get_face_by_id(face_id)
        if face:
            return {
                'id': face['id'],
                'name': face['name'],
                'embedding': face['embedding'].tolist(),
                'image_data': face['image_data'],
                'gender': face['gender'],
                'age': face['age'],
                'created_at': face['created_at'],
                'updated_at': face['updated_at']
            }
        return None
    
    def get_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Get all faces with a specific name."""
        faces = self.db.get_faces_by_name(name)
        return [
            {
                'id': face['id'],
                'name': face['name'],
                'embedding': face['embedding'].tolist(),
                'image_data': face['image_data'],
                'gender': face['gender'],
                'age': face['age'],
                'created_at': face['created_at'],
                'updated_at': face['updated_at']
            }
            for face in faces
        ]
    
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """Get all faces from the database."""
        faces = self.db.get_all_faces()
        return [
            {
                'id': face['id'],
                'name': face['name'],
                'embedding': face['embedding'].tolist(),
                'image_data': face['image_data'],
                'gender': face['gender'],
                'age': face['age'],
                'created_at': face['created_at'],
                'updated_at': face['updated_at']
            }
            for face in faces
        ]
    
    def update_face(self, face_id: int, name: Optional[str] = None,
                    embedding: Optional[np.ndarray] = None,
                    image_data: Optional[bytes] = None,
                    gender: Optional[int] = None,
                    age: Optional[int] = None) -> bool:
        """Update an existing face."""
        updated = self.db.update_face(
            face_id=face_id,
            name=name,
            embedding=embedding,
            image_data=image_data,
            gender=gender,
            age=age
        )
        
        if updated and embedding is not None:
            # Update search index
            self.search_engine.update_face(face_id, embedding)
        
        return updated
    
    def delete_face(self, face_id: int) -> bool:
        """Delete a face from database and search index."""
        # Remove from search index first
        self.search_engine.remove_face(face_id)
        
        # Then delete from database
        return self.db.delete_face(face_id)
    
    def delete_faces_by_name(self, name: str) -> int:
        """Delete all faces with a specific name."""
        # Get all faces with this name to remove from search index
        faces = self.db.get_faces_by_name(name)
        for face in faces:
            self.search_engine.remove_face(face['id'])
        
        # Delete from database
        return self.db.delete_faces_by_name(name)
    
    def search_similar_faces(self, query_embedding: np.ndarray, k: int = 5,
                           threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar faces to the query embedding.

        Args:
            query_embedding: The embedding to search for
            k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of dictionaries with face information and similarity scores
        """
        results = self.search_engine.search(query_embedding, k, threshold)

        similar_faces = []
        for face_id, similarity in results:
            face = self.get_face_by_id(face_id)
            if face:
                face['similarity'] = similarity
                similar_faces.append(face)

        return similar_faces
    
    async def search_faces_by_image(self, image_data: bytes, k: int = 5,
                            threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar faces by providing an image.

        Args:
            image_data: Raw image bytes
            k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of dictionaries with face information and similarity scores
        """
        if not self.processing:
            raise ValueError("Processing module not provided")

        # Extract embedding from the query image
        from if_rest.schemas import Images
        images = Images(data=[base64.b64encode(image_data).decode('utf-8')])

        result = await self.processing.extract(
            images=images,
            extract_embedding=True,
            extract_ga=False,
            return_face_data=False,
            threshold=0.5
        )

        # Get the first face from the first image
        if result['data'] and result['data'][0]['faces']:
            face_data = result['data'][0]['faces'][0]

            if 'vec' in face_data:
                query_embedding = np.array(face_data['vec'], dtype=np.float32)
                return self.search_similar_faces(query_embedding, k, threshold)

        return []