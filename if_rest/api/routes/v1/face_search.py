from typing import List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from if_rest.core.face_manager import FaceManager
from if_rest.api.routes.v1.face_management import FaceResponse


router = APIRouter()


class SearchResponse(BaseModel):
    id: int
    name: str
    gender: int = None
    age: int = None
    similarity: float
    created_at: str = None
    updated_at: str = None


@router.post('/search', tags=['Face Search'])
async def search_similar_faces(
    file: UploadFile = File(...),
    k: int = Form(5),
    threshold: float = Form(0.6)
) -> List[SearchResponse]:
    """
    Search for similar faces in the database using an uploaded image.
    
    - **file**: Query image containing the face to search for
    - **k**: Number of top results to return (default: 5)
    - **threshold**: Minimum similarity threshold (default: 0.6)
    """
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")
    
    if not file:
        raise HTTPException(status_code=400, detail="Image file is required")
    
    # Read image data
    image_data = await file.read()
    
    # Search for similar faces
    results = await face_manager.search_faces_by_image(
        image_data=image_data,
        k=k,
        threshold=threshold
    )
    
    return [
        SearchResponse(
            id=face['id'],
            name=face['name'],
            gender=face['gender'],
            age=face['age'],
            similarity=face['similarity'],
            created_at=face['created_at'],
            updated_at=face['updated_at']
        )
        for face in results
    ]


@router.post('/search/embedding', tags=['Face Search'])
async def search_by_embedding(
    embedding: str = Form(...),  # JSON string of embedding array
    k: int = Form(5),
    threshold: float = Form(0.6)
) -> List[SearchResponse]:
    """
    Search for similar faces using a face embedding.
    
    - **embedding**: JSON string of the embedding array
    - **k**: Number of top results to return (default: 5)
    - **threshold**: Minimum similarity threshold (default: 0.6)
    """
    import json
    import numpy as np
    
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")
    
    try:
        embedding_array = json.loads(embedding)
        embedding_np = np.array(embedding_array, dtype=np.float32)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid embedding format")
    
    # Search for similar faces
    results = face_manager.search_similar_faces(
        query_embedding=embedding_np,
        k=k,
        threshold=threshold
    )
    
    return [
        SearchResponse(
            id=face['id'],
            name=face['name'],
            gender=face['gender'],
            age=face['age'],
            similarity=face['similarity'],
            created_at=face['created_at'],
            updated_at=face['updated_at']
        )
        for face in results
    ]