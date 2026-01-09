import base64
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from pydantic import BaseModel

from if_rest.core.face_manager import FaceManager
from if_rest.api.routes.v1.face_management import FaceResponse, get_face_manager


router = APIRouter()


class SearchResponse(BaseModel):
    id: int
    name: str
    gender: Optional[int] = None
    age: Optional[int] = None
    image_data: Optional[str] = None  # Base64 encoded image data
    similarity: float
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.post('/search', tags=['Face Search'])
async def search_similar_faces(
    file: UploadFile = File(None),
    image_data: str = Form(None),  # Base64 encoded image data
    k: int = Form(5),
    threshold: float = Form(0.5),
    face_manager = Depends(get_face_manager)
) -> List[SearchResponse]:
    """
    Search for similar faces in the database using an uploaded image or base64 encoded image data.

    - **file**: Query image containing the face to search for (alternative to image_data)
    - **image_data**: Base64 encoded image data (alternative to file)
    - **k**: Number of top results to return (default: 5)
    - **threshold**: Minimum similarity threshold (default: 0.5)
    """
    if not file and not image_data:
        raise HTTPException(status_code=400, detail="Either image file or base64 image data is required")

    # Handle image data from either file upload or base64 string
    if file:
        # Read image data from uploaded file
        image_bytes = await file.read()
    elif image_data:
        # Decode base64 image data
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # Search for similar faces
    results = await face_manager.search_faces_by_image(
        image_data=image_bytes,
        k=k,
        threshold=threshold
    )

    return [
        SearchResponse(
            id=face['id'],
            name=face['name'],
            gender=face['gender'],
            age=face['age'],
            image_data=base64.b64encode(face['image_data']).decode('utf-8') if face['image_data'] else None,
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
    threshold: float = Form(0.5),
    face_manager = Depends(get_face_manager)
) -> List[SearchResponse]:
    """
    Search for similar faces using a face embedding.

    - **embedding**: JSON string of the embedding array
    - **k**: Number of top results to return (default: 5)
    - **threshold**: Minimum similarity threshold (default: 0.5)
    """
    import json
    import numpy as np

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
            image_data=base64.b64encode(face['image_data']).decode('utf-8') if face['image_data'] else None,
            similarity=face['similarity'],
            created_at=face['created_at'],
            updated_at=face['updated_at']
        )
        for face in results
    ]