import base64
from typing import Annotated, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from if_rest.core.face_manager import FaceManager
from if_rest.core.processing import ProcessingDep


router = APIRouter()


class FaceResponse(BaseModel):
    id: int
    name: str
    gender: Optional[int] = None
    age: Optional[int] = None
    image_data: Optional[str] = None  # Base64 encoded image data
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    similarity: Optional[float] = None


class AddFaceRequest(BaseModel):
    name: str


class UpdateFaceRequest(BaseModel):
    name: Optional[str] = None


# Initialize face manager as a global variable to be set during app startup
face_manager: Optional[FaceManager] = None


def set_face_manager(fm: FaceManager):
    global face_manager
    face_manager = fm


def get_face_manager():
    """Dependency to get the face manager instance"""
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")
    return face_manager


@router.post('/faces', tags=['Face Management'])
async def add_face(
    name: str = Form(...),
    file: UploadFile = File(...)
) -> FaceResponse:
    """
    Add a new face to the database.
    
    - **name**: Name/label for the face
    - **file**: Image file containing the face
    """
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")
    
    if not file:
        raise HTTPException(status_code=400, detail="Image file is required")
    
    # Read image data
    image_data = await file.read()
    
    # Add face to database
    result = await face_manager.add_face_from_image(name=name, image_data=image_data)
    
    if result is None:
        raise HTTPException(status_code=400, detail="No face detected in the provided image")
    
    return FaceResponse(
        id=result['id'],
        name=result['name'],
        gender=result['gender'],
        age=result['age'],
        image_data=base64.b64encode(result['image_data']).decode('utf-8') if result['image_data'] else None
    )


@router.get('/faces/{face_id}', tags=['Face Management'])
async def get_face(face_id: int) -> FaceResponse:
    """
    Get face information by ID.
    """
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")

    face = face_manager.get_face_by_id(face_id)

    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    return FaceResponse(
        id=face['id'],
        name=face['name'],
        gender=face['gender'],
        age=face['age'],
        image_data=base64.b64encode(face['image_data']).decode('utf-8') if face['image_data'] else None,
        created_at=face['created_at'],
        updated_at=face['updated_at']
    )


@router.get('/faces', tags=['Face Management'])
async def get_all_faces() -> List[FaceResponse]:
    """
    Get all faces from the database.
    """
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")

    faces = face_manager.get_all_faces()

    return [
        FaceResponse(
            id=face['id'],
            name=face['name'],
            gender=face['gender'],
            age=face['age'],
            image_data=base64.b64encode(face['image_data']).decode('utf-8') if face['image_data'] else None,
            created_at=face['created_at'],
            updated_at=face['updated_at']
        )
        for face in faces
    ]


@router.get('/faces/search', tags=['Face Management'])
async def search_faces_by_name(name: str) -> List[FaceResponse]:
    """
    Search faces by name (partial match).
    """
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")

    faces = face_manager.get_faces_by_name(name)

    return [
        FaceResponse(
            id=face['id'],
            name=face['name'],
            gender=face['gender'],
            age=face['age'],
            image_data=base64.b64encode(face['image_data']).decode('utf-8') if face['image_data'] else None,
            created_at=face['created_at'],
            updated_at=face['updated_at']
        )
        for face in faces
    ]


@router.put('/faces/{face_id}', tags=['Face Management'])
async def update_face(
    face_id: int,
    request: UpdateFaceRequest
) -> FaceResponse:
    """
    Update an existing face.
    """
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")

    # Check if face exists
    existing_face = face_manager.get_face_by_id(face_id)
    if not existing_face:
        raise HTTPException(status_code=404, detail="Face not found")

    # Update the face
    updated = face_manager.update_face(
        face_id=face_id,
        name=request.name
    )

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update face")

    # Return updated face
    updated_face = face_manager.get_face_by_id(face_id)
    return FaceResponse(
        id=updated_face['id'],
        name=updated_face['name'],
        gender=updated_face['gender'],
        age=updated_face['age'],
        image_data=base64.b64encode(updated_face['image_data']).decode('utf-8') if updated_face['image_data'] else None,
        created_at=updated_face['created_at'],
        updated_at=updated_face['updated_at']
    )


@router.delete('/faces/{face_id}', tags=['Face Management'])
async def delete_face(face_id: int) -> dict:
    """
    Delete a face by ID.
    """
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")
    
    deleted = face_manager.delete_face(face_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Face not found")
    
    return {"message": "Face deleted successfully", "id": face_id}


@router.delete('/faces', tags=['Face Management'])
async def delete_faces_by_name(name: str) -> dict:
    """
    Delete all faces with a specific name.
    """
    if face_manager is None:
        raise HTTPException(status_code=500, detail="Face manager not initialized")
    
    deleted_count = face_manager.delete_faces_by_name(name)
    
    return {"message": f"Deleted {deleted_count} faces", "name": name}