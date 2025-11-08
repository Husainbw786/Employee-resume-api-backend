from typing import Dict, Any
from fastapi import APIRouter
from app.controllers.resume_controller import search_resumes_controller

router = APIRouter(prefix="/api/v1", tags=["resumes"])


@router.post("/search")
async def search_resumes(request: Dict[str, Any]):
    """
    Search for resumes matching a job description
    
    Request body:
    - **JD**: Job description text (required, minimum 10 characters)
    - **count**: Number of top candidates to return (optional, default: 10, min: 1, max: 100)
    """
    return search_resumes_controller(request)

