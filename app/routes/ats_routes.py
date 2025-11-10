from typing import Dict, Any
from fastapi import APIRouter
from app.controllers.ats_controller import calculate_ats_score_controller

router = APIRouter(prefix="/api/v1", tags=["ats"])


@router.post("/ats-score")
async def calculate_ats_score(request: Dict[str, Any]):
    """
    Calculate ATS score for a resume against a job description
    
    Request body:
    - **resume_url**: URL to the resume document (required)
    - **job_description**: Job description text (required, minimum 10 characters)
    
    Returns:
    - **overall_score**: Overall ATS score (0-100)
    - **category_scores**: Detailed scores for each category
    - **job_alignment**: Job-specific alignment analysis
    - **summary**: Overall assessment summary
    - **recommendations**: List of improvement suggestions
    - **risk_level**: Risk assessment (LOW/MEDIUM/HIGH)
    - **confidence_score**: Confidence in the scoring (0-100)
    """
    return await calculate_ats_score_controller(request)

