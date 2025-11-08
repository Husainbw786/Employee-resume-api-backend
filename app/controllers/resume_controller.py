import logging
from typing import Dict, Any
from fastapi import HTTPException
from app.services.resume_service import search_candidates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def search_resumes_controller(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Controller logic for searching resumes
    
    Args:
        request: Dictionary containing JD and count
        
    Returns:
        Response dictionary with candidates
    """
    try:
        # Extract parameters from request body
        JD = request.get("JD", "")
        count = request.get("count", 10)
        
        # Validate inputs
        if not JD or len(JD) < 10:
            raise HTTPException(status_code=400, detail="JD must be at least 10 characters")
        
        if count < 1 or count > 100:
            raise HTTPException(status_code=400, detail="count must be between 1 and 100")
        
        logger.info(f"Search request received: JD length={len(JD)}, count={count}")
        
        # Search for candidates
        candidates_data = search_candidates(job_description=JD, count=count)
        
        response = {
            "success": True,
            "message": f"Found {len(candidates_data)} matching candidates",
            "candidates": candidates_data,
            "total_found": len(candidates_data),
            "job_description": JD
        }
        
        logger.info(f"Search completed: {len(candidates_data)} candidates found")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

