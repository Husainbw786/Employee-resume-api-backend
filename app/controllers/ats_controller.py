import logging
from typing import Dict, Any
from fastapi import HTTPException
from app.services.ats_scorer import ATSScorer
from app.services.resume_extractor import extract_text_from_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def calculate_ats_score_controller(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Controller logic for calculating ATS score for a resume
    
    Args:
        request: Dictionary containing resume_url and job_description
        
    Returns:
        Response dictionary with ATS score details
    """
    try:
        # Extract parameters from request body
        resume_url = request.get("resume_url", "")
        job_description = request.get("job_description", "")
        
        # Validate inputs
        if not resume_url:
            raise HTTPException(status_code=400, detail="resume_url is required")
        
        if not job_description or len(job_description) < 10:
            raise HTTPException(status_code=400, detail="job_description must be at least 10 characters")
        
        logger.info(f"ATS score request received: resume_url={resume_url}, JD length={len(job_description)}")
        
        # Extract resume text from URL
        logger.info(f"Extracting resume text from URL: {resume_url}")
        resume_text = extract_text_from_url(resume_url)
        
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Unable to extract sufficient text from resume URL. Please ensure the URL is accessible and contains a valid resume."
            )
        
        logger.info(f"Resume text extracted successfully. Length: {len(resume_text)} characters")
        
        # Calculate ATS score
        ats_scorer = ATSScorer()
        ats_result = await ats_scorer.calculate_ats_score(
            resume_text=resume_text,
            job_description=job_description
        )
        
        # Add success indicator
        ats_result["success"] = True
        
        logger.info(f"ATS scoring completed: Overall score={ats_result.get('overall_score', 'N/A')}")
        return ats_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Server error in ATS scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

