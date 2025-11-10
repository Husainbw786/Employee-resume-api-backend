import openai
import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ATSScorer:
    """Service to score resumes using OpenAI GPT-4o based on ATS criteria"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"  # Using GPT-4o as requested
    
    async def calculate_ats_score(self, resume_text: str, job_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive ATS score for a resume
        
        Args:
            resume_text: The extracted text from the resume
            job_description: Optional job description for targeted scoring
            
        Returns:
            Dictionary containing detailed ATS scoring results
        """
        try:
            logger.info("Starting ATS score calculation using GPT-4o")
            
            # Create system and user messages
            system_message = self._create_system_message()
            user_message = self._create_user_message(resume_text, job_description)
            
            # Combine system and user messages into a single input
            combined_input = f"{system_message}\n\n{user_message}"
            
            response = self.client.responses.create(
                model=self.model,
                input=combined_input,
                reasoning={ "effort": "minimal" }
            )
            
            # Parse the response
            result = self._parse_openai_response(response.output_text)
            
            # Add timestamp
            result["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"ATS scoring completed. Overall score: {result.get('overall_score', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ATS scoring: {str(e)}")
            return self._create_fallback_response(str(e))
    
    def _create_system_message(self) -> str:
        """Create the system message with ATS evaluation guidelines"""
        return """You are an expert ATS (Applicant Tracking System) evaluator with deep knowledge of resume screening, recruitment best practices, and candidate verification. Analyze resumes with a critical eye for authenticity, consistency, and quality.

**SCORING CRITERIA (Total: 100 points):**

1. **Tech Stack Consistency – 20 points**
   - Evaluate if technologies are used together logically within projects
   - Check for realistic relationships between tools/frameworks
   - Look for contradictory tech combinations (e.g., Spring Boot + Django in same project)
   - Assess if tech choices make sense for the claimed experience level

2. **LinkedIn Authenticity & Alignment – 15 points**
   - Based on resume content, assess likelihood of matching LinkedIn profile
   - Evaluate completeness of professional information provided
   - Check for professional presentation and credibility indicators
   - Note any red flags that might indicate profile mismatches

3. **Project Depth and Relevance – 20 points**
   - Analyze project descriptions for sufficient detail (problem, responsibilities, tools, outcomes)
   - Verify if responsibilities align with claimed role seniority
   - Check if experience scale matches years of experience
   - Assess technical depth and real-world applicability

4. **Resume Length & Format Quality – 10 points**
   - Evaluate structure, readability, and organization
   - Check for appropriate length (1-2 pages junior, 2-3 pages mid-level+)
   - Assess section clarity and formatting consistency
   - Look for grammar, spelling, and presentation errors

5. **Duplicate or Template Content – 15 points**
   - Identify generic or overly templated language
   - Look for unique, personalized content vs. boilerplate text
   - Assess authenticity of descriptions and achievements
   - Check for robotic or copy-paste indicators

6. **Employment Timeline Coherence – 10 points**
   - Verify logical employment progression and dates
   - Check for unexplained gaps or overlapping positions
   - Assess career growth consistency with experience level
   - Evaluate job transition patterns

7. **Education and Certification Validation – 10 points**
   - Assess relevance and credibility of educational background
   - Check alignment between certifications and claimed skills
   - Evaluate if education supports the career trajectory
   - Look for appropriate timing and progression

**OUTPUT FORMAT:**
You MUST respond with ONLY a valid JSON object in this exact structure (no markdown, no code blocks):
{
    "overall_score": <0-100>,
    "category_scores": {
        "tech_stack_consistency": {
            "score": <0-20>,
            "feedback": "Detailed analysis...",
            "red_flags": ["flag1", "flag2"]
        },
        "linkedin_authenticity": {
            "score": <0-15>,
            "feedback": "Detailed analysis...",
            "red_flags": ["flag1", "flag2"]
        },
        "project_depth": {
            "score": <0-20>,
            "feedback": "Detailed analysis...",
            "red_flags": ["flag1", "flag2"]
        },
        "format_quality": {
            "score": <0-10>,
            "feedback": "Detailed analysis...",
            "red_flags": ["flag1", "flag2"]
        },
        "content_authenticity": {
            "score": <0-15>,
            "feedback": "Detailed analysis...",
            "red_flags": ["flag1", "flag2"]
        },
        "timeline_coherence": {
            "score": <0-10>,
            "feedback": "Detailed analysis...",
            "red_flags": ["flag1", "flag2"]
        },
        "education_validation": {
            "score": <0-10>,
            "feedback": "Detailed analysis...",
            "red_flags": ["flag1", "flag2"]
        }
    },
    "summary": "Overall assessment summary...",
    "recommendations": ["improvement1", "improvement2", "improvement3"],
    "risk_level": "LOW|MEDIUM|HIGH",
    "confidence_score": <0-100>,
    "job_alignment": {
        "job_match_score": <0-100>,
        "relevant_skills_found": ["skill1", "skill2"],
        "missing_critical_skills": ["skill1", "skill2"],
        "experience_level_match": "Analysis of experience level fit...",
        "job_specific_recommendations": ["recommendation1", "recommendation2"]
    }
}

When a job description is provided, include detailed job_alignment analysis. When no job description is provided, you can omit the job_alignment section or set job_match_score to null.
"""
    
    def _create_user_message(self, resume_text: str, job_description: Optional[str] = None) -> str:
        """Create the user message with resume and job description"""
        message = f"""Please analyze the following resume and provide a comprehensive ATS score.

**RESUME TEXT:**
{resume_text}
"""
        
        if job_description:
            message += f"""

**JOB DESCRIPTION:**
{job_description}

Please evaluate how well this resume aligns with the job description, including:
- Skill match and gaps
- Experience level fit
- Specific recommendations for this role
"""
        
        return message
    
    def _parse_openai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse OpenAI response and extract structured data"""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                # Validate required fields
                if 'overall_score' in parsed_data and 'category_scores' in parsed_data:
                    return parsed_data
            
            # If JSON parsing fails, create structured response from text
            return self._create_structured_response_from_text(response_text)
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from OpenAI response, creating structured response")
            return self._create_structured_response_from_text(response_text)
    
    def _create_structured_response_from_text(self, response_text: str) -> Dict[str, Any]:
        """Create structured response when JSON parsing fails"""
        # Extract overall score if mentioned
        import re
        score_match = re.search(r'(?:overall|total|final).*?score.*?(\d+)', response_text, re.IGNORECASE)
        overall_score = int(score_match.group(1)) if score_match else 75
        
        return {
            "overall_score": overall_score,
            "category_scores": {
                "tech_stack_consistency": {
                    "score": 15,
                    "feedback": "Analysis based on text review",
                    "red_flags": []
                },
                "linkedin_authenticity": {
                    "score": 12,
                    "feedback": "Analysis based on text review",
                    "red_flags": []
                },
                "project_depth": {
                    "score": 15,
                    "feedback": "Analysis based on text review",
                    "red_flags": []
                },
                "format_quality": {
                    "score": 8,
                    "feedback": "Analysis based on text review",
                    "red_flags": []
                },
                "content_authenticity": {
                    "score": 11,
                    "feedback": "Analysis based on text review",
                    "red_flags": []
                },
                "timeline_coherence": {
                    "score": 8,
                    "feedback": "Analysis based on text review",
                    "red_flags": []
                },
                "education_validation": {
                    "score": 8,
                    "feedback": "Analysis based on text review",
                    "red_flags": []
                }
            },
            "summary": response_text[:500] + "..." if len(response_text) > 500 else response_text,
            "recommendations": ["Review detailed feedback", "Consider professional formatting", "Enhance project descriptions"],
            "risk_level": "MEDIUM",
            "confidence_score": 70,
            "parsing_note": "Response was parsed from unstructured text due to JSON parsing issues"
        }
    
    def _create_fallback_response(self, error_message: str) -> Dict[str, Any]:
        """Create fallback response when OpenAI call fails"""
        return {
            "overall_score": 0,
            "category_scores": {
                "tech_stack_consistency": {"score": 0, "feedback": "Unable to analyze", "red_flags": []},
                "linkedin_authenticity": {"score": 0, "feedback": "Unable to analyze", "red_flags": []},
                "project_depth": {"score": 0, "feedback": "Unable to analyze", "red_flags": []},
                "format_quality": {"score": 0, "feedback": "Unable to analyze", "red_flags": []},
                "content_authenticity": {"score": 0, "feedback": "Unable to analyze", "red_flags": []},
                "timeline_coherence": {"score": 0, "feedback": "Unable to analyze", "red_flags": []},
                "education_validation": {"score": 0, "feedback": "Unable to analyze", "red_flags": []}
            },
            "summary": f"ATS scoring failed due to technical error: {error_message}",
            "recommendations": ["Please try again", "Ensure resume URL is accessible", "Check OpenAI API configuration"],
            "risk_level": "UNKNOWN",
            "confidence_score": 0,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }

