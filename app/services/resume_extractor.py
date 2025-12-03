import re
import os
import json
import logging
import requests
from typing import Dict, Optional
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize OpenAI client
_openai_client = None


def _get_openai_client():
    """Get or initialize OpenAI client"""
    global _openai_client
    if _openai_client is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAI(api_key=openai_api_key)
    return _openai_client


def extract_email(text: str) -> Optional[str]:
    """Extract email from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else ""


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text"""
    phone_patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # International
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'\d{10}',  # 10 digits
        r'\+?\d{1,3}[-.\s]?\d{10}',  # +91 format
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            return phones[0]
    return ""


def extract_linkedin(text: str) -> Optional[str]:
    """Extract LinkedIn URL from text"""
    linkedin_patterns = [
        r'https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?',
        r'linkedin\.com/in/[A-Za-z0-9_-]+/?',
    ]
    
    for pattern in linkedin_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            url = matches[0]
            if not url.startswith('http'):
                url = 'https://' + url
            return url
    return ""


def extract_skills(text: str) -> str:
    """Extract skills from text - looks for common skill keywords"""
    common_skills = [
        'Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go',
        'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django', 'Flask', 'FastAPI', 'Spring',
        'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'NoSQL',
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'CI/CD',
        'Git', 'GitHub', 'GitLab', 'Bitbucket',
        'Machine Learning', 'Deep Learning', 'AI', 'NLP', 'Computer Vision',
        'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy',
        'REST API', 'GraphQL', 'Microservices', 'Agile', 'Scrum',
        'HTML', 'CSS', 'Bootstrap', 'Tailwind', 'SASS',
        'TypeScript', 'jQuery', 'Redux', 'Next.js',
        'Linux', 'Unix', 'Windows', 'macOS'
    ]
    
    found_skills = []
    text_upper = text.upper()
    
    for skill in common_skills:
        if skill.upper() in text_upper:
            found_skills.append(skill)
    
    return ', '.join(found_skills[:15])  # Limit to 15 skills


def extract_position(text: str) -> str:
    """Extract position/title from text - usually near the beginning"""
    lines = text.split('\n')
    
    # Common position keywords
    position_keywords = [
        'developer', 'engineer', 'programmer', 'architect', 'lead', 'senior', 'junior',
        'manager', 'analyst', 'consultant', 'specialist', 'administrator', 'designer',
        'scientist', 'researcher', 'coordinator', 'director', 'head', 'chief'
    ]
    
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        line_lower = line.lower().strip()
        if any(keyword in line_lower for keyword in position_keywords):
            if len(line.strip()) < 100:  # Likely a title, not a paragraph
                return line.strip()
    
    return ""


def extract_name(text: str) -> str:
    """Extract name from resume - usually in the first few lines"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return ""
    
    # Common patterns to skip (not names)
    skip_keywords = [
        'resume', 'curriculum vitae', 'cv', 'profile', 'summary', 'objective',
        'contact', 'email', 'phone', 'address', 'linkedin', 'github',
        'professional', 'experience', 'education', 'skills'
    ]
    
    # Try to find name in first 5 lines
    for line in lines[:5]:
        line_lower = line.lower()
        
        # Skip lines with skip keywords
        if any(keyword in line_lower for keyword in skip_keywords):
            continue
        
        # Skip lines with email or phone patterns
        if '@' in line or re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', line):
            continue
        
        # Skip very short or very long lines
        if len(line) < 3 or len(line) > 50:
            continue
        
        # Check if line looks like a name (2-4 words, mostly alphabetic)
        words = line.split()
        if 2 <= len(words) <= 4:
            # Check if mostly alphabetic
            if all(word.replace('-', '').replace("'", '').isalpha() for word in words):
                return line
    
    # Fallback: return first non-empty line if nothing found
    return lines[0] if lines else ""


def extract_experience(text: str) -> str:
    """Extract total years of experience from resume"""
    # Patterns to look for experience mentions
    experience_patterns = [
        r'(\d+\.?\d*)\s*(?:\+)?\s*years?\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+\.?\d*)\s*(?:\+)?\s*years?',
        r'total\s+experience[:\s]+(\d+\.?\d*)\s*(?:\+)?\s*years?',
        r'(\d+\.?\d*)\s*(?:\+)?\s*yrs?\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+\.?\d*)\s*(?:\+)?\s*yrs?',
    ]
    
    years = []
    text_lower = text.lower()
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                year_value = float(match)
                if 0 <= year_value <= 50:  # Reasonable range
                    years.append(year_value)
            except ValueError:
                continue
    
    if years:
        # Return the maximum as it's likely the total experience
        max_years = max(years)
        return f"{max_years:.1f}" if max_years % 1 != 0 else str(int(max_years))
    
    # Alternative: Try to calculate from work history dates
    date_patterns = [
        r'(19|20)\d{2}\s*[-–]\s*(?:(19|20)\d{2}|present|current)',
    ]
    
    work_years = set()
    for pattern in date_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                start_year = int(match[0] + match[0][-2:]) if len(match[0]) == 2 else int(match[0])
                # Add to work years set
                work_years.add(start_year)
    
    return ""


def read_docx_from_url(url: str) -> str:
    """Read DOCX file from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        doc = Document(BytesIO(response.content))
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error reading DOCX from {url}: {str(e)}")
        return ""


def read_pdf_from_url(url: str) -> str:
    """Read PDF file from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        pdf = PdfReader(BytesIO(response.content))
        text = []
        for page in pdf.pages:
            text.append(page.extract_text())
        return '\n'.join(text)
    except Exception as e:
        logger.error(f"Error reading PDF from {url}: {str(e)}")
        return ""


def extract_text_from_url(url: str) -> str:
    """
    Extract raw text content from resume URL
    
    Args:
        url: URL of the resume file (supports PDF, DOCX, and Google Docs)
        
    Returns:
        Extracted text content
    """
    try:
        # Handle Google Docs Viewer URLs (gview)
        if 'docs.google.com/gview' in url:
            # Extract the actual document URL from the gview parameter
            url_match = re.search(r'url=([^&]+)', url)
            if url_match:
                actual_url = url_match.group(1)
                # URL decode if necessary
                from urllib.parse import unquote
                actual_url = unquote(actual_url)
                logger.info(f"Extracted actual URL from gview: {actual_url}")
                # Recursively call with the actual URL
                return extract_text_from_url(actual_url)
        
        # Handle Google Docs URLs
        if 'docs.google.com' in url:
            # Convert Google Docs view/edit URL to export URL
            if '/edit' in url or '/view' in url:
                doc_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
                if doc_id_match:
                    doc_id = doc_id_match.group(1)
                    # Export as PDF
                    export_url = f'https://docs.google.com/document/d/{doc_id}/export?format=pdf'
                    text = read_pdf_from_url(export_url)
                    if text:
                        return text
        
        # Read resume content based on file type
        if url.lower().endswith('.docx') or 'export?format=docx' in url.lower():
            text = read_docx_from_url(url)
        elif url.lower().endswith('.pdf') or 'export?format=pdf' in url.lower():
            text = read_pdf_from_url(url)
        else:
            logger.warning(f"Unsupported file format: {url}")
            return ""
        
        return text if text else ""
        
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        return ""


def extract_resume_info(url: str) -> Dict[str, str]:
    """
    Extract information from resume URL using OpenAI GPT-4o-mini
    
    Args:
        url: URL of the resume file
        
    Returns:
        Dictionary with extracted information
    """
    try:
        # Extract text from URL
        text = extract_text_from_url(url)
        
        if not text:
            return {
                "name": "",
                "email": "",
                "contact_number": "",
                "linkedin_url": "",
                "skills": "",
                "position": "",
                "total_experience": ""
            }
        
        # Use OpenAI to extract structured information
        client = _get_openai_client()
        
        prompt = f"""Extract the following information from this resume and return it as a JSON object with these exact keys:
- name: Full name of the candidate
- email: Email address
- contact_number: Phone number (with country code if available)
- linkedin_url: LinkedIn profile URL (complete URL with https://)
- skills: Comma-separated list of technical skills (max 15 most relevant)
- position: Current or most recent job title/position
- total_experience: Total years of professional experience (just the number, e.g., "5" or "5.5")

Rules:
- Return ONLY valid JSON, no additional text
- If a field is not found, use an empty string ""
- For skills, prioritize technical skills and frameworks
- For total_experience, if you see ranges like "5+ years", use "5"; if you see "3-5 years", use "5"
- Make sure the name is properly capitalized

Resume text:
{text[:4000]}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a resume parsing assistant. Extract information accurately and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Ensure all required fields are present
        extracted_info = {
            "name": result.get("name", ""),
            "email": result.get("email", ""),
            "contact_number": result.get("contact_number", ""),
            "linkedin_url": result.get("linkedin_url", ""),
            "skills": result.get("skills", ""),
            "position": result.get("position", ""),
            "total_experience": result.get("total_experience", "")
        }
        
        logger.info(f"Successfully extracted info using OpenAI: {extracted_info.get('name', 'Unknown')}")
        return extracted_info
        
    except Exception as e:
        logger.error(f"Error extracting resume info from {url}: {str(e)}")
        # Fallback to empty values
        return {
            "name": "",
            "email": "",
            "contact_number": "",
            "linkedin_url": "",
            "skills": "",
            "position": "",
            "total_experience": ""
        }


