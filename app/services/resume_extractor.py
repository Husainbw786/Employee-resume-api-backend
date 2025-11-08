import re
import logging
import requests
from typing import Dict, Optional
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


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


def extract_resume_info(url: str) -> Dict[str, str]:
    """
    Extract information from resume URL
    
    Args:
        url: URL of the resume file
        
    Returns:
        Dictionary with extracted information
    """
    try:
        # Read resume content based on file type
        if url.lower().endswith('.docx'):
            text = read_docx_from_url(url)
        elif url.lower().endswith('.pdf'):
            text = read_pdf_from_url(url)
        else:
            logger.warning(f"Unsupported file format: {url}")
            return {
                "email": "",
                "contact_number": "",
                "linkedin_url": "",
                "skills": "",
                "position": ""
            }
        
        if not text:
            return {
                "email": "",
                "contact_number": "",
                "linkedin_url": "",
                "skills": "",
                "position": ""
            }
        
        # Extract information
        return {
            "email": extract_email(text),
            "contact_number": extract_phone(text),
            "linkedin_url": extract_linkedin(text),
            "skills": extract_skills(text),
            "position": extract_position(text)
        }
        
    except Exception as e:
        logger.error(f"Error extracting resume info from {url}: {str(e)}")
        return {
            "email": "",
            "contact_number": "",
            "linkedin_url": "",
            "skills": "",
            "position": ""
        }


