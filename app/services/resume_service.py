import os
import logging
from typing import Dict, List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from app.services.resume_extractor import extract_resume_info

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global variables for connections (initialized once)
_pc = None
_index = None
_openai_client = None

# Configuration
PINECONE_INDEX_NAME = "resume-db"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1024


def _initialize_connections():
    """Initialize Pinecone and OpenAI connections"""
    global _pc, _index, _openai_client
    
    if _index is not None:
        return  # Already initialized
    
    # Load API keys
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    logger.info("API keys loaded successfully")
    
    # Initialize Pinecone client
    _pc = Pinecone(api_key=pinecone_api_key)
    logger.info("Pinecone client initialized")
    
    # Initialize OpenAI client
    _openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized")
    
    # Connect to or create index
    existing_indexes = [idx.name for idx in _pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating new index: {PINECONE_INDEX_NAME}")
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info(f"Index {PINECONE_INDEX_NAME} created successfully")
    else:
        logger.info(f"Index {PINECONE_INDEX_NAME} already exists")
    
    # Connect to index
    _index = _pc.Index(PINECONE_INDEX_NAME)
    logger.info(f"Connected to index: {PINECONE_INDEX_NAME}")


def _create_embedding(text: str) -> List[float]:
    """Create embedding using OpenAI"""
    global _openai_client
    try:
        response = _openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSION
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        raise


def search_candidates(job_description: str, count: int = 10) -> List[Dict]:
    """
    Search for candidates matching the job description
    
    Args:
        job_description: The job description text
        count: Number of top candidates to return (default: 10)
        
    Returns:
        List of candidate dictionaries with filename, score, url, view_url, and text_length
    """
    global _index
    
    # Initialize connections if not already done
    _initialize_connections()
    
    try:
        logger.info(f"Searching for candidates with count={count}")
        
        # Create embedding for job description
        query_embedding = _create_embedding(job_description)
        logger.info("Job description embedding created")
        
        # Query Pinecone
        results = _index.query(
            vector=query_embedding,
            top_k=count,
            include_metadata=True
        )
        
        candidates = []
        for match in results.matches:
            filename = match.id
            score = float(match.score)
            
            # Get data from Pinecone metadata
            metadata = match.metadata if match.metadata else {}
            url = metadata.get("url", "")
            view_url = metadata.get("view_url", "")
            text_length = metadata.get("text_length", 0)
            
            # Try to get additional info from metadata first
            skills = metadata.get("skills", "")
            linkedin_url = metadata.get("linkedin_url", "")
            email = metadata.get("email", "")
            contact_number = metadata.get("contact_number", "")
            position = metadata.get("position", "")
            
            # If metadata doesn't have the info and URL is available, extract from resume
            if url and not all([skills, linkedin_url, email, contact_number, position]):
                logger.info(f"Extracting info from resume: {filename}")
                extracted_info = extract_resume_info(url)
                
                # Use extracted info if metadata is empty
                skills = skills or extracted_info.get("skills", "")
                linkedin_url = linkedin_url or extracted_info.get("linkedin_url", "")
                email = email or extracted_info.get("email", "")
                contact_number = contact_number or extracted_info.get("contact_number", "")
                position = position or extracted_info.get("position", "")
            
            candidates.append({
                "filename": filename,
                "score": score,
                "url": url,
                "view_url": view_url,
                "text_length": text_length,
                "skills": skills,
                "linkedin_url": linkedin_url,
                "email": email,
                "contact_number": contact_number,
                "position": position
            })
        
        logger.info(f"Found {len(candidates)} candidates")
        return candidates
        
    except Exception as e:
        logger.error(f"Error searching candidates: {str(e)}")
        raise
