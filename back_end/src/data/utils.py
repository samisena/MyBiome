"""
Utility functions and decorators for the MyBiome data pipeline.
This module contains common patterns abstracted from the main modules.
"""

import time
import json
import functools
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from pathlib import Path
import sys

from src.data.config import setup_logging

F = TypeVar('F', bound=Callable[..., Any])


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      backoff_factor: float = 2.0, 
                      exceptions: tuple = (Exception,)) -> Callable[[F], F]:
    """
    Decorator that retries function calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
                        raise
                    
                    delay = base_delay * (backoff_factor ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
        return wrapper
    return decorator


def rate_limit(delay: float) -> Callable[[F], F]:
    """
    Decorator that adds rate limiting to function calls.
    
    Args:
        delay: Minimum delay between calls in seconds
    """
    last_call_time = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_id = id(func)
            current_time = time.time()
            
            if func_id in last_call_time:
                elapsed = current_time - last_call_time[func_id]
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            
            last_call_time[func_id] = time.time()
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


def log_execution_time(func: F) -> F:
    """Decorator that logs function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
            
    return wrapper


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_paper_data(paper: Dict) -> Dict:
    """
    Validate and clean paper data structure.
    
    Args:
        paper: Dictionary containing paper information
        
    Returns:
        Validated paper dictionary
        
    Raises:
        ValidationError: If required fields are missing or invalid
    """
    required_fields = ['pmid', 'title']
    
    for field in required_fields:
        if field not in paper or not paper[field]:
            raise ValidationError(f"Missing required field: {field}")
    
    # Clean and validate data
    validated = {
        'pmid': str(paper['pmid']).strip(),
        'title': str(paper['title']).strip(),
        'abstract': paper.get('abstract', '').strip(),
        'journal': paper.get('journal', 'Unknown journal').strip(),
        'publication_date': paper.get('publication_date', '').strip(),
        'doi': paper.get('doi', '').strip() if paper.get('doi') else None,
        'pmc_id': paper.get('pmc_id', '').strip() if paper.get('pmc_id') else None,
        'keywords': paper.get('keywords') if isinstance(paper.get('keywords'), list) else None,
        'has_fulltext': bool(paper.get('has_fulltext', False)),
        'fulltext_source': paper.get('fulltext_source'),
        'fulltext_path': paper.get('fulltext_path')
    }
    
    return validated


def validate_correlation_data(correlation: Dict) -> Dict:
    """
    Validate and clean correlation data structure.
    
    Args:
        correlation: Dictionary containing correlation information
        
    Returns:
        Validated correlation dictionary
        
    Raises:
        ValidationError: If required fields are missing or invalid
    """
    required_fields = ['paper_id', 'probiotic_strain', 'health_condition', 
                      'correlation_type', 'extraction_model']
    
    for field in required_fields:
        if field not in correlation or not correlation[field]:
            raise ValidationError(f"Missing required field: {field}")
    
    # Check for placeholder values that should not be saved to database
    placeholder_patterns = ['...', 'N/A', 'n/a', 'NA', 'na', 'null', 'NULL', 
                           'unknown', 'Unknown', 'UNKNOWN', 'placeholder', 
                           'Placeholder', 'PLACEHOLDER', 'TBD', 'tbd', 'TODO', 'todo',
                           'probiotics', 'Probiotics', 'PROBIOTICS', 
                           'various strains', 'multiple strains', 'various probiotics',
                           'multiple probiotics']
    
    probiotic_strain = str(correlation['probiotic_strain']).strip()
    health_condition = str(correlation['health_condition']).strip()
    
    # Check if probiotic strain is a placeholder
    if probiotic_strain in placeholder_patterns or len(probiotic_strain) < 3:
        raise ValidationError(f"Invalid probiotic strain placeholder: '{probiotic_strain}'")
    
    # Check if health condition is a placeholder  
    if health_condition in placeholder_patterns or len(health_condition) < 3:
        raise ValidationError(f"Invalid health condition placeholder: '{health_condition}'")
    
    # Additional checks for common placeholder patterns
    if probiotic_strain.lower().startswith(('unknown', 'placeholder', 'various', 'multiple')):
        raise ValidationError(f"Probiotic strain appears to be a placeholder: '{probiotic_strain}'")
    
    if health_condition.lower().startswith(('unknown', 'placeholder', 'various', 'multiple')):
        raise ValidationError(f"Health condition appears to be a placeholder: '{health_condition}'")
    
    # Validate correlation type
    valid_types = ['positive', 'negative', 'neutral', 'inconclusive']
    if correlation['correlation_type'] not in valid_types:
        raise ValidationError(f"Invalid correlation type: {correlation['correlation_type']}")
    
    # Validate numeric fields
    for field in ['correlation_strength', 'confidence_score', 'verification_confidence']:
        if field in correlation and correlation[field] is not None:
            try:
                value = float(correlation[field])
                if not 0.0 <= value <= 1.0:
                    raise ValidationError(f"{field} must be between 0.0 and 1.0")
                correlation[field] = value
            except (ValueError, TypeError):
                raise ValidationError(f"{field} must be a number between 0.0 and 1.0")
    
    if 'sample_size' in correlation and correlation['sample_size'] is not None:
        try:
            correlation['sample_size'] = int(correlation['sample_size'])
            if correlation['sample_size'] < 0:
                raise ValidationError("sample_size must be non-negative")
        except (ValueError, TypeError):
            raise ValidationError("sample_size must be an integer")
    
    return correlation


def parse_json_safely(text: str, paper_id: str = "unknown") -> List[Dict]:
    """
    Safely parse JSON from LLM output with robust error handling.
    
    Args:
        text: JSON text to parse
        paper_id: Paper ID for error reporting
        
    Returns:
        List of parsed objects
    """
    logger = setup_logging(__name__)
    
    try:
        # Clean the text
        text = text.strip()
        
        # Handle code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        text = text.strip()
        
        if not text:
            return []
        
        # Try standard parsing first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
            else:
                logger.warning(f"Unexpected JSON type for {paper_id}: {type(result)}")
                return []
        except json.JSONDecodeError:
            # Try repair strategies
            return _repair_json(text, paper_id)
            
    except Exception as e:
        logger.error(f"Error parsing JSON for {paper_id}: {e}")
        return []


def _repair_json(text: str, paper_id: str) -> List[Dict]:
    """Attempt to repair malformed JSON."""
    import re
    logger = setup_logging(__name__)
    
    # Strategy 1: Complete truncated JSON
    if text.startswith('[') and not text.endswith(']'):
        open_braces = text.count('{') - text.count('}')
        if open_braces > 0:
            text += '}' * open_braces
        text += ']'
        
        try:
            result = json.loads(text)
            if isinstance(result, list):
                logger.info(f"Repaired truncated JSON for {paper_id}")
                return result
        except:
            pass
    
    # Strategy 2: Extract individual objects
    try:
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(object_pattern, text, re.DOTALL)
        
        objects = []
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    objects.append(obj)
            except:
                continue
        
        if objects:
            logger.info(f"Extracted {len(objects)} objects from malformed JSON for {paper_id}")
            return objects
    except:
        pass
    
    # Strategy 3: Fix common formatting issues
    try:
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Ensure array brackets
        if not text.strip().startswith('['):
            text = '[' + text
        if not text.strip().endswith(']'):
            text = text + ']'
        
        result = json.loads(text)
        if isinstance(result, list):
            logger.info(f"Fixed formatting issues for {paper_id}")
            return result
    except:
        pass
    
    logger.error(f"All JSON repair strategies failed for {paper_id}")
    return []


def batch_process(items: List[Any], batch_size: int = 10) -> List[List[Any]]:
    """
    Split a list into batches of specified size.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def safe_file_write(file_path: Union[str, Path], content: str, 
                   encoding: str = 'utf-8') -> bool:
    """
    Safely write content to file with error handling.
    
    Args:
        file_path: Path to write to
        content: Content to write
        encoding: File encoding
        
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logging(__name__)
    
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        logger.debug(f"Successfully wrote to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to {file_path}: {e}")
        return False


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def calculate_success_rate(successful: int, total: int) -> float:
    """Calculate success rate as percentage."""
    if total == 0:
        return 0.0
    return (successful / total) * 100.0


def read_fulltext_content(fulltext_path: str) -> Optional[str]:
    """
    Read full-text content from XML or PDF files.
    
    Args:
        fulltext_path: Path to the full-text file
        
    Returns:
        Extracted text content or None if failed
    """
    if not fulltext_path or not Path(fulltext_path).exists():
        return None
    
    try:
        file_path = Path(fulltext_path)
        
        if file_path.suffix.lower() == '.xml':
            # Parse PMC XML content
            return _extract_text_from_pmc_xml(file_path)
        elif file_path.suffix.lower() == '.pdf':
            # For PDF, would need PyPDF2 or similar - placeholder for now
            return f"[PDF content from {file_path.name} - PDF parsing not implemented yet]"
        else:
            # Try reading as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        logger = setup_logging(__name__)
        logger.error(f"Error reading fulltext from {fulltext_path}: {e}")
        return None


def _extract_text_from_pmc_xml(xml_path: Path) -> str:
    """Extract readable text from PMC XML format."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract title
        title_elem = root.find('.//article-title')
        title = title_elem.text if title_elem is not None else ""
        
        # Extract abstract
        abstract_parts = []
        for abstract in root.findall('.//abstract'):
            for p in abstract.findall('.//p'):
                if p.text:
                    abstract_parts.append(p.text)
        abstract = " ".join(abstract_parts)
        
        # Extract body text
        body_parts = []
        for body in root.findall('.//body'):
            for p in body.findall('.//p'):
                if p.text:
                    body_parts.append(p.text)
        body_text = " ".join(body_parts)
        
        # Combine sections
        full_text = f"Title: {title}\n\nAbstract: {abstract}\n\nBody: {body_text}"
        return full_text.strip()
        
    except Exception:
        # Fallback to raw XML content
        with open(xml_path, 'r', encoding='utf-8') as f:
            return f.read()