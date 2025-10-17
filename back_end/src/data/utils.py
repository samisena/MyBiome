"""
Utility functions and decorators for the MyBiome data pipeline.
This module contains common patterns abstracted from the main modules.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from pathlib import Path

try:
    from .config import setup_logging
except ImportError:
    from back_end.src.data.config import setup_logging

F = TypeVar('F', bound=Callable[..., Any])


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

        # Remove qwen3 <think> tags (chain-of-thought reasoning)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

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


# Removed safe_file_write - basic error handling is covered by error_handler.py


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


def normalize_string(
    text: str,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    remove_extra_spaces: bool = False,
    min_length: int = 0,
    max_length: Optional[int] = None
) -> Optional[str]:
    """
    Normalize a string with common transformations.

    This utility eliminates duplicate string normalization patterns
    found across validators, processors, and clustering code.

    Args:
        text: Input string to normalize
        lowercase: Convert to lowercase (default: True)
        strip_whitespace: Strip leading/trailing whitespace (default: True)
        remove_extra_spaces: Collapse multiple spaces to single space (default: False)
        min_length: Minimum length (return None if shorter, default: 0)
        max_length: Maximum length (truncate if longer, default: None)

    Returns:
        Normalized string, or None if validation fails

    Examples:
        >>> normalize_string("  Hello World  ")
        'hello world'

        >>> normalize_string("Test", min_length=5)
        None

        >>> normalize_string("  Multiple   Spaces  ", remove_extra_spaces=True)
        'multiple spaces'
    """
    if not text or not isinstance(text, str):
        return None

    # Strip whitespace
    if strip_whitespace:
        text = text.strip()

    # Check if empty after stripping
    if not text:
        return None

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove extra spaces
    if remove_extra_spaces:
        text = ' '.join(text.split())

    # Check minimum length
    if min_length > 0 and len(text) < min_length:
        return None

    # Truncate to maximum length
    if max_length is not None and len(text) > max_length:
        text = text[:max_length]

    return text


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