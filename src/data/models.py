"""
Data models for the PubMed collection system.
These are shared across all other modules.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass 
class Paper:
    """Represents a research paper"""
    pmid: str
    title: str
    abstract: str
    journal: str
    publication_date: str
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None