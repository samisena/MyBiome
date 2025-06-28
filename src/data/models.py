"""
Data models for the PubMed collection system.
These are shared across all other modules.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass #* generates several methods for the class - including __init__()
class Author:
    """Represents a paper author"""
    last_name: str   #* Type annotations - data type of inputs
    first_name: str
    initials: str
    affiliations: Optional[str] = None

@dataclass 
class Paper:
    """Represents a research paper"""
    pmid: str
    title: str
    abstract: str
    authors: List[Author]
    journal: str
    publication_date: str
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None