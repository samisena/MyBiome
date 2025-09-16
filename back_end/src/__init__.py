"""
MyBiome data pipeline package.
"""
__version__ = "3.0.0"

# Core configuration
from .data.config import config, setup_logging

# Main pipeline classes
from .llm.pipeline import ResearchPipeline
# from .data.enhanced_resumable_pipeline import EnhancedResumablePipeline  # File not found

# Database and data management
from .paper_collection.database_manager import DatabaseManager, database_manager

# Processing modules
from .paper_collection.pubmed_collector import PubMedCollector
# from .llm.probiotic_analyzer import ProbioticAnalyzer  # Module removed
from .paper_collection.paper_parser import PubmedParser
from .paper_collection.fulltext_retriever import FullTextRetriever

# Utilities
from .data.utils import (
    parse_json_safely, batch_process
)
from .data.validators import validation_manager
from .data.error_handler import error_handler

__all__ = [
    'config', 'setup_logging',
    'ResearchPipeline',
    'DatabaseManager', 'database_manager',
    'PubMedCollector',
    'PubmedParser', 'FullTextRetriever',
    'parse_json_safely', 'batch_process', 'validation_manager', 'error_handler'
]