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
from .llm.probiotic_analyzer import ProbioticAnalyzer
from .paper_collection.paper_parser import PubmedParser
from .paper_collection.fulltext_retriever import FullTextRetriever

# Utilities
from .data.utils import (
    ValidationError, validate_paper_data, validate_correlation_data,
    log_execution_time, retry_with_backoff, rate_limit, 
    parse_json_safely, batch_process, safe_file_write
)

__all__ = [
    'config', 'setup_logging',
    'ResearchPipeline',
    'DatabaseManager', 'database_manager',
    'PubMedCollector', 'ProbioticAnalyzer',
    'PubmedParser', 'FullTextRetriever',
    'ValidationError', 'validate_paper_data', 'validate_correlation_data',
    'log_execution_time', 'retry_with_backoff', 'rate_limit',
    'parse_json_safely', 'batch_process', 'safe_file_write'
]