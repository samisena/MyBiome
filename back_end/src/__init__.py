"""
MyBiome data pipeline package.
"""
__version__ = "3.0.0"

# Core configuration
from .data.config import config, setup_logging

# Main pipeline classes
from .data.enhanced_pipeline import EnhancedResearchPipeline
from .data.enhanced_resumable_pipeline import EnhancedResumablePipeline

# Database and data management
from .data.database_manager_enhanced import EnhancedDatabaseManager, database_manager
from .data.api_clients import APIClientManager, client_manager

# Processing modules
from .data.pubmed_collector_enhanced import EnhancedPubMedCollector
from .data.probiotic_analyzer_enhanced import EnhancedProbioticAnalyzer
from .data.paper_parser_enhanced import EnhancedPubmedParser
from .data.fulltext_retriever_enhanced import EnhancedFullTextRetriever

# Utilities
from .data.utils import (
    ValidationError, validate_paper_data, validate_correlation_data,
    log_execution_time, retry_with_backoff, rate_limit, 
    parse_json_safely, batch_process, safe_file_write
)

__all__ = [
    'config', 'setup_logging',
    'EnhancedResearchPipeline', 'EnhancedResumablePipeline',
    'EnhancedDatabaseManager', 'database_manager',
    'APIClientManager', 'client_manager',
    'EnhancedPubMedCollector', 'EnhancedProbioticAnalyzer',
    'EnhancedPubmedParser', 'EnhancedFullTextRetriever',
    'ValidationError', 'validate_paper_data', 'validate_correlation_data',
    'log_execution_time', 'retry_with_backoff', 'rate_limit',
    'parse_json_safely', 'batch_process', 'safe_file_write'
]