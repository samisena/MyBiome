"""
MyBiome data pipeline package.
"""
__version__ = "3.0.0"

# Core configuration
from .data.config import config, setup_logging

# Database and data management
try:
    from .data_collection.database_manager import DatabaseManager, database_manager
except ImportError:
    # Fallback for older structure
    try:
        from .paper_collection.database_manager import DatabaseManager, database_manager
    except ImportError:
        DatabaseManager = None
        database_manager = None

# Processing modules
try:
    from .data_collection.pubmed_collector import PubMedCollector
except ImportError:
    try:
        from .paper_collection.pubmed_collector import PubMedCollector
    except ImportError:
        PubMedCollector = None

try:
    from .data_collection.paper_parser import PubmedParser
except ImportError:
    try:
        from .paper_collection.paper_parser import PubmedParser
    except ImportError:
        PubmedParser = None

try:
    from .data_collection.fulltext_retriever import FullTextRetriever
except ImportError:
    try:
        from .paper_collection.fulltext_retriever import FullTextRetriever
    except ImportError:
        FullTextRetriever = None

# Utilities
from .data.utils import (
    parse_json_safely, batch_process
)
from .data.validators import validation_manager
from .data.error_handler import error_handler

__all__ = [
    'config', 'setup_logging',
    'DatabaseManager', 'database_manager',
    'PubMedCollector',
    'PubmedParser', 'FullTextRetriever',
    'parse_json_safely', 'batch_process', 'validation_manager', 'error_handler'
]