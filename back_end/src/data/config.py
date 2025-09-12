"""
Simplified configuration management for the MyBiome data pipeline.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)


class Config:
    """Simplified configuration class."""
    
    def __init__(self):
        # Database
        self.db_name = 'pubmed_research.db'
        self.db_path = project_root / "data" / "processed" / self.db_name
        self.max_connections = 5  # Reduced from 10
        
        # API settings
        self.ncbi_api_key = os.getenv("NCBI_API_KEY")
        self.email = os.getenv("EMAIL", "samisena@outlook.com")
        self.api_timeout = 30
        self.max_retries = 3
        
        # URLs
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pmc_base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        self.unpaywall_base_url = "https://api.unpaywall.org/v2"
        
        # Rate limiting (simplified)
        self.api_delay = 0.5  # Single delay for all APIs
        
        # LLM settings
        self.llm_base_url = "http://localhost:11434/v1"
        self.llm_model = "llama3.1:8b"
        self.llm_temperature = 0.3
        self.llm_max_tokens = 4096
        
        # Multi-LLM for consensus (simplified)
        self.consensus_models = ["gemma2:9b", "qwen2.5:14b"]
        
        # Paths
        self.project_root = project_root
        self.data_root = project_root / "data"
        self.raw_data = project_root / "data" / "raw"
        self.processed_data = project_root / "data" / "processed"
        self.logs_dir = project_root / "data" / "logs"
        self.papers_dir = project_root / "data" / "raw" / "papers"
        self.metadata_dir = project_root / "data" / "raw" / "metadata"
        self.fulltext_dir = project_root / "data" / "raw" / "fulltext"
        self.pmc_dir = project_root / "data" / "raw" / "fulltext" / "pmc"
        self.pdf_dir = project_root / "data" / "raw" / "fulltext" / "pdf"
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories."""
        directories = [
            self.raw_data, self.processed_data, self.logs_dir,
            self.papers_dir, self.metadata_dir, self.fulltext_dir, self.pmc_dir, self.pdf_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate(self):
        """Simple validation."""
        issues = []
        
        if not self.ncbi_api_key:
            issues.append("Missing NCBI_API_KEY environment variable")
        
        if "@" not in self.email:
            issues.append("Invalid email format")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }


def setup_logging(name: str, log_file: str = None, level: int = logging.INFO):
    """Simplified logging setup."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = config.logs_dir / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global configuration instance
config = Config()

# Backward compatibility classes
class LLMConfig:
    """Simple LLM configuration for backward compatibility."""
    def __init__(self, model_name=None, base_url=None, temperature=0.3, max_tokens=4096, api_key="not-needed"):
        self.model_name = model_name or config.llm_model
        self.base_url = base_url or config.llm_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key


class MultiLLMConfig:
    """Simple multi-LLM configuration for backward compatibility."""
    def __init__(self):
        self.models = [
            LLMConfig(model_name=model) 
            for model in config.consensus_models
        ]
        self.consensus_threshold = 1.0
        self.conflict_resolution = "manual_review"


class DatabaseConfig:
    """Simple database configuration for backward compatibility."""
    def __init__(self):
        self.name = config.db_name
        self.path = config.db_path
        self.max_connections = config.max_connections


class APIConfig:
    """Simple API configuration for backward compatibility."""
    def __init__(self):
        self.ncbi_api_key = config.ncbi_api_key
        self.email = config.email
        self.pubmed_base_url = config.pubmed_base_url
        self.pmc_base_url = config.pmc_base_url
        self.unpaywall_base_url = config.unpaywall_base_url
        self.pmc_delay = config.api_delay
        self.unpaywall_delay = config.api_delay
        self.api_timeout = config.api_timeout
        self.max_retries = config.max_retries


class PathConfig:
    """Simple path configuration for backward compatibility."""
    def __init__(self):
        self.project_root = config.project_root
        self.data_root = config.data_root
        self.raw_data = config.raw_data
        self.processed_data = config.processed_data
        self.papers_dir = config.papers_dir
        self.metadata_dir = config.metadata_dir
        self.fulltext_dir = config.fulltext_dir
        self.pmc_dir = config.pmc_dir
        self.pdf_dir = config.pdf_dir
        self.logs_dir = config.logs_dir


# Add backward compatibility attributes to config
config.database = DatabaseConfig()
config.api = APIConfig()
config.paths = PathConfig()
config.llm = LLMConfig()
config.multi_llm = MultiLLMConfig()


# Method for backward compatibility
def get_llm_config(model_name=None, base_url=None):
    """Get LLM configuration with optional overrides."""
    return LLMConfig(model_name=model_name, base_url=base_url)

config.get_llm_config = get_llm_config