"""
Unified configuration management for the MyBiome intervention research platform.
Combines simple and complex configuration approaches with backward compatibility.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Configuration is standalone - no sys.path modifications needed

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)


class UnifiedConfig:
    """Unified configuration class that handles all system settings."""
    
    def __init__(self):
        # Core paths
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
        
        # Database configuration
        self.db_name = 'intervention_research.db'
        self.db_path = self.processed_data / self.db_name
        self.max_connections = 5  # Simplified for SQLite
        
        # API configuration
        self.ncbi_api_key = os.getenv("NCBI_API_KEY")
        self.email = os.getenv("EMAIL", "samisena@outlook.com")
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pmc_base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        self.unpaywall_base_url = "https://api.unpaywall.org/v2"
        self.api_timeout = 30
        self.max_retries = 3
        self.api_delay = 0.5  # Unified delay for all APIs
        
        # LLM configuration - simplified dual model setup
        self.llm_base_url = "http://localhost:11434/v1"
        self.llm_temperature = 0.3
        self.llm_max_tokens = 4096
        
        # Dual model configuration
        self.dual_models = ["gemma2:9b", "qwen2.5:14b"]
        
        # Intervention-specific settings
        self.intervention_categories = [
            "exercise", "diet", "supplement", "medication", "therapy", "lifestyle"
        ]
        self.intervention_batch_size = 5
        self.max_papers_per_condition = 100
        
        # Create directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories."""
        directories = [
            self.raw_data, self.processed_data, self.logs_dir,
            self.papers_dir, self.metadata_dir, self.fulltext_dir, 
            self.pmc_dir, self.pdf_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_llm_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration for a specific model."""
        return {
            'base_url': self.llm_base_url,
            'model_name': model_name or "gemma2:9b",
            'temperature': self.llm_temperature,
            'max_tokens': self.llm_max_tokens,
            'api_key': "not-needed"  # For local Ollama
        }
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration."""
        issues = []
        warnings = []
        
        # Check API key
        if not self.ncbi_api_key:
            warnings.append("Missing NCBI_API_KEY - API rate limits will apply")
        
        # Check email format
        if "@" not in self.email:
            issues.append("Invalid email format")
        
        # Check directory permissions
        try:
            test_file = self.processed_data / ".test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            issues.append(f"Cannot write to processed data directory: {e}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

    # Backward compatibility properties
    @property
    def database(self):
        """Backward compatibility for database config."""
        class DatabaseCompat:
            def __init__(self, config):
                self.name = config.db_name
                self.path = config.db_path
                self.max_connections = config.max_connections
        return DatabaseCompat(self)
    
    @property
    def api(self):
        """Backward compatibility for API config."""
        class APICompat:
            def __init__(self, config):
                self.ncbi_api_key = config.ncbi_api_key
                self.email = config.email
                self.pubmed_base_url = config.pubmed_base_url
                self.pmc_base_url = config.pmc_base_url
                self.unpaywall_base_url = config.unpaywall_base_url
                self.pmc_delay = config.api_delay
                self.unpaywall_delay = config.api_delay
                self.api_timeout = config.api_timeout
                self.max_retries = config.max_retries
        return APICompat(self)
    
    @property
    def paths(self):
        """Backward compatibility for paths config."""
        class PathsCompat:
            def __init__(self, config):
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
        return PathsCompat(self)
    
    @property
    def llm(self):
        """Backward compatibility for LLM config."""
        return self.get_llm_config()


def setup_logging(name: str, log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """Centralized logging setup."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = config.logs_dir / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global unified configuration instance
config = UnifiedConfig()

# Backward compatibility exports
class LLMConfig:
    """Backward compatibility class."""
    def __init__(self, model_name=None, base_url=None, temperature=0.3, max_tokens=4096, api_key="not-needed"):
        self.model_name = model_name or config.dual_models[0]
        self.base_url = base_url or config.llm_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

# Add backward compatibility method - fix recursion by calling the actual method
def get_llm_config_compat(model_name=None, base_url=None):
    """Backward compatibility wrapper for get_llm_config."""
    return LLMConfig(model_name=model_name, base_url=base_url)

config.get_llm_config = get_llm_config_compat