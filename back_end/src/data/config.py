"""
Centralized configuration management for the MyBiome data pipeline.
This module handles all environment variables, paths, and API configurations.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables once at module level
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    name: str = 'pubmed_research.db'
    path: Optional[Path] = None
    max_connections: int = 10
    
    def __post_init__(self):
        if self.path is None:
            self.path = project_root / "data" / "processed" / self.name


@dataclass  
class APIConfig:
    """API configuration settings."""
    ncbi_api_key: Optional[str] = None
    email: str = "samisena@outlook" \
    ".com"
    pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    pmc_base_url: str = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    unpaywall_base_url: str = "https://api.unpaywall.org/v2"
    
    # Rate limiting settings
    pmc_delay: float = 0.5
    unpaywall_delay: float = 1.0
    api_timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        self.ncbi_api_key = os.getenv("NCBI_API_KEY")
        email_env = os.getenv("EMAIL")
        if email_env:
            self.email = email_env


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    base_url: str = "http://localhost:11434/v1"
    model_name: str = "llama3.1:8b"
    temperature: float = 0.3
    max_tokens: int = 4096
    api_key: str = "not-needed"  # For local Ollama


@dataclass
class PathConfig:
    """File system path configuration."""
    project_root: Path = project_root
    data_root: Path = project_root / "data"
    raw_data: Path = project_root / "data" / "raw"
    processed_data: Path = project_root / "data" / "processed"
    papers_dir: Path = project_root / "data" / "raw" / "papers"
    metadata_dir: Path = project_root / "data" / "raw" / "metadata"
    fulltext_dir: Path = project_root / "data" / "raw" / "fulltext"
    pmc_dir: Path = project_root / "data" / "raw" / "fulltext" / "pmc"
    pdf_dir: Path = project_root / "data" / "raw" / "fulltext" / "pdf"
    logs_dir: Path = project_root / "data" / "logs"


class Config:
    """Main configuration class that aggregates all settings."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.paths = PathConfig()
        self.llm = LLMConfig()
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all necessary directories."""
        directories = [
            self.paths.raw_data,
            self.paths.processed_data,
            self.paths.papers_dir,
            self.paths.metadata_dir,
            self.paths.fulltext_dir,
            self.paths.pmc_dir,
            self.paths.pdf_dir,
            self.paths.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_llm_config(self, model_name: Optional[str] = None, 
                      base_url: Optional[str] = None) -> LLMConfig:
        """Get LLM configuration with optional overrides."""
        config = LLMConfig(
            base_url=base_url or self.llm.base_url,
            model_name=model_name or self.llm.model_name,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            api_key=self.llm.api_key
        )
        return config
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        issues = []
        warnings = []
        
        # Check required API keys
        if not self.api.ncbi_api_key:
            issues.append("Missing NCBI_API_KEY environment variable")
        
        # Check directory permissions
        try:
            test_file = self.paths.processed_data / ".test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            issues.append(f"Cannot write to processed data directory: {e}")
        
        # Check email format
        if "@" not in self.api.email:
            warnings.append("Email format may be invalid")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }


def setup_logging(name: str, log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """Centralized logging setup."""
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        config = Config()
        log_path = config.paths.logs_dir / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Import management functionality
def ensure_project_in_path():
    """Ensure the project root is in sys.path for imports."""
    import sys
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def get_module_path(*components):
    """Get a module path relative to the project root."""
    return ".".join(["back_end", "src"] + list(components))


# Global configuration instance
config = Config()